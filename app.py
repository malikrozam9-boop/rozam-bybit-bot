import os
import time
import hmac
import hashlib
import json
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import ccxt

# ==============================================================================
# 1. PAGE CONFIGURATION & ENVIRONMENT VARIABLES
# ==============================================================================
st.set_page_config(
    page_title="Rozam Bot — Bybit Spot Chandelier Exit (22, 3)",
    layout="wide",
    initial_sidebar_state="expanded",
)

BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "").strip()
BYBIT_BASE_URL = os.environ.get("BYBIT_BASE_URL", "https://api.bybit.com").rstrip("/")

SUPPORTED_COINS = {
    "SOL/USDT": {"symbol": "SOL/USDT", "bybit_symbol": "SOLUSDT", "base": "SOL", "decimals": 2, "qty_decimals": 4},
    "BTC/USDT": {"symbol": "BTC/USDT", "bybit_symbol": "BTCUSDT", "base": "BTC", "decimals": 2, "qty_decimals": 6},
    "ETH/USDT": {"symbol": "ETH/USDT", "bybit_symbol": "ETHUSDT", "base": "ETH", "decimals": 2, "qty_decimals": 5},
    "BNB/USDT": {"symbol": "BNB/USDT", "bybit_symbol": "BNBUSDT", "base": "BNB", "decimals": 2, "qty_decimals": 4},
    "XRP/USDT": {"symbol": "XRP/USDT", "bybit_symbol": "XRPUSDT", "base": "XRP", "decimals": 4, "qty_decimals": 2},
}

CATEGORY = "spot"
INTERVAL = "15m"
ATR_PERIOD = 22
ATR_MULT = 3.0


# ==============================================================================
# 2. MARKET DATA & ORDER EXECUTION
# ==============================================================================
def fetch_bybit_spot_klines(symbol: str, timeframe: str = INTERVAL, limit: int = 200) -> pd.DataFrame:
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        exchange_fallback = ccxt.binance({"enableRateLimit": True})
        ohlcv = exchange_fallback.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    if not ohlcv:
        raise ValueError(f"No kline data returned for {symbol}.")

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["turnover"] = df["close"] * df["volume"]
    df["timestamp"] = df["timestamp"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def place_bybit_spot_order_usdt(
    symbol: str,
    side: str,
    trade_amount_usdt: float,
    last_price: float,
    qty_decimals: int = 4,
) -> dict:
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise PermissionError("Missing BYBIT_API_KEY or BYBIT_API_SECRET in Streamlit Secrets.")

    exchange = ccxt.bybit({
        "apiKey": BYBIT_API_KEY,
        "secret": BYBIT_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })

    if side == "Buy":
        base_qty = trade_amount_usdt / max(last_price, 1e-9)
        response = exchange.create_market_buy_order(symbol, base_qty, params={"marketUnit": "quoteCoin", "qty": f"{trade_amount_usdt:.2f}"})
    else:
        base_qty = trade_amount_usdt / max(last_price, 1e-9)
        response = exchange.create_market_sell_order(symbol, base_qty)

    return {"retCode": 0, "retMsg": "OK", "result": response}


# ==============================================================================
# 3. CHANDELIER EXIT (22, 3) CALCULATION
# ==============================================================================
def calculate_chandelier_exit(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    multiplier: float = ATR_MULT,
    use_close: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    df["atr"] = df["tr"].ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    if use_close:
        highest_val = df["close"].rolling(window=period, min_periods=period).max()
        lowest_val = df["close"].rolling(window=period, min_periods=period).min()
    else:
        highest_val = df["high"].rolling(window=period, min_periods=period).max()
        lowest_val = df["low"].rolling(window=period, min_periods=period).min()

    raw_long_stop = highest_val - (multiplier * df["atr"])
    raw_short_stop = lowest_val + (multiplier * df["atr"])

    n = len(df)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction = np.ones(n, dtype=int)

    closes = df["close"].values
    raw_ls = raw_long_stop.values
    raw_ss = raw_short_stop.values

    for i in range(n):
        if np.isnan(raw_ls[i]) or np.isnan(raw_ss[i]):
            continue

        if i == 0 or np.isnan(long_stop[i - 1]):
            long_stop[i] = raw_ls[i]
            short_stop[i] = raw_ss[i]
            direction[i] = 1
            continue

        prev_long = long_stop[i - 1]
        prev_short = short_stop[i - 1]

        if closes[i - 1] > prev_long:
            long_stop[i] = max(raw_ls[i], prev_long)
        else:
            long_stop[i] = raw_ls[i]

        if closes[i - 1] < prev_short:
            short_stop[i] = min(raw_ss[i], prev_short)
        else:
            short_stop[i] = raw_ss[i]

        if closes[i] > prev_short:
            direction[i] = 1
        elif closes[i] < prev_long:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    df["long_stop"] = long_stop
    df["short_stop"] = short_stop
    df["direction"] = direction
    df["prev_direction"] = df["direction"].shift(1).fillna(1).astype(int)

    df["buy_signal"] = (df["direction"] == 1) & (df["prev_direction"] == -1)
    df["sell_signal"] = (df["direction"] == -1) & (df["prev_direction"] == 1)

    return df


# ==============================================================================
# 4. PLOTLY CHART
# ==============================================================================
def build_plotly_chart(df: pd.DataFrame, pair_label: str, symbol: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=f"{pair_label} (15m)",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        )
    )

    long_line = np.where(df["direction"] == 1, df["long_stop"], np.nan)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=long_line,
            mode="lines",
            name=f"Chandelier Long Stop ({ATR_PERIOD}, {ATR_MULT})",
            line=dict(color="#10b981", width=2),
            connectgaps=False,
        )
    )

    short_line = np.where(df["direction"] == -1, df["short_stop"], np.nan)
    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=short_line,
            mode="lines",
            name=f"Chandelier Short Stop ({ATR_PERIOD}, {ATR_MULT})",
            line=dict(color="#ef4444", width=2),
            connectgaps=False,
        )
    )

    buys = df[df["buy_signal"]]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["datetime"],
                y=buys["low"] * 0.996,
                mode="markers+text",
                name="BUY Signal",
                text=["BUY"] * len(buys),
                textposition="bottom center",
                marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(width=1, color="#ffffff")),
            )
        )

    sells = df[df["sell_signal"]]
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells["datetime"],
                y=sells["high"] * 1.004,
                mode="markers+text",
                name="SELL Signal",
                text=["SELL"] * len(sells),
                textposition="top center",
                marker=dict(symbol="triangle-down", size=14, color="#ef4444", line=dict(width=1, color="#ffffff")),
            )
        )

    fig.update_layout(
        title=f"Bybit Spot {pair_label} ({symbol}) · 15m Timeframe · Chandelier Exit ({ATR_PERIOD}, {ATR_MULT})",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=640,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ==============================================================================
# 5. MAIN APP
# ==============================================================================
def main():
    if "executed_bar_signals" not in st.session_state:
        st.session_state.executed_bar_signals = set()
    if "order_history" not in st.session_state:
        st.session_state.order_history = []
    if "trade_amount_usdt" not in st.session_state:
        st.session_state.trade_amount_usdt = 50.0

    with st.sidebar:
        st.header("Control Panel")

        selected_pair = st.selectbox(
            "1. Select Spot Coin Pair",
            options=list(SUPPORTED_COINS.keys()),
            index=0,
        )
        coin_meta = SUPPORTED_COINS[selected_pair]
        symbol = coin_meta["symbol"]
        base_coin = coin_meta["base"]
        price_decimals = coin_meta["decimals"]
        qty_decimals = coin_meta["qty_decimals"]

        st.divider()

        st.subheader("2. Trade Amount (USDT)")
        slider_usdt = st.slider(
            "Select Trade Size ($10 - $500 USDT)",
            min_value=10.0,
            max_value=500.0,
            value=float(st.session_state.trade_amount_usdt),
            step=5.0,
            format="$%.0f",
        )
        input_usdt = st.number_input(
            "Exact Trade Amount (USDT)",
            min_value=10.0,
            max_value=500.0,
            value=float(slider_usdt),
            step=5.0,
            format="%.2f",
        )
        trade_amount_usdt = float(input_usdt)
        st.session_state.trade_amount_usdt = trade_amount_usdt

        st.divider()

        st.subheader("Strategy & Execution Settings")
        keys_configured = bool(BYBIT_API_KEY and BYBIT_API_SECRET)
        if keys_configured:
            st.success(f"Bybit API Key Active")
        else:
            st.warning("BYBIT_API_KEY / BYBIT_API_SECRET not set.")

        use_close_extremes = st.checkbox("Use Close Price for Extremums", value=True)
        signal_bar_mode = st.selectbox(
            "Signal Evaluation Candle",
            options=["Closed Candle (Confirmed)", "Live Forming Candle"],
            index=0,
        )
        auto_trade_enabled = st.toggle(
            f"Auto-Trade {selected_pair} (${trade_amount_usdt:.0f} USDT)",
            value=False,
        )

        auto_refresh = st.toggle("Auto-Refresh Live Chart", value=True)
        refresh_seconds = st.slider("Refresh Interval (s)", min_value=5, max_value=60, value=15)

    st.title(f"Rozam Bot — {selected_pair} Bybit Spot Chandelier Exit (22, 3)")
    
    try:
        raw_df = fetch_bybit_spot_klines(symbol=symbol, timeframe=INTERVAL, limit=200)
        df = calculate_chandelier_exit(
            raw_df,
            period=ATR_PERIOD,
            multiplier=ATR_MULT,
            use_close=use_close_extremes,
        )
    except Exception as exc:
        st.error(f"Error fetching market data: {exc}")
        return

    eval_row = df.iloc[-2] if signal_bar_mode.startswith("Closed") else df.iloc[-1]
    latest_row = df.iloc[-1]
    current_price = float(latest_row["close"])
    estimated_base_qty = trade_amount_usdt / max(current_price, 1e-9)

    with st.sidebar:
        st.divider()
        st.subheader(f"Manual {selected_pair} Spot Trade")
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button(f"BUY ${trade_amount_usdt:.0f}", use_container_width=True, type="primary"):
                try:
                    res = place_bybit_spot_order_usdt(
                        symbol=symbol,
                        side="Buy",
                        trade_amount_usdt=trade_amount_usdt,
                        last_price=current_price,
                        qty_decimals=qty_decimals,
                    )
                    st.session_state.order_history.insert(0, {"type": "MANUAL_BUY", "response": res})
                    st.toast(f"Bought successfully!")
                except Exception as exc:
                    st.error(f"Manual Buy Failed: {exc}")

        with col_sell:
            if st.button(f"SELL ${trade_amount_usdt:.0f}", use_container_width=True):
                try:
                    res = place_bybit_spot_order_usdt(
                        symbol=symbol,
                        side="Sell",
                        trade_amount_usdt=trade_amount_usdt,
                        last_price=current_price,
                        qty_decimals=qty_decimals,
                    )
                    st.session_state.order_history.insert(0, {"type": "MANUAL_SELL", "response": res})
                    st.toast(f"Sold successfully!")
                except Exception as exc:
                    st.error(f"Manual Sell Failed: {exc}")

    bar_ts = int(eval_row["timestamp"])
    if auto_trade_enabled:
        if eval_row["buy_signal"]:
            sig_key = f"{symbol}_{bar_ts}_BUY"
            if sig_key not in st.session_state.executed_bar_signals:
                try:
                    order_res = place_bybit_spot_order_usdt(
                        symbol=symbol,
                        side="Buy",
                        trade_amount_usdt=trade_amount_usdt,
                        last_price=current_price,
                        qty_decimals=qty_decimals,
                    )
                    st.session_state.executed_bar_signals.add(sig_key)
                    st.session_state.order_history.insert(0, {"type": "AUTO_BUY", "response": order_res})
                except Exception as exc:
                    st.error(f"Auto-Buy Error: {exc}")

        elif eval_row["sell_signal"]:
            sig_key = f"{symbol}_{bar_ts}_SELL"
            if sig_key not in st.session_state.executed_bar_signals:
                try:
                    order_res = place_bybit_spot_order_usdt(
                        symbol=symbol,
                        side="Sell",
                        trade_amount_usdt=trade_amount_usdt,
                        last_price=current_price,
                        qty_decimals=qty_decimals,
                    )
                    st.session_state.executed_bar_signals.add(sig_key)
                    st.session_state.order_history.insert(0, {"type": "AUTO_SELL", "response": order_res})
                except Exception as exc:
                    st.error(f"Auto-Sell Error: {exc}")

    fig = build_plotly_chart(df, pair_label=selected_pair, symbol=symbol)
    st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
