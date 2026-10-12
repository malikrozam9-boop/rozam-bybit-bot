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
    "SOL/USDT": {"symbol": "SOLUSDT", "base": "SOL", "decimals": 2, "qty_decimals": 4},
    "BTC/USDT": {"symbol": "BTCUSDT", "base": "BTC", "decimals": 2, "qty_decimals": 6},
    "ETH/USDT": {"symbol": "ETHUSDT", "base": "ETH", "decimals": 2, "qty_decimals": 5},
    "BNB/USDT": {"symbol": "BNBUSDT", "base": "BNB", "decimals": 2, "qty_decimals": 4},
    "XRP/USDT": {"symbol": "XRPUSDT", "base": "XRP", "decimals": 4, "qty_decimals": 2},
}

CATEGORY = "spot"
INTERVAL = "15"  # 15-minute timeframe
ATR_PERIOD = 22
ATR_MULT = 3.0


# ==============================================================================
# 2. BYBIT V5 REST API HELPERS (MARKET DATA & SPOT ORDER EXECUTION)
# ==============================================================================
def _generate_bybit_signature(timestamp: str, api_key: str, recv_window: str, payload: str, api_secret: str) -> str:
    """Generate HMAC-SHA256 signature for Bybit V5 API."""
    param_str = f"{timestamp}{api_key}{recv_window}{payload}"
    return hmac.new(
        api_secret.encode("utf-8"),
        param_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def fetch_bybit_spot_klines(symbol: str, interval: str = INTERVAL, limit: int = 200) -> pd.DataFrame:
    """
    Fetch 15-minute Spot OHLCV klines from Bybit V5 API (/v5/market/kline) for the selected symbol.
    Returns a chronological DataFrame with columns:
    [timestamp, datetime, open, high, low, close, volume, turnover]
    """
    url = f"{BYBIT_BASE_URL}/v5/market/kline"
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit API Error ({data.get('retCode')}): {data.get('retMsg')}")

    raw_list = data.get("result", {}).get("list", [])
    if not raw_list:
        raise ValueError(f"No kline data returned from Bybit Spot API for {symbol}.")

    df = pd.DataFrame(
        raw_list,
        columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
    )
    df["timestamp"] = df["timestamp"].astype(np.int64)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    # Bybit returns newest first; sort ascending by timestamp
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
    """
    Execute an actual Market order on Bybit Spot (/v5/order/create) dynamically sized by Trade Amount in USDT.
    - For Spot Market BUY: uses marketUnit='quoteCoin' with qty = USDT amount (e.g. '50' for $50 USDT).
    - For Spot Market SELL: converts USDT amount to base coin quantity (trade_amount_usdt / last_price)
      with marketUnit='baseCoin' so the exact USDT-equivalent position is sold.
    """
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise PermissionError("Missing BYBIT_API_KEY or BYBIT_API_SECRET environment variables.")

    endpoint = "/v5/order/create"
    if side == "Buy":
        order_qty_str = f"{trade_amount_usdt:.2f}"
        market_unit = "quoteCoin"
    else:
        base_qty = trade_amount_usdt / max(last_price, 1e-9)
        order_qty_str = f"{base_qty:.{qty_decimals}f}"
        market_unit = "baseCoin"

    payload_dict = {
        "category": "spot",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": order_qty_str,
        "marketUnit": market_unit,
    }
    payload_json = json.dumps(payload_dict, separators=(",", ":"))
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    signature = _generate_bybit_signature(timestamp, BYBIT_API_KEY, recv_window, payload_json, BYBIT_API_SECRET)

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json",
    }

    response = requests.post(f"{BYBIT_BASE_URL}{endpoint}", headers=headers, data=payload_json, timeout=10)
    response.raise_for_status()
    return response.json()


# ==============================================================================
# 3. CHANDELIER EXIT (22, 3) CALCULATION
# ==============================================================================
def calculate_chandelier_exit(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    multiplier: float = ATR_MULT,
    use_close: bool = True,
) -> pd.DataFrame:
    """
    Calculate Chandelier Exit (period=22, multiplier=3.0) matching TradingView standard logic:
      1. True Range & Wilder's RMA ATR(22)
      2. Long Stop = Highest(Close or High, 22) - 3.0 * ATR(22)
      3. Short Stop = Lowest(Close or Low, 22) + 3.0 * ATR(22)
      4. Trailing Ratchet & Direction state machine (+1 Bullish / -1 Bearish)
      5. Buy Signal on flip from -1 to +1; Sell Signal on flip from +1 to -1
    """
    df = df.copy()

    # True Range
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

    # Wilder's Smoothing (RMA) for ATR
    df["atr"] = df["tr"].ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Rolling Highest / Lowest
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
# 4. PLOTLY LIVE CANDLESTICK + CHANDELIER EXIT CHART
# ==============================================================================
def build_plotly_chart(df: pd.DataFrame, pair_label: str, symbol: str) -> go.Figure:
    """Create interactive Plotly Candlestick chart with Chandelier Exit stops & signals."""
    fig = go.Figure()

    # 1. Candlesticks for selected coin
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

    # 2. Active Long Stop (only visible when direction == 1)
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

    # 3. Active Short Stop (only visible when direction == -1)
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

    # 4. Buy Signal Markers
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

    # 5. Sell Signal Markers
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
# 5. STREAMLIT SIDEBAR CONTROL PANEL & DYNAMIC TRADING ENGINE
# ==============================================================================
def main():
    if "executed_bar_signals" not in st.session_state:
        st.session_state.executed_bar_signals = set()
    if "order_history" not in st.session_state:
        st.session_state.order_history = []
    if "trade_amount_usdt" not in st.session_state:
        st.session_state.trade_amount_usdt = 50.0

    # --------------------------------------------------------------------------
    # STREAMLIT SIDEBAR (CONTROL PANEL)
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("Control Panel")

        # 1. Dropdown Menu for Coin Selection (SOL/USDT, BTC/USDT, ETH/USDT, etc.)
        selected_pair = st.selectbox(
            "1. Select Spot Coin Pair",
            options=list(SUPPORTED_COINS.keys()),
            index=0,
            help="Select the Bybit Spot pair. Changing this updates the live 15m chart and Spot orders immediately.",
        )
        coin_meta = SUPPORTED_COINS[selected_pair]
        symbol = coin_meta["symbol"]
        base_coin = coin_meta["base"]
        price_decimals = coin_meta["decimals"]
        qty_decimals = coin_meta["qty_decimals"]

        st.divider()

        # 2. Slider + Number Input for Trade Amount in USDT ($10 to $500)
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

        # Indicator & Auto-Trade Controls
        st.subheader("Strategy & Execution Settings")
        keys_configured = bool(BYBIT_API_KEY and BYBIT_API_SECRET)
        if keys_configured:
            st.success(f"Bybit API Key Active ({BYBIT_API_KEY[:4]}...{BYBIT_API_KEY[-4:]})")
        else:
            st.warning("BYBIT_API_KEY / BYBIT_API_SECRET not set in env.")

        use_close_extremes = st.checkbox("Use Close Price for Extremums", value=True)
        signal_bar_mode = st.selectbox(
            "Signal Evaluation Candle",
            options=["Closed Candle (Confirmed)", "Live Forming Candle"],
            index=0,
        )
        auto_trade_enabled = st.toggle(
            f"Auto-Trade {selected_pair} (${trade_amount_usdt:.0f} USDT)",
            value=False,
            help="Automatically executes a Bybit Spot Market order for the selected coin and USDT amount on Buy/Sell flips.",
        )

        auto_refresh = st.toggle("Auto-Refresh Live Chart", value=True)
        refresh_seconds = st.slider("Refresh Interval (s)", min_value=5, max_value=60, value=15)

    # --------------------------------------------------------------------------
    # FETCH LIVE 15M DATA FOR SELECTED COIN & CALCULATE CHANDELIER EXIT (22, 3)
    # --------------------------------------------------------------------------
    st.title(f"Rozam Bot — {selected_pair} Bybit Spot Chandelier Exit (22, 3)")
    st.caption(
        f"Active Pair: {selected_pair} ({symbol}) · Order Size: ${trade_amount_usdt:,.2f} USDT · "
        f"Timeframe: 15-Minute · Indicator: Chandelier Exit ({ATR_PERIOD}, {ATR_MULT})"
    )

    try:
        raw_df = fetch_bybit_spot_klines(symbol=symbol, interval=INTERVAL, limit=200)
        df = calculate_chandelier_exit(
            raw_df,
            period=ATR_PERIOD,
            multiplier=ATR_MULT,
            use_close=use_close_extremes,
        )
    except Exception as exc:
        st.error(f"Error fetching {selected_pair} data from Bybit Spot API: {exc}")
        return

    eval_row = df.iloc[-2] if signal_bar_mode.startswith("Closed") else df.iloc[-1]
    latest_row = df.iloc[-1]
    current_price = float(latest_row["close"])
    estimated_base_qty = trade_amount_usdt / max(current_price, 1e-9)

    # Sidebar Manual Execution Buttons (now that current_price is known)
    with st.sidebar:
        st.divider()
        st.subheader(f"Manual {selected_pair} Spot Trade")
        st.caption(f"${trade_amount_usdt:,.2f} USDT ≈ {estimated_base_qty:.{qty_decimals}f} {base_coin}")
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
                    st.session_state.order_history.insert(
                        0,
                        {
                            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "pair": selected_pair,
                            "type": "MANUAL_BUY",
                            "side": "Buy",
                            "amount_usdt": trade_amount_usdt,
                            "est_base_qty": round(estimated_base_qty, qty_decimals),
                            "response": res,
                        },
                    )
                    st.toast(f"Bought ${trade_amount_usdt:.0f} of {selected_pair}: {res.get('retMsg')}")
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
                    st.session_state.order_history.insert(
                        0,
                        {
                            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "pair": selected_pair,
                            "type": "MANUAL_SELL",
                            "side": "Sell",
                            "amount_usdt": trade_amount_usdt,
                            "est_base_qty": round(estimated_base_qty, qty_decimals),
                            "response": res,
                        },
                    )
                    st.toast(f"Sold ${trade_amount_usdt:.0f} of {selected_pair}: {res.get('retMsg')}")
                except Exception as exc:
                    st.error(f"Manual Sell Failed: {exc}")

    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{selected_pair} Spot Price", f"${current_price:,.{price_decimals}f}")
    m2.metric(
        "Trade Size (USDT -> Coin)",
        f"${trade_amount_usdt:,.0f} USDT",
        f"≈ {estimated_base_qty:.{qty_decimals}f} {base_coin}",
    )
    m3.metric(
        "Chandelier Trend",
        "BULLISH (LONG)" if latest_row["direction"] == 1 else "BEARISH (SHORT)",
    )
    active_stop = latest_row["long_stop"] if latest_row["direction"] == 1 else latest_row["short_stop"]
    m4.metric("Active Trailing Stop", f"${active_stop:,.{price_decimals}f}")

    latest_sig_text = "HOLD / NO FLIP"
    if eval_row["buy_signal"]:
        latest_sig_text = "BUY SIGNAL"
    elif eval_row["sell_signal"]:
        latest_sig_text = "SELL SIGNAL"
    m5.metric("Evaluation Candle Signal", latest_sig_text)

    # Auto-Trade Execution on Chandelier Exit Signal Flip
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
                    st.session_state.order_history.insert(
                        0,
                        {
                            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "pair": selected_pair,
                            "type": "AUTO_CHANDELIER_BUY",
                            "side": "Buy",
                            "amount_usdt": trade_amount_usdt,
                            "est_base_qty": round(estimated_base_qty, qty_decimals),
                            "bar_time": str(eval_row["datetime"]),
                            "response": order_res,
                        },
                    )
                    st.success(f"Auto-Executed Spot BUY of ${trade_amount_usdt:.2f} USDT on {selected_pair}!")
                except Exception as exc:
                    st.error(f"Auto-Buy Order Execution Error: {exc}")

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
                    st.session_state.order_history.insert(
                        0,
                        {
                            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "pair": selected_pair,
                            "type": "AUTO_CHANDELIER_SELL",
                            "side": "Sell",
                            "amount_usdt": trade_amount_usdt,
                            "est_base_qty": round(estimated_base_qty, qty_decimals),
                            "bar_time": str(eval_row["datetime"]),
                            "response": order_res,
                        },
                    )
                    st.warning(f"Auto-Executed Spot SELL of ${trade_amount_usdt:.2f} USDT on {selected_pair}!")
                except Exception as exc:
                    st.error(f"Auto-Sell Order Execution Error: {exc}")

    # Render Live Plotly Chart for Selected Coin
    fig = build_plotly_chart(df, pair_label=selected_pair, symbol=symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Recent Signals & Order Execution Logs
    col_signals, col_orders = st.columns(2)
    with col_signals:
        st.subheader(f"Recent {selected_pair} Chandelier Exit (22, 3) Signals")
        signals_df = df[df["buy_signal"] | df["sell_signal"]].copy()
        if not signals_df.empty:
            signals_df["Signal"] = np.where(signals_df["buy_signal"], "BUY", "SELL")
            display_cols = signals_df[["datetime", "Signal", "close", "atr", "long_stop", "short_stop"]].tail(10)
            st.dataframe(display_cols.sort_values("datetime", ascending=False), use_container_width=True)
        else:
            st.info(f"No trend-flip signals detected for {selected_pair} in the current 200-candle window.")

    with col_orders:
        st.subheader("Bybit Spot Order Execution Log")
        if st.session_state.order_history:
            st.json(st.session_state.order_history[:10])
        else:
            st.info("No Spot orders executed in this session yet.")

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()
