import yfinance as yf # Add yfinance in requirements.txt

def fetch_bybit_spot_klines(symbol: str, timeframe: str = INTERVAL, limit: int = 200) -> pd.DataFrame:
    """
    Fetch 15-minute Spot OHLCV data using yfinance (e.g., SOL-USD) to completely bypass 
    exchange geo-blocking on US cloud servers, while keeping Bybit execution active.
    """
    # Map symbol like SOL/USDT to Yahoo Finance ticker SOL-USD
    base = symbol.split('/')[0]
    ticker = f"{base}-USD"
    
    # Fetch 15m data from Yahoo Finance (yf supports up to 60 days of 15m data)
    df = yf.download(ticker, period="5d", interval="15m", progress=False)
    
    if df.empty:
        # Fallback to daily if 15m is restricted
        df = yf.download(ticker, period="60d", interval="1d", progress=False)

    if df.empty:
        raise ValueError(f"Could not fetch data for {ticker} from alternative sources.")

    # Flatten multi-index columns if present in newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    
    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        c_lower = str(col).lower()
        if 'datetime' in c_lower or 'date' in c_lower:
            col_mapping[col] = "datetime"
        elif 'open' in c_lower:
            col_mapping[col] = "open"
        elif 'high' in c_lower:
            col_mapping[col] = "high"
        elif 'low' in c_lower:
            col_mapping[col] = "low"
        elif 'close' in c_lower:
            col_mapping[col] = "close"
        elif 'volume' in c_lower:
            col_mapping[col] = "volume"

    df = df.rename(columns=col_mapping)
    
    # Keep required columns
    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    for rc in required_cols:
        if rc not in df.columns:
            raise KeyError(f"Missing column {rc} in market data.")

    df = df[required_cols].dropna().tail(limit).copy()
    
    df["timestamp"] = pd.to_datetime(df["datetime"]).astype(np.int64) // 10**6
    df["turnover"] = df["close"] * df["volume"]
    
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df
