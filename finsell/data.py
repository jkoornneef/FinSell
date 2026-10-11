"""Data fetching and preparation utilities."""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock/crypto symbol (e.g. "ENB.TO", "BTC-USD")
        period: How far back to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Bar size (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume
    """
    print(f"[DEBUG] Download e : {ticker}")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    print(f"[DEBUG] Download e complete : {ticker}")
    df.columns = df.columns.droplevel("Ticker") if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df


def load_csv(path: str, date_col: str = "Date") -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    return df
