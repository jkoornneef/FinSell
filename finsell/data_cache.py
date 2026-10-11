"""Cached data fetcher for yfinance.

Two tiers:
  - Monthly: 10 years (≈120 data points)
  - Daily:   last 1 year (≈252 data points)

Cache lives in data/cache/<TICKER>/ as parquet files.
Daily tier does incremental fetch (only the gap since last cached date).
Sector/industry ETF data is cached the same way.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_path(ticker: str) -> Path:
    p = CACHE_DIR / ticker.upper()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meta_path(ticker: str) -> Path:
    return _cache_path(ticker) / "meta.json"


def _read_meta(ticker: str) -> dict:
    mp = _meta_path(ticker)
    if mp.exists():
        return json.loads(mp.read_text())
    return {}


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Read a parquet file if it is valid; if not, remove the corrupt cache and return None."""
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _write_meta(ticker: str, meta: dict):
    _meta_path(ticker).write_text(json.dumps(meta, default=str, indent=2))


def _tier_file(ticker: str, tier_name: str) -> Path:
    return _cache_path(ticker) / f"{tier_name}.parquet"


def _get_fetch_window(*, now: datetime | None = None, lookback_days: int) -> tuple[datetime, datetime]:
    """Return a fetch window bounded to the current day, never tomorrow."""
    current = now or datetime.now()
    start = current - timedelta(days=lookback_days)
    end = current.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, end


def _fetch(ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch from yfinance for a date range and interval."""
    print(f"[DEBUG] Download d : {ticker}, interval={interval}, start={start.date()}, end={end.date()}")
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )
    print(f"[DEBUG] Download d complete : {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    return df


def _load_daily(ticker: str, force: bool = False) -> pd.DataFrame:
    """Load/update daily data (last 1 year). Incremental on subsequent calls."""
    today = datetime.now()
    start, end = _get_fetch_window(now=today, lookback_days=365)
    fp = _tier_file(ticker, "daily")
    meta = _read_meta(ticker)

    cached_df = None
    if fp.exists() and not force:
        cached_df = _read_parquet_safe(fp)
        if cached_df is not None:
            cached_df.index = pd.to_datetime(cached_df.index)

    if cached_df is not None and len(cached_df) > 0 and not force:
        last_date = cached_df.index.max()
        gap_start = last_date + timedelta(days=1)
        # If there is a gap between the cached last date and the desired end date,
        # attempt an incremental fetch — but avoid repeatedly retrying when the
        # market is closed (weekends, holidays) by recording when we last checked.
        new_data_appended = False
        if gap_start < end:
            now = datetime.now()
            threshold = timedelta(hours=12)
            last_checked_iso = meta.get("daily", {}).get("last_checked")
            last_checked = None
            if last_checked_iso:
                try:
                    last_checked = datetime.fromisoformat(last_checked_iso)
                except Exception:
                    last_checked = None

            should_fetch = last_checked is None or (now - last_checked) > threshold
            if should_fetch:
                new_df = _fetch(ticker, "1d", gap_start, end)
                if len(new_df) > 0:
                    orig_len = len(cached_df)
                    combined = pd.concat([cached_df, new_df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.sort_index(inplace=True)
                    # Only treat it as new data if the combined frame grew.
                    if len(combined) > orig_len:
                        cached_df = combined
                        new_data_appended = True
                    else:
                        # Received overlapping rows but no net new rows;
                        # record last_checked to avoid immediate refetches.
                        meta_daily = meta.get("daily", {})
                        meta_daily["last_checked"] = now.isoformat()
                        meta["daily"] = meta_daily
                        _write_meta(ticker, meta)
                else:
                    # No rows returned — record last_checked to avoid
                    # refetching repeatedly during closed markets.
                    meta_daily = meta.get("daily", {})
                    meta_daily["last_checked"] = now.isoformat()
                    meta["daily"] = meta_daily
                    _write_meta(ticker, meta)

        # Trim to 1-year window and write only if content changed.
        trimmed = cached_df[cached_df.index >= pd.Timestamp(start)]
        if new_data_appended or len(trimmed) != len(cached_df):
            trimmed.to_parquet(fp)
            meta_daily = meta.get("daily", {})
            meta_daily["last_updated"] = datetime.now().isoformat()
            if new_data_appended and "last_checked" in meta_daily:
                meta_daily.pop("last_checked", None)
            meta["daily"] = meta_daily
            _write_meta(ticker, meta)
        return trimmed

    # Full fetch
    df = _fetch(ticker, "1d", start, end)
    if len(df) > 0:
        df.to_parquet(fp)
        meta["daily"] = {"last_updated": datetime.now().isoformat()}
        _write_meta(ticker, meta)
    return df


def _load_monthly(ticker: str, force: bool = False) -> pd.DataFrame:
    """Load monthly data (10 years). Re-fetches if cache older than 7 days."""
    today = datetime.now()
    start, end = _get_fetch_window(now=today, lookback_days=3650)
    fp = _tier_file(ticker, "monthly")
    meta = _read_meta(ticker)

    if fp.exists() and not force:
        cached_df = _read_parquet_safe(fp)
        if cached_df is None:
            return _load_monthly(ticker, force=True)
        cached_df.index = pd.to_datetime(cached_df.index)
        last_updated = meta.get("monthly", {}).get("last_updated")
        if last_updated:
            lu = datetime.fromisoformat(last_updated)
            if datetime.now() - lu < timedelta(days=7):
                return cached_df

    df = _fetch(ticker, "1mo", start, end)
    if len(df) > 0:
        df.to_parquet(fp)
        meta["monthly"] = {"last_updated": datetime.now().isoformat()}
        _write_meta(ticker, meta)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_history(ticker: str, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch cached history for a ticker.

    Returns:
        (monthly_df, daily_df) — both have OHLCV columns with DatetimeIndex.
        monthly_df: ~10 years, daily_df: ~1 year.
    """
    monthly = _load_monthly(ticker, force=force)
    daily = _load_daily(ticker, force=force)
    return monthly, daily


def fetch_combined(ticker: str, force: bool = False) -> pd.DataFrame:
    """Fetch and merge into a single series: monthly for >1yr ago, daily for last year.

    Returns:
        DataFrame sorted by date with a '_resolution' column ('monthly' or 'daily').
    """
    monthly, daily = fetch_history(ticker, force=force)
    cutoff = datetime.now() - timedelta(days=365)

    if len(monthly) > 0:
        monthly = monthly[monthly.index < pd.Timestamp(cutoff)].copy()
        monthly["_resolution"] = "monthly"

    if len(daily) > 0:
        daily = daily.copy()
        daily["_resolution"] = "daily"

    frames = [df for df in [monthly, daily] if len(df) > 0]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames)
    combined.sort_index(inplace=True)
    return combined


def get_ticker_info(ticker: str, force: bool = False) -> dict:
    """Get ticker info (fundamentals, industry, sector, etc.), cached for 30 days."""
    meta = _read_meta(ticker)
    info = meta.get("info")

    if info and meta.get("info_fetched") and not force:
        fetched = datetime.fromisoformat(meta["info_fetched"])
        if datetime.now() - fetched < timedelta(days=30):
            return info

    try:
        t = yf.Ticker(ticker)
        ti = t.info
        info = {
            # Identity
            "name": ti.get("shortName", ticker),
            "industry": ti.get("industry", "Unknown"),
            "sector": ti.get("sector", "Unknown"),
            "currency": ti.get("currency", "USD"),
            "sector_etf": _sector_to_etf(ti.get("sector", "")),
            # Valuation
            "trailingPE": ti.get("trailingPE"),
            "forwardPE": ti.get("forwardPE"),
            "priceToBook": ti.get("priceToBook"),
            "marketCap": ti.get("marketCap"),
            # Risk / volatility
            "beta": ti.get("beta"),
            "fiftyTwoWeekHigh": ti.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": ti.get("fiftyTwoWeekLow"),
            # Growth
            "earningsGrowth": ti.get("earningsGrowth"),
            "revenueGrowth": ti.get("revenueGrowth"),
            # Dividend
            "dividendYield": ti.get("dividendYield"),
            "trailingAnnualDividendRate": ti.get("trailingAnnualDividendRate"),
            # Volume
            "averageVolume": ti.get("averageVolume"),
            "averageVolume10days": ti.get("averageVolume10days"),
        }
    except Exception:
        info = {
            "name": ticker,
            "industry": "Unknown",
            "sector": "Unknown",
            "currency": "USD",
            "sector_etf": None,
        }

    meta["info"] = info
    meta["info_fetched"] = datetime.now().isoformat()
    _write_meta(ticker, meta)
    return info


def fetch_dividends(ticker: str, force: bool = False) -> pd.DataFrame:
    """Fetch and cache dividend history."""
    fp = _cache_path(ticker) / "dividends.parquet"
    meta = _read_meta(ticker)

    if fp.exists() and not force:
        cached_df = _read_parquet_safe(fp)
        if cached_df is not None:
            last_updated = meta.get("dividends", {}).get("last_updated")
            if last_updated:
                lu = datetime.fromisoformat(last_updated)
                if datetime.now() - lu < timedelta(days=7):
                    return cached_df

    try:
        t = yf.Ticker(ticker)
        df = t.dividends.to_frame(name="Dividend")
    except Exception:
        df = pd.DataFrame()

    if len(df) > 0:
        df.to_parquet(fp)
    meta["dividends"] = {"last_updated": datetime.now().isoformat()}
    _write_meta(ticker, meta)
    return df


def fetch_splits(ticker: str, force: bool = False) -> pd.DataFrame:
    """Fetch and cache stock split history."""
    fp = _cache_path(ticker) / "splits.parquet"
    meta = _read_meta(ticker)

    if fp.exists() and not force:
        cached_df = _read_parquet_safe(fp)
        if cached_df is not None:
            last_updated = meta.get("splits", {}).get("last_updated")
            if last_updated:
                lu = datetime.fromisoformat(last_updated)
                if datetime.now() - lu < timedelta(days=30):
                    return cached_df

    try:
        t = yf.Ticker(ticker)
        df = t.splits.to_frame(name="Split")
    except Exception:
        df = pd.DataFrame()

    if len(df) > 0:
        df.to_parquet(fp)
    meta["splits"] = {"last_updated": datetime.now().isoformat()}
    _write_meta(ticker, meta)
    return df


def fetch_earnings_dates(ticker: str, force: bool = False) -> pd.DataFrame:
    """Fetch and cache earnings dates (past and upcoming)."""
    fp = _cache_path(ticker) / "earnings_dates.parquet"
    meta = _read_meta(ticker)

    if fp.exists() and not force:
        cached_df = _read_parquet_safe(fp)
        if cached_df is not None:
            last_updated = meta.get("earnings_dates", {}).get("last_updated")
            if last_updated:
                lu = datetime.fromisoformat(last_updated)
                if datetime.now() - lu < timedelta(days=7):
                    return cached_df

    try:
        t = yf.Ticker(ticker)
        df = t.earnings_dates
    except Exception:
        df = pd.DataFrame()

    if df is not None and len(df) > 0:
        df.to_parquet(fp)
    else:
        df = pd.DataFrame()
    meta["earnings_dates"] = {"last_updated": datetime.now().isoformat()}
    _write_meta(ticker, meta)
    return df


# Sector → representative ETF mapping
SECTOR_ETFS = {
    # Sectors
    "Technology": "XLK", 
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
    "SP500": "SPY",       # SPDR S&P 500 ETF (proxy — ^GSPC has gaps)
    "DJI": "DIA",         # SPDR Dow Jones Industrial Average ETF
    "TSX": "^GSPTSE",     # S&P/TSX Composite Index (actual index)
    "NASDAQ": "^IXIC",    # Nasdaq Composite Index (actual index)
}

# Major indices (use actual index tickers where yfinance supports them)
INDEX_ETFS = {
    "SP500": "SPY",       # SPDR S&P 500 ETF (proxy — ^GSPC has gaps)
    "DJI": "DIA",         # SPDR Dow Jones Industrial Average ETF
    "TSX": "^GSPTSE",     # S&P/TSX Composite Index (actual index)
    "NASDAQ": "^IXIC",    # Nasdaq Composite Index (actual index)
}


def _sector_to_etf(sector: str) -> str | None:
    """Map yfinance sector name to SPDR sector ETF ticker."""
    return SECTOR_ETFS.get(sector)


def fetch_sector_history(ticker: str, sector_override: str | None = None, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Fetch the sector ETF history for a given stock ticker.

    Args:
        ticker: Stock ticker to look up sector for.
        sector_override: If provided, use this sector name instead of yfinance's.
    Returns (monthly_df, daily_df) for the sector ETF, or None if sector unknown.
    """
    if sector_override:
        etf = _sector_to_etf(sector_override)
    else:
        info = get_ticker_info(ticker)
        etf = info.get("sector_etf")
    if not etf:
        return None
    return fetch_history(etf, force=force)


def fetch_all(ticker: str, sector_override: str | None = None, force: bool = False) -> dict:
    """Fetch and cache everything for a ticker. Returns a summary dict."""
    monthly, daily = fetch_history(ticker, force=force)
    info = get_ticker_info(ticker, force=force)
    dividends = fetch_dividends(ticker, force=force)
    splits = fetch_splits(ticker, force=force)
    earnings = fetch_earnings_dates(ticker, force=force)
    sector = fetch_sector_history(ticker, sector_override=sector_override, force=force)
    indices = fetch_index_histories(force=force)

    return {
        "info": info,
        "monthly": monthly,
        "daily": daily,
        "dividends": dividends,
        "splits": splits,
        "earnings_dates": earnings,
        "sector": sector,
        "indices": indices,
    }


def fetch_index_histories(force: bool = False) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Fetch cached history for major market indices (via ETF proxies).

    Returns:
        Dict mapping index name → (monthly_df, daily_df).
    """
    results = {}
    for name, etf in INDEX_ETFS.items():
        try:
            monthly, daily = fetch_history(etf, force=force)
            results[name] = (monthly, daily)
        except Exception:
            pass
    return results


def invalidate_cache(ticker: str):
    """Remove all cached data for a ticker."""
    import shutil

    p = _cache_path(ticker)
    if p.exists():
        shutil.rmtree(p)
