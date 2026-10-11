"""Composite Rating and Relative Strength scoring.

Approximates IBD-style Composite Rating using:
  - RS Rating: 12-month weighted price performance vs a stock universe
  - Industry Group RS: Sector ETF performance vs SPY
  - EPS Rating: Earnings growth (delegates to fundamentals.earnings_growth_score)
  - SMR Rating: Sales growth, profit margins, ROE
  - Accumulation/Distribution: Volume-weighted price action
"""

import numpy as np
import pandas as pd
import yfinance as yf


# SPDR sector ETFs mapped to GICS sector names
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "SP500": "SPY",       # SPDR S&P 500 ETF (proxy — ^GSPC has gaps)
    "DJI": "DIA",         # SPDR Dow Jones Industrial Average ETF
    "TSX": "^GSPTSE",     # S&P/TSX Composite Index (actual index)
    "NASDAQ": "^IXIC",    # Nasdaq Composite Index (actual index)
}


def _safe_yf_info(ticker: str) -> dict:
    """Return ticker info as a dict, or an empty dict on Yahoo errors."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {}
    return info if isinstance(info, dict) else {}


def _weighted_return(prices: pd.Series) -> float:
    """IBD-style weighted 12-month return.

    Weights: 40% most recent quarter, 20% each for the other three quarters.
    """
    if len(prices) < 252:
        # Fall back to simple return if not enough data
        return (prices.iloc[-1] / prices.iloc[0]) - 1.0

    # Split into 4 quarters (most recent last)
    q_len = len(prices) // 4
    q4 = prices.iloc[-q_len:]  # most recent
    q3 = prices.iloc[-2 * q_len: -q_len]
    q2 = prices.iloc[-3 * q_len: -2 * q_len]
    q1 = prices.iloc[: -3 * q_len]

    r1 = (q1.iloc[-1] / q1.iloc[0]) - 1.0 if len(q1) > 1 else 0.0
    r2 = (q2.iloc[-1] / q2.iloc[0]) - 1.0 if len(q2) > 1 else 0.0
    r3 = (q3.iloc[-1] / q3.iloc[0]) - 1.0 if len(q3) > 1 else 0.0
    r4 = (q4.iloc[-1] / q4.iloc[0]) - 1.0 if len(q4) > 1 else 0.0

    return 0.20 * r1 + 0.20 * r2 + 0.20 * r3 + 0.40 * r4


def rs_rating(ticker: str, universe: list[str] | None = None) -> float:
    """Compute Relative Strength percentile (0-99) for a stock.

    Compares the stock's weighted 12-month return against a universe of stocks.
    Default universe is S&P 500 sector ETFs + SPY as a quick proxy.

    Args:
        ticker: Stock symbol.
        universe: List of ticker symbols to rank against. If None, uses
            the sector ETFs + SPY as a small but representative universe.

    Returns:
        RS percentile 0-99 (higher is better).
    """
    if universe is None:
        universe = list(set(SECTOR_ETFS.values())) + ["SPY"]

    all_tickers = list(set([ticker] + universe))
    print(f"[DEBUG] Download b : {all_tickers}")
    try:
        data = yf.download(all_tickers, period="1y", interval="1d", progress=False)
    except Exception:
        return 0.0
    print(f"[DEBUG] Download b complete : {all_tickers}")

    if data is None or getattr(data, "empty", True):
        return 0.0

    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data[["Close"]]
        closes.columns = [ticker]

    # Compute weighted returns for all
    returns = {}
    for sym in closes.columns:
        series = closes[sym].dropna()
        if len(series) > 20:
            returns[sym] = _weighted_return(series)

    if ticker not in returns:
        return 0.0

    # Percentile rank
    all_rets = sorted(returns.values())
    stock_ret = returns[ticker]
    rank = sum(1 for r in all_rets if r <= stock_ret)
    percentile = (rank / len(all_rets)) * 99.0
    return float(percentile)


def industry_group_rs(ticker: str) -> float:
    """Compute Industry Group Relative Strength percentile (0-99).

    Compares the stock's sector ETF performance vs SPY over 12 months,
    then ranks among all sector ETFs.

    Returns:
        Percentile 0-99.
    """
    info = _safe_yf_info(ticker)
    sector = info.get("sector", "")

    etf_symbol = SECTOR_ETFS.get(sector)
    if not etf_symbol:
        return 50.0  # neutral if sector unknown

    # Fetch all sector ETFs + SPY
    all_etfs = list(set(SECTOR_ETFS.values()))
    symbols = list(set(all_etfs + ["SPY"]))
    print(f"[DEBUG] Download c : {symbols}")
    try:
        data = yf.download(symbols, period="1y", interval="1d", progress=False)
    except Exception:
        return 50.0
    print(f"[DEBUG] Download c complete : {symbols}")

    if data is None or getattr(data, "empty", True):
        return 50.0
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        return 50.0

    # Relative returns vs SPY
    spy_ret = _weighted_return(closes["SPY"].dropna())
    relative_rets = {}
    for etf in all_etfs:
        if etf in closes.columns:
            series = closes[etf].dropna()
            if len(series) > 20:
                relative_rets[etf] = _weighted_return(series) - spy_ret

    if etf_symbol not in relative_rets:
        return 50.0

    # Percentile rank among sector ETFs
    all_rels = sorted(relative_rets.values())
    stock_rel = relative_rets[etf_symbol]
    rank = sum(1 for r in all_rels if r <= stock_rel)
    percentile = (rank / len(all_rels)) * 99.0
    return float(percentile)


def smr_rating(ticker: str) -> float:
    """Compute Sales/Margins/ROE rating percentile (0-99).

    Uses quarterly financials from yfinance:
      - Revenue growth (most recent quarter YoY)
      - Profit margin (net income / revenue)
      - ROE (net income / stockholders equity)

    Returns:
        Score 0-99.
    """
    tk = yf.Ticker(ticker)

    try:
        quarterly_income = tk.quarterly_income_stmt
        quarterly_balance = tk.quarterly_balance_sheet
    except Exception:
        return 50.0

    if quarterly_income.empty:
        return 50.0

    # Revenue growth
    revenue_row = None
    for label in ["Total Revenue", "Revenue"]:
        if label in quarterly_income.index:
            revenue_row = quarterly_income.loc[label]
            break

    rev_growth = 0.0
    if revenue_row is not None and len(revenue_row) >= 5:
        current = revenue_row.iloc[0]
        prior = revenue_row.iloc[4]  # same quarter prior year
        if prior > 0:
            rev_growth = (current - prior) / prior

    # Profit margin
    net_income_row = None
    for label in ["Net Income", "Net Income Common Stockholders"]:
        if label in quarterly_income.index:
            net_income_row = quarterly_income.loc[label]
            break

    margin = 0.0
    if net_income_row is not None and revenue_row is not None:
        rev_val = revenue_row.iloc[0]
        ni_val = net_income_row.iloc[0]
        if rev_val > 0:
            margin = ni_val / rev_val

    # ROE
    roe = 0.0
    if net_income_row is not None and not quarterly_balance.empty:
        equity_row = None
        for label in ["Stockholders Equity", "Total Stockholder Equity",
                      "Stockholders' Equity"]:
            if label in quarterly_balance.index:
                equity_row = quarterly_balance.loc[label]
                break
        if equity_row is not None:
            eq_val = equity_row.iloc[0]
            ni_val = net_income_row.iloc[0]
            if eq_val > 0:
                roe = ni_val / eq_val

    # Score: combine rev growth, margin, ROE into 0-99
    # Rev growth: 25%+ → top score; Margin: 20%+ → top; ROE: 17%+ → top
    rev_score = np.clip(rev_growth / 0.25, 0.0, 1.0)
    margin_score = np.clip(margin / 0.20, 0.0, 1.0)
    roe_score = np.clip(roe / 0.17, 0.0, 1.0)

    raw = (rev_score * 0.40 + margin_score * 0.30 + roe_score * 0.30)
    return float(np.clip(raw * 99.0, 0.0, 99.0))


def accumulation_distribution_rating(df: pd.DataFrame) -> float:
    """Compute Accumulation/Distribution rating (0-99) from OHLCV data.

    Uses the Money Flow Multiplier and volume to determine whether
    institutional accumulation or distribution is occurring.

    Args:
        df: DataFrame with Close, High, Low, Volume columns (last ~50 days).

    Returns:
        Percentile score 0-99. Higher = more accumulation.
    """
    if len(df) < 20:
        return 50.0

    recent = df.tail(50).copy()

    high = recent["High"]
    low = recent["Low"]
    close = recent["Close"]
    volume = recent["Volume"]

    # Money Flow Multiplier
    hl_range = high - low
    mfm = np.where(hl_range > 0, ((close - low) - (high - close)) / hl_range, 0.0)

    # Money Flow Volume
    mfv = mfm * volume.values

    # AD line
    ad_line = np.cumsum(mfv)

    # Score: is the AD line trending up? Compare last 10 days vs first 10 days
    if len(ad_line) >= 20:
        recent_ad = np.mean(ad_line[-10:])
        earlier_ad = np.mean(ad_line[:10])
        total_vol = volume.sum()
        if total_vol > 0:
            # Normalize by total volume to get a -1 to 1 ratio
            ratio = (recent_ad - earlier_ad) / total_vol
            # Map to 0-99: -0.1 → 0, 0 → 50, +0.1 → 99
            score = 50.0 + ratio * 500.0
            return float(np.clip(score, 0.0, 99.0))

    return 50.0


def composite_rating_score(
    ticker: str,
    universe: list[str] | None = None,
) -> float:
    """Compute a 0-2 composite rating score for a stock.

    Approximates IBD's Composite Rating by combining:
      - RS Rating (weighted 12-month price performance percentile)
      - Industry Group RS (sector ETF performance vs SPY)
      - EPS Rating (earnings growth)
      - SMR Rating (Sales/Margins/ROE)
      - Accumulation/Distribution

    Scoring:
      0.0 — RS < 80 AND Composite < 90 (weak on both measures)
      0.0–1.0 — Meets one threshold partially
      1.0 — Meets RS ≥ 80 OR Composite ≥ 90
      1.0–2.0 — Meets both RS ≥ 80 AND Composite ≥ 90
      2.0 — RS ≥ 90 AND Composite ≥ 95

    Args:
        ticker: Stock symbol.
        universe: Optional list of tickers for RS ranking.

    Returns:
        Float score between 0.0 and 2.0.
    """
    from .fundamentals import earnings_growth_score

    # --- Fetch price data ---
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
    except Exception:
        return 0.0
    if df is None or getattr(df, "empty", True):
        return 0.0
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")
    if df.empty or len(df) < 50:
        return 0.0

    # --- Sub-ratings (all 0-99 percentiles) ---
    try:
        rs = rs_rating(ticker, universe)
    except Exception:
        rs = 0.0
    try:
        ig_rs = industry_group_rs(ticker)
    except Exception:
        ig_rs = 50.0
    try:
        smr = smr_rating(ticker)
    except Exception:
        smr = 50.0
    try:
        ad = accumulation_distribution_rating(df)
    except Exception:
        ad = 50.0

    # EPS: reuse earnings_growth_score (0-2 scale) → map to 0-99
    tk = yf.Ticker(ticker)
    try:
        income = tk.quarterly_income_stmt
        eps_row = None
        for label in ["Basic EPS", "Diluted EPS"]:
            if label in income.index:
                eps_row = income.loc[label]
                break

        if eps_row is not None and len(eps_row) >= 5:
            quarterly_eps = list(reversed(eps_row.dropna().tolist()))
        else:
            quarterly_eps = []

        annual_income = tk.income_stmt
        annual_eps_row = None
        for label in ["Basic EPS", "Diluted EPS"]:
            if label in annual_income.index:
                annual_eps_row = annual_income.loc[label]
                break

        if annual_eps_row is not None and len(annual_eps_row) >= 2:
            annual_eps = list(reversed(annual_eps_row.dropna().tolist()))
        else:
            annual_eps = []

        eps_raw = earnings_growth_score(quarterly_eps, annual_eps)
        eps = (eps_raw / 2.0) * 99.0  # map 0-2 → 0-99
    except Exception:
        eps = 50.0

    # --- Composite percentile (weighted blend of sub-ratings) ---
    # IBD weights are proprietary; reasonable approximation:
    #   RS: 30%, EPS: 30%, Industry Group RS: 15%, SMR: 15%, A/D: 10%
    composite = (
        0.30 * rs
        + 0.30 * eps
        + 0.15 * ig_rs
        + 0.15 * smr
        + 0.10 * ad
    )

    # --- Map to 0-2 score ---
    # Two thresholds: RS ≥ 80 and Composite ≥ 90
    rs_meets = rs >= 80
    comp_meets = composite >= 90

    if rs_meets and comp_meets:
        # Both met: score 1.0-2.0 based on how far above thresholds
        rs_excess = np.clip((rs - 80) / 19.0, 0.0, 1.0)  # 80→0, 99→1
        comp_excess = np.clip((composite - 90) / 9.0, 0.0, 1.0)  # 90→0, 99→1
        score = 1.0 + 0.5 * rs_excess + 0.5 * comp_excess
    elif rs_meets or comp_meets:
        # One met: score 0.5-1.0
        if rs_meets:
            score = 0.5 + 0.5 * np.clip((composite - 70) / 20.0, 0.0, 1.0)
        else:
            score = 0.5 + 0.5 * np.clip((rs - 60) / 20.0, 0.0, 1.0)
    else:
        # Neither met: 0.0-0.5 based on proximity
        rs_progress = np.clip(rs / 80.0, 0.0, 1.0)
        comp_progress = np.clip(composite / 90.0, 0.0, 1.0)
        score = 0.25 * rs_progress + 0.25 * comp_progress

    return float(np.clip(score, 0.0, 2.0))
