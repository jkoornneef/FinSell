"""Distribution day counting and scoring.

Tracks institutional selling pressure via distribution days —
days where price declines 0.2%+ on higher volume than the prior session.
Works for individual stocks or indices (same mechanics).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DistributionDayScore:
    """Distribution day accumulation score.

    Attributes:
        score: -1.0 to +1.0 — severity of distribution.
            -1.0 = no distribution days, healthy accumulation
            -0.5 = 1-2 distribution days, normal churn
             0.0 = 3 distribution days, watch list
            +0.5 = 4-5 distribution days, caution
            +1.0 = 6+ distribution days, heavy institutional selling
        count: Number of distribution days in the lookback window.
        days: List of dicts with date, pct_change, volume_ratio for each dist day.
        stalling_days: Count of stalling days (up on heavy volume but small gain).
        details: Dict with summary statistics.
    """

    score: float
    count: int
    days: list[dict]
    stalling_days: int
    details: dict | None


def score_distribution_days(
    df: pd.DataFrame,
    lookback: int = 25,
    decline_threshold: float = -0.002,
    stalling_gain_max: float = 0.002,
    volume_increase_min: float = 1.1,
    heavy_volume_threshold: float = 1.25,
) -> DistributionDayScore:
    """Count and score distribution days over a rolling window.

    A distribution day occurs when:
      1. Price closes down 0.2% or more from the prior close
      2. Volume is ABOVE the 50-day average volume

    A stalling day (also bearish) occurs when:
      1. Price closes up, but gains less than 0.2%
      2. Volume is 25%+ above the 50-day average (institutions selling into strength)

    The IBD rule: 4-5+ distribution days within 25 trading days signals
    a potential market top or stock distribution.

    Score interpretation:
        -1.0 = 0 distribution days, clean accumulation
        -0.5 = 1-2 days, normal market action
         0.0 = 3 days, neutral / watch
        +0.5 = 4-5 days, elevated selling pressure
        +1.0 = 6+ days, heavy distribution (danger)

    Note: Works identically for stocks and indices — just pass
    the appropriate OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with Close and Volume columns.
        lookback: Rolling window in trading days (25 ≈ 5 weeks).
        decline_threshold: Minimum decline to qualify (-0.2% = -0.002).
        stalling_gain_max: Max gain for a stalling day (0.2%).
        volume_increase_min: Volume must be >= this ratio of 50-day avg (1.0 = at avg).
        heavy_volume_threshold: Volume ratio vs 50-day avg for stalling detection.

    Returns:
        DistributionDayScore with count, individual days, and overall score.
    """
    if len(df) < lookback + 2:
        return DistributionDayScore(-1.0, 0, [], 0, None)

    close = df["Close"].values
    volume = df["Volume"].values if "Volume" in df.columns else None

    if volume is None:
        return DistributionDayScore(-1.0, 0, [], 0, None)

    # Use the most recent `lookback` bars
    window_start = len(df) - lookback
    window_df = df.iloc[window_start:]

    distribution_days = []
    stalling_days = 0

    # 50-day average volume — use the period BEFORE the lookback window
    # so the average isn't inflated by recent distribution itself
    vol_avg_end = window_start
    vol_avg_start = max(0, vol_avg_end - 50)
    if vol_avg_end - vol_avg_start < 10:
        # Not enough pre-window data, fall back to full 50-day
        vol_avg_start = max(0, len(df) - 50)
        vol_avg_end = len(df)
    avg_volume_50 = np.mean(volume[vol_avg_start:vol_avg_end])

    # Volatility-adjusted decline threshold: the decline must be meaningful
    # relative to the stock's normal daily movement
    if len(close) >= 60:
        recent_close = close[-60:]
        daily_returns = np.diff(recent_close) / recent_close[:-1]
    else:
        daily_returns = np.diff(close) / close[:-1]

    if len(daily_returns) > 10:
        avg_daily_vol = np.std(daily_returns)
        # Effective threshold: at least 0.2%, but also at least 0.4x the daily volatility
        effective_decline = min(decline_threshold, -0.4 * avg_daily_vol)
    else:
        effective_decline = decline_threshold

    for i in range(1, len(window_df)):
        idx = window_start + i
        if idx < 1:
            continue

        today_close = close[idx]
        prev_close = close[idx - 1]
        today_volume = volume[idx]

        if prev_close <= 0 or avg_volume_50 <= 0:
            continue

        pct_change = (today_close - prev_close) / prev_close
        vol_vs_avg = today_volume / avg_volume_50

        # Distribution day: meaningful decline + volume above 50-day average
        if pct_change <= effective_decline and vol_vs_avg >= volume_increase_min:
            date = window_df.index[i] if hasattr(window_df.index[i], 'strftime') else i
            distribution_days.append({
                "date": str(date),
                "pct_change": float(pct_change),
                "volume_ratio": float(vol_vs_avg),
                "bar_index": int(idx),
            })

        # Stalling day: tiny gain + heavy volume (institutions selling into strength)
        elif 0 < pct_change <= stalling_gain_max:
            if vol_vs_avg >= heavy_volume_threshold:
                stalling_days += 1

    count = len(distribution_days)

    # --- Score: map count to -1 to +1 ---
    # 0 days → -1.0
    # 1 day  → -0.67
    # 2 days → -0.33
    # 3 days →  0.0
    # 4 days → +0.33
    # 5 days → +0.67
    # 6+ days → +1.0
    # Stalling days count as half a distribution day
    effective_count = count + stalling_days * 0.5

    if effective_count <= 0:
        score = -1.0
    elif effective_count <= 3:
        score = -1.0 + (effective_count / 3.0)
    elif effective_count <= 6:
        score = (effective_count - 3.0) / 3.0
    else:
        score = 1.0

    # --- Clustering penalty: distribution days close together are worse ---
    clustering_bonus = 0.0
    if count >= 3:
        bar_indices = [d["bar_index"] for d in distribution_days]
        gaps = [bar_indices[i+1] - bar_indices[i] for i in range(len(bar_indices) - 1)]
        if gaps:
            avg_gap = np.mean(gaps)
            # Tight clustering (avg gap < 3 bars) adds severity
            if avg_gap < 3:
                clustering_bonus = 0.2
            elif avg_gap < 5:
                clustering_bonus = 0.1

    score = float(np.clip(score + clustering_bonus, -1.0, 1.0))

    # --- Details ---
    details = {
        "distribution_count": count,
        "stalling_count": stalling_days,
        "effective_count": float(effective_count),
        "lookback_bars": lookback,
        "avg_decline_pct": float(np.mean([d["pct_change"] for d in distribution_days])) if distribution_days else 0.0,
        "avg_volume_ratio": float(np.mean([d["volume_ratio"] for d in distribution_days])) if distribution_days else 0.0,
        "clustering_bonus": float(clustering_bonus),
    }

    return DistributionDayScore(
        score=score,
        count=count,
        days=distribution_days,
        stalling_days=stalling_days,
        details=details,
    )
