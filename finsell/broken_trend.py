"""Broken trend line detection and scoring.

Identifies two key bearish structural breaks:
  1. Weekly close below a long-term uptrend line
  2. 200-day moving average turning down
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BrokenTrendScore:
    """Broken trend line score.

    Attributes:
        score: -1.0 to +1.0 — severity of the trend break.
            -1.0 = trend firmly intact, price well above support
            -0.5 = approaching trend line from above
             0.0 = testing the trend line (at support)
            +0.5 = closed below trend line (initial break)
            +1.0 = confirmed break (multiple closes below + MA200 declining)
        trendline_break: Whether price has closed below the uptrend line.
        ma200_declining: Whether the 200-day MA slope has turned negative.
        details: Dict with measurements.
    """

    score: float
    trendline_break: bool
    ma200_declining: bool
    details: dict | None


def _fit_uptrend_line(close: np.ndarray, lookback: int = 200) -> tuple[float, float]:
    """Fit a linear uptrend line to significant lows over a lookback period.

    Returns (slope, intercept) in price-per-bar units, anchored to the
    start of the lookback window.
    """
    if len(close) < lookback:
        lookback = len(close)

    region = close[-lookback:]

    # Find swing lows (local minima over 10-bar windows)
    lows_x = []
    lows_y = []
    window = 10

    for i in range(window, len(region) - window):
        segment = region[i - window: i + window + 1]
        if region[i] == np.min(segment):
            lows_x.append(i)
            lows_y.append(region[i])

    # Need at least 2 points to draw a line
    if len(lows_x) < 2:
        # Fall back: use the first and last significant lows
        first_low_idx = np.argmin(region[:lookback // 3])
        last_low_idx = lookback // 2 + np.argmin(region[lookback // 2:])
        lows_x = [first_low_idx, last_low_idx]
        lows_y = [region[first_low_idx], region[last_low_idx]]

    # For uptrend: fit a line through the lower support points
    # Use only lows that form an ascending sequence (filter non-ascending)
    ascending_x = [lows_x[0]]
    ascending_y = [lows_y[0]]
    for i in range(1, len(lows_x)):
        if lows_y[i] >= ascending_y[-1]:
            ascending_x.append(lows_x[i])
            ascending_y.append(lows_y[i])

    if len(ascending_x) < 2:
        ascending_x = lows_x[:2]
        ascending_y = lows_y[:2]

    # Linear regression on the ascending lows
    x = np.array(ascending_x, dtype=float)
    y = np.array(ascending_y, dtype=float)

    # Least squares fit
    n = len(x)
    if n < 2:
        return 0.0, float(close[-1])

    slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2 + 1e-10)
    intercept = (np.sum(y) - slope * np.sum(x)) / n

    # Convert to absolute position (offset from start of lookback to end of array)
    # The trendline value at bar i (from start of lookback) = intercept + slope * i
    # At the current bar (end of lookback): trendline_now = intercept + slope * (lookback - 1)
    return float(slope), float(intercept)


def _weekly_close(df: pd.DataFrame) -> pd.Series:
    """Get weekly closing prices from daily data."""
    if hasattr(df.index, 'to_period'):
        weekly = df["Close"].resample("W").last().dropna()
    else:
        # Fallback: group every 5 bars
        close = df["Close"].values
        weekly_closes = [close[i] for i in range(4, len(close), 5)]
        weekly = pd.Series(weekly_closes)
    return weekly


def score_broken_trend(
    df: pd.DataFrame,
    trendline_lookback: int = 200,
    ma_period: int = 200,
    ma_slope_period: int = 20,
    break_threshold: float = 0.01,
    weekly_confirm: bool = True,
) -> BrokenTrendScore:
    """Score whether a long-term uptrend has been broken.

    Detects two independent bearish signals:
      1. Price closes below the long-term uptrend line (drawn along
         ascending swing lows over ~200 bars). Weekly close confirmation
         is stronger than intraday.
      2. The 200-day moving average slope turns negative (trend no longer rising).

    Score interpretation:
        -1.0 = trend firmly intact, price well above trendline + MA200 rising
        -0.5 = price approaching trendline from above, or MA200 flattening
         0.0 = price at the trendline, testing support
        +0.5 = closed below trendline OR MA200 turning down
        +1.0 = both confirmed: below trendline + MA200 declining

    Args:
        df: OHLCV DataFrame. Daily data, needs 200+ bars.
        trendline_lookback: Bars used to fit the uptrend line.
        ma_period: Moving average period (200).
        ma_slope_period: Bars to measure MA slope direction.
        break_threshold: How far below trendline counts as a break (1% default).
        weekly_confirm: If True, checks weekly close for stronger confirmation.

    Returns:
        BrokenTrendScore with proximity, break flags, and details.
    """
    if len(df) < ma_period + 10:
        return BrokenTrendScore(-1.0, False, False, None)

    close = df["Close"].values

    # --- 200-day Moving Average analysis ---
    ma200 = pd.Series(close).rolling(window=ma_period).mean().values

    current_ma200 = ma200[-1]
    prior_ma200 = ma200[-ma_slope_period] if len(ma200) > ma_slope_period else ma200[0]

    if np.isnan(current_ma200) or np.isnan(prior_ma200) or prior_ma200 <= 0:
        return BrokenTrendScore(-1.0, False, False, None)

    ma200_slope = (current_ma200 - prior_ma200) / prior_ma200
    ma200_declining = ma200_slope < -0.001  # Slightly negative threshold

    # --- Uptrend line analysis ---
    slope, intercept = _fit_uptrend_line(close, lookback=trendline_lookback)

    # Trendline value at current bar position
    # Position within the lookback window: the last bar is at index (lookback - 1)
    effective_lookback = min(trendline_lookback, len(close))
    trendline_now = intercept + slope * (effective_lookback - 1)

    # Is the trend line ascending? (only valid for uptrend breaks)
    uptrend_valid = slope > 0

    current_price = close[-1]

    # Distance from trendline (positive = above, negative = below)
    if trendline_now > 0 and uptrend_valid:
        distance_pct = (current_price - trendline_now) / trendline_now
    else:
        # No valid uptrend — can't break what doesn't exist
        distance_pct = 0.0
        uptrend_valid = False

    # Weekly close confirmation
    weekly_break = False
    if weekly_confirm and uptrend_valid:
        weekly_closes = _weekly_close(df)
        if len(weekly_closes) >= 2:
            last_weekly_close = weekly_closes.iloc[-1]
            weekly_break = last_weekly_close < trendline_now * (1 - break_threshold * 0.5)

    # Daily break
    daily_break = uptrend_valid and distance_pct < -break_threshold
    trendline_break = daily_break or weekly_break

    # --- Score calculation ---
    # Component 1: Trendline proximity/break
    if not uptrend_valid:
        trendline_score = 0.0
    elif distance_pct > 0.05:
        # Well above trendline
        trendline_score = -1.0
    elif distance_pct > 0.02:
        # Comfortably above
        trendline_score = -0.5
    elif distance_pct > 0:
        # Approaching trendline
        trendline_score = -0.5 + 0.5 * (1.0 - distance_pct / 0.02)
    elif distance_pct > -break_threshold:
        # At or slightly below (testing)
        trendline_score = 0.0 + 0.3 * (abs(distance_pct) / break_threshold)
    else:
        # Below trendline
        break_severity = min(abs(distance_pct) / 0.05, 1.0)
        trendline_score = 0.3 + 0.4 * break_severity
        if weekly_break:
            trendline_score += 0.1  # Weekly confirmation bonus

    # Component 2: MA200 slope
    if ma200_slope > 0.01:
        ma_score = -1.0  # Strongly rising
    elif ma200_slope > 0.003:
        ma_score = -0.5  # Rising
    elif ma200_slope > -0.001:
        ma_score = 0.0  # Flat
    elif ma200_slope > -0.005:
        ma_score = 0.3  # Starting to turn down
    else:
        ma_score = 0.7  # Clearly declining

    # Price vs MA200
    price_vs_ma200 = (current_price - current_ma200) / current_ma200
    if price_vs_ma200 < -0.02:
        ma_score += 0.3  # Below MA200 adds severity

    # Combined score: 55% trendline, 45% MA200
    score = 0.55 * trendline_score + 0.45 * min(ma_score, 1.0)

    # Both confirming = push toward +1.0
    if trendline_break and ma200_declining:
        score = max(score, 0.7)
        score = min(score + 0.2, 1.0)

    score = float(np.clip(score, -1.0, 1.0))

    return BrokenTrendScore(
        score=score,
        trendline_break=bool(trendline_break),
        ma200_declining=bool(ma200_declining),
        details={
            "current_price": float(current_price),
            "trendline_value": float(trendline_now) if uptrend_valid else None,
            "distance_from_trendline_pct": float(distance_pct),
            "trendline_slope_per_bar": float(slope),
            "ma200_value": float(current_ma200),
            "ma200_slope_pct": float(ma200_slope),
            "price_vs_ma200_pct": float(price_vs_ma200),
            "weekly_close_break": bool(weekly_break),
            "uptrend_valid": bool(uptrend_valid),
        },
    )
