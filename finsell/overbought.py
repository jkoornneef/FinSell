"""Overbought indicator scoring.

Short-term profit-taking signal based on RSI reaching overbought levels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OverboughtScore:
    """RSI overbought score for profit-locking signals.

    Attributes:
        score: -1.0 to +1.0 — proximity to overbought condition.
            -1.0 = oversold (RSI ~30), no reason to sell
            -0.5 = neutral zone (RSI ~50)
             0.0 = approaching overbought (RSI ~65)
            +0.5 = overbought threshold hit (RSI = 70)
            +1.0 = extreme overbought (RSI 80+), urgent profit-lock
        rsi: Current RSI value.
        rsi_slope: Rate of RSI change (positive = still climbing).
        details: Dict with additional context.
    """

    score: float
    rsi: float
    rsi_slope: float
    details: dict | None


def score_overbought(
    df: pd.DataFrame,
    rsi_period: int = 14,
    overbought_level: float = 70.0,
    extreme_level: float = 80.0,
    slope_lookback: int = 3,
) -> OverboughtScore:
    """Score how overbought a stock is using RSI.

    A short-term signal for locking in profits. When RSI hits 70,
    momentum is stretched and a pullback becomes likely. Higher RSI
    readings increase urgency.

    Score interpretation:
        -1.0 = RSI near 30 (oversold, no sell pressure)
        -0.5 = RSI near 50 (neutral)
         0.0 = RSI near 65 (warming up, watch)
        +0.5 = RSI hits 70 (overbought — consider trimming)
        +1.0 = RSI 80+ (extremely stretched — lock profits)

    The slope of RSI adds nuance:
      - RSI at 72 but declining → more urgent (momentum fading)
      - RSI at 72 but still rising → watch but not yet urgent

    Args:
        df: OHLCV DataFrame. Needs at least rsi_period + slope_lookback bars.
        rsi_period: RSI calculation period (14 standard).
        overbought_level: RSI level that triggers overbought (70).
        extreme_level: RSI level for extreme overbought (80).
        slope_lookback: Bars to measure RSI trend direction.

    Returns:
        OverboughtScore with proximity score, RSI value, and slope.
    """
    if len(df) < rsi_period + slope_lookback + 1:
        return OverboughtScore(-1.0, 50.0, 0.0, None)

    # Compute RSI
    close = df["Close"]
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()

    # Use exponential smoothing after initial SMA (Wilder's method)
    for i in range(rsi_period + 1, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (rsi_period - 1) + gain.iloc[i]) / rsi_period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (rsi_period - 1) + loss.iloc[i]) / rsi_period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    current_rsi = float(rsi.iloc[-1])
    if np.isnan(current_rsi):
        return OverboughtScore(-1.0, 50.0, 0.0, None)

    # RSI slope (rate of change over last N bars)
    rsi_values = rsi.iloc[-slope_lookback - 1:].values
    if len(rsi_values) >= 2 and not np.any(np.isnan(rsi_values)):
        rsi_slope = float(rsi_values[-1] - rsi_values[0]) / slope_lookback
    else:
        rsi_slope = 0.0

    # --- Score: map RSI to -1 to +1 ---
    # 30 → -1.0, 50 → -0.5, 65 → 0.0, 70 → +0.5, 80+ → +1.0
    if current_rsi <= 30:
        base_score = -1.0
    elif current_rsi <= 50:
        # 30→-1.0, 50→-0.5
        base_score = -1.0 + 0.5 * ((current_rsi - 30) / 20.0)
    elif current_rsi <= 65:
        # 50→-0.5, 65→0.0
        base_score = -0.5 + 0.5 * ((current_rsi - 50) / 15.0)
    elif current_rsi <= overbought_level:
        # 65→0.0, 70→+0.5
        base_score = 0.5 * ((current_rsi - 65) / (overbought_level - 65))
    elif current_rsi <= extreme_level:
        # 70→+0.5, 80→+1.0
        base_score = 0.5 + 0.5 * ((current_rsi - overbought_level) / (extreme_level - overbought_level))
    else:
        base_score = 1.0

    # Slope adjustment: declining RSI when overbought = more urgent
    slope_adj = 0.0
    if current_rsi >= overbought_level:
        if rsi_slope < -1.0:
            # RSI declining while overbought — momentum fading, more urgent
            slope_adj = 0.1
        elif rsi_slope > 2.0:
            # RSI still climbing fast — less urgent (trend may continue)
            slope_adj = -0.05

    score = float(np.clip(base_score + slope_adj, -1.0, 1.0))

    # Duration overbought: how many bars RSI has been above 70
    bars_overbought = 0
    for i in range(len(rsi) - 1, max(len(rsi) - 30, 0), -1):
        if not np.isnan(rsi.iloc[i]) and rsi.iloc[i] >= overbought_level:
            bars_overbought += 1
        else:
            break

    return OverboughtScore(
        score=score,
        rsi=current_rsi,
        rsi_slope=rsi_slope,
        details={
            "rsi_period": rsi_period,
            "overbought_level": overbought_level,
            "bars_overbought": bars_overbought,
            "rsi_slope_per_bar": rsi_slope,
            "slope_adjustment": slope_adj,
        },
    )
