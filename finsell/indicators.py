"""Technical indicator helpers using pandas-ta."""

import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators to OHLCV data.

    Adds: SMA(20), SMA(50), EMA(12), EMA(26), RSI(14), MACD, Bollinger Bands, ATR(14).
    """
    df = df.copy()
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.ema(length=12, append=True)
    df.ta.ema(length=26, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    return df


def ema5(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Compute the 5-period Exponential Moving Average."""
    return df.ta.ema(close=df[col], length=5)


def sma50_sma200(df: pd.DataFrame, col: str = "Close") -> pd.DataFrame:
    """Compute the 50-period and 200-period Simple Moving Averages.

    Returns:
        DataFrame with columns 'SMA_50' and 'SMA_200'.
    """
    result = pd.DataFrame(index=df.index)
    result["SMA_50"] = df.ta.sma(close=df[col], length=50)
    result["SMA_200"] = df.ta.sma(close=df[col], length=200)
    return result


def golden_cross_indicator(
    df: pd.DataFrame,
    col: str = "Close",
    lookback: int = 50,
) -> pd.Series:
    """Compute a -1 to +1 indicator for the Golden Cross (50 SMA crossing above 200 SMA).

    The indicator measures temporal proximity to the crossover point:
        -1.0 = furthest point in the lead-up (50 SMA well below 200 SMA, converging)
        ~0.0 = the moment of crossing
        +1.0 = furthest point after the cross (time has passed since the event)

    NaN where insufficient data exists (first 200 bars) or no cross detected in window.

    Args:
        df: OHLCV DataFrame.
        col: Price column to use for SMA calculation.
        lookback: Number of bars on each side of the cross to scale the indicator.
            Controls how quickly the value moves from -1→0 and 0→+1.

    Returns:
        Series with values from -1.0 to +1.0.
    """
    import numpy as np

    sma50 = df.ta.sma(close=df[col], length=50)
    sma200 = df.ta.sma(close=df[col], length=200)

    # Difference: positive = 50 above 200 (bullish), negative = below
    diff = sma50 - sma200

    # Detect crossover points: diff goes from negative to positive
    cross_signal = (diff > 0) & (diff.shift(1) <= 0)
    cross_indices = np.where(cross_signal.values)[0]

    result = pd.Series(np.nan, index=df.index)

    if len(cross_indices) == 0:
        # No cross found — use convergence/divergence as raw signal
        valid = diff.dropna()
        if len(valid) == 0:
            return result
        # If 50 is below 200 and gap is narrowing, we're approaching (-1 to 0)
        # If 50 is above 200, we're past it (0 to +1)
        # Normalize by the max absolute gap in the lookback window
        for i in range(len(df)):
            if np.isnan(diff.iloc[i]):
                continue
            window_start = max(0, i - lookback)
            window = diff.iloc[window_start:i + 1].dropna()
            if len(window) == 0:
                continue
            max_gap = window.abs().max()
            if max_gap == 0:
                result.iloc[i] = 0.0
            elif diff.iloc[i] < 0:
                # Below: scale -1 to 0 based on how close the gap is to closing
                result.iloc[i] = -(abs(diff.iloc[i]) / max_gap)
            else:
                result.iloc[i] = abs(diff.iloc[i]) / max_gap
        return result.clip(-1.0, 1.0)

    # For each bar, find the nearest golden cross and compute distance
    for i in range(len(df)):
        if np.isnan(diff.iloc[i]):
            continue

        # Find the closest cross point
        distances = i - cross_indices
        # Prefer the most recent cross (last one at or before current bar)
        past_crosses = distances[distances >= 0]
        future_crosses = distances[distances < 0]

        if len(past_crosses) > 0:
            # We're after a cross — value is 0 to +1
            bars_since = past_crosses.min()  # closest past cross
            result.iloc[i] = min(bars_since / lookback, 1.0)
        elif len(future_crosses) > 0:
            # We're before the next cross — value is -1 to 0
            bars_until = abs(future_crosses.max())  # closest future cross
            result.iloc[i] = -min(bars_until / lookback, 1.0)

    return result.clip(-1.0, 1.0)
