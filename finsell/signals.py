"""Signal generation from indicators and patterns."""

import numpy as np
import pandas as pd
import pandas_ta as ta


def rsi_signals(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.DataFrame:
    """Generate buy/sell signals from RSI crossings.

    Buy when RSI crosses above oversold, sell when RSI crosses below overbought.
    """
    rsi_col = [c for c in df.columns if "RSI" in c]
    if not rsi_col:
        raise ValueError("RSI column not found. Run add_indicators first.")

    rsi = df[rsi_col[0]]
    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals.loc[(rsi > oversold) & (rsi.shift(1) <= oversold), "signal"] = 1  # Buy
    signals.loc[(rsi < overbought) & (rsi.shift(1) >= overbought), "signal"] = -1  # Sell
    return signals


def macd_crossover_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals from MACD / signal line crossovers."""
    macd_col = [c for c in df.columns if c.startswith("MACD_")]
    signal_col = [c for c in df.columns if c.startswith("MACDs_")]

    if not macd_col or not signal_col:
        raise ValueError("MACD columns not found. Run add_indicators first.")

    macd = df[macd_col[0]]
    signal_line = df[signal_col[0]]

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals.loc[(macd > signal_line) & (macd.shift(1) <= signal_line.shift(1)), "signal"] = 1
    signals.loc[(macd < signal_line) & (macd.shift(1) >= signal_line.shift(1)), "signal"] = -1
    return signals


def bollinger_band_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate signals from Bollinger Band touches.

    Buy when price touches lower band, sell when price touches upper band.
    """
    lower = [c for c in df.columns if c.startswith("BBL_")]
    upper = [c for c in df.columns if c.startswith("BBU_")]

    if not lower or not upper:
        raise ValueError("Bollinger Band columns not found. Run add_indicators first.")

    signals = pd.DataFrame(index=df.index)
    signals["signal"] = 0
    signals.loc[df["Close"] <= df[lower[0]], "signal"] = 1
    signals.loc[df["Close"] >= df[upper[0]], "signal"] = -1
    return signals


def trailing_stop_signal(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 21,
    col: str = "Close",
) -> pd.Series:
    """Generate a trailing stop signal (0 to 1) using 10-day and 21-day EMAs.

    The signal measures how urgently a trailing stop is being approached or
    breached. Uses the relationship between price, the fast EMA (10), and
    the slow EMA (21) to gauge stop pressure:

        0.0 = Safe — price well above both EMAs, no stop pressure
        0.5 = Caution — price has broken the fast EMA, nearing slow EMA
        1.0 = Stop triggered — price has broken below the slow EMA

    The signal ramps smoothly between these levels based on where price sits
    relative to the two EMA envelopes.

    Args:
        df: OHLCV DataFrame.
        fast_period: Fast EMA period (10 default — tight trailing stop).
        slow_period: Slow EMA period (21 default — wider stop confirmation).
        col: Price column to evaluate.

    Returns:
        Series with values 0.0 to 1.0.
    """
    price = df[col].values
    ema_fast = ta.ema(df[col], length=fast_period).values
    ema_slow = ta.ema(df[col], length=slow_period).values

    result = np.full(len(df), np.nan)

    for i in range(len(df)):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            continue

        fast = ema_fast[i]
        slow = ema_slow[i]
        p = price[i]

        # Distance between the two EMAs defines the "buffer zone"
        ema_gap = fast - slow

        if p >= fast:
            # Price above fast EMA — safe zone
            # Score based on how far above: further above = more safe (closer to 0)
            distance_above = p - fast
            buffer = abs(ema_gap) if ema_gap != 0 else fast * 0.01
            # Ramp from 0.2 (just above fast) to 0.0 (well above)
            result[i] = max(0.0, 0.2 - min(distance_above / buffer, 1.0) * 0.2)
        elif p >= slow:
            # Price between fast and slow EMA — caution zone
            # Ramp from 0.2 (at fast EMA) to 0.7 (at slow EMA)
            zone_size = fast - slow if fast > slow else slow * 0.01
            position = (fast - p) / zone_size if zone_size > 0 else 1.0
            position = min(max(position, 0.0), 1.0)
            result[i] = 0.2 + position * 0.5
        else:
            # Price below slow EMA — stop zone
            # Ramp from 0.7 (just below slow) to 1.0 (well below)
            distance_below = slow - p
            buffer = abs(ema_gap) if ema_gap != 0 else slow * 0.01
            breach_severity = min(distance_below / buffer, 1.0) if buffer > 0 else 1.0
            result[i] = 0.7 + breach_severity * 0.3

    return pd.Series(np.clip(result, 0.0, 1.0), index=df.index, name="trailing_stop")
