"""Pyramiding pattern scores: Flat Base and Moving Average Support.

These are secondary entry signals for adding to existing positions
after missing an initial breakout.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FlatBaseScore:
    """Flat base (secondary base) pattern score for pyramiding.

    Attributes:
        score: -1.0 to +1.0 — temporal proximity to breakout.
            -1.0 = early in formation (consolidation just starting)
            ~0.0 = at the breakout point
            +1.0 = just past breakout, still actionable
        quality: 0.0 to 1.0 — how well the base fits ideal flat base criteria.
            Evaluates: duration (5-6 weeks ideal), correction depth (10-15% ideal),
            tightness (low volatility within the base).
        details: Dict with pattern measurements, or None if no base found.
    """

    score: float
    quality: float
    details: dict | None


def score_flat_base(
    df: pd.DataFrame,
    min_weeks: int = 4,
    max_weeks: int = 8,
    ideal_min_weeks: int = 5,
    ideal_max_weeks: int = 6,
    ideal_correction_low: float = 0.10,
    ideal_correction_high: float = 0.15,
    correction_abs_min: float = 0.05,
    correction_abs_max: float = 0.20,
) -> FlatBaseScore:
    """Score how close the current price action is to a flat base breakout.

    A flat base is a tight, sideways consolidation (5-6 weeks, 10-15% correction)
    that forms after a prior advance. It represents institutional accumulation
    without shaking out weak holders.

    The score is -1 to +1 representing temporal position relative to breakout:
        -1.0 = consolidation just began (early in base)
        -0.5 = midway through base formation
         0.0 = at the breakout point (price crossing base ceiling)
        +0.5 = just above breakout, still valid entry
        +1.0 = extended past breakout

    Args:
        df: OHLCV DataFrame. Daily data, needs ~60+ bars for detection.
            Should include a prior advance before the base.
        min_weeks: Minimum base duration in weeks (bars = weeks * 5).
        max_weeks: Maximum base duration in weeks.
        ideal_min_weeks: Sweet spot lower bound for duration.
        ideal_max_weeks: Sweet spot upper bound for duration.
        ideal_correction_low: Ideal minimum correction depth (10%).
        ideal_correction_high: Ideal maximum correction depth (15%).
        correction_abs_min: Absolute min correction to qualify (5%).
        correction_abs_max: Absolute max correction to qualify (20%).

    Returns:
        FlatBaseScore with proximity score, quality score, and pattern details.
    """
    if len(df) < min_weeks * 5 + 10:
        return FlatBaseScore(-1.0, 0.0, None)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    min_bars = min_weeks * 5
    max_bars = max_weeks * 5

    best_score = None
    best_quality = 0.0

    # Scan for flat bases: look for a high point followed by tight consolidation
    # Search backward from current position to find the most recent base
    for lookback_start in range(min_bars, min(max_bars + 20, len(df) - 10)):
        # The base ceiling is approximately the high of the prior advance
        start_pos = len(df) - 1 - lookback_start
        if start_pos < 10:
            break

        # Find the ceiling (highest high in a window around start_pos)
        ceiling_window = high[max(0, start_pos - 5): start_pos + 5]
        ceiling = np.max(ceiling_window)
        ceiling_idx = start_pos - 5 + np.argmax(ceiling_window)

        # Base region: from ceiling to current
        base_region_start = ceiling_idx
        base_region = df.iloc[base_region_start:]

        if len(base_region) < min_bars:
            continue

        base_high = base_region["High"].max()
        base_low = base_region["Low"].min()

        # Correction depth from ceiling
        correction = (ceiling - base_low) / ceiling if ceiling > 0 else 0.0

        # Filter: correction must be in valid range
        if correction < correction_abs_min or correction > correction_abs_max:
            continue

        # Duration in weeks
        duration_bars = len(base_region)
        duration_weeks = duration_bars / 5.0

        if duration_weeks < min_weeks or duration_weeks > max_weeks + 2:
            continue

        # Tightness: weekly closes should be within a narrow range
        # Use standard deviation of closes relative to the mean
        base_closes = base_region["Close"].values
        base_mean = np.mean(base_closes)
        tightness = 1.0 - min(np.std(base_closes) / (base_mean * 0.05), 1.0)

        # --- Quality Score (0-1) ---
        # Duration quality
        if ideal_min_weeks <= duration_weeks <= ideal_max_weeks:
            duration_quality = 1.0
        elif duration_weeks < ideal_min_weeks:
            duration_quality = max(0.0, (duration_weeks - min_weeks) / (ideal_min_weeks - min_weeks))
        else:
            duration_quality = max(0.0, 1.0 - (duration_weeks - ideal_max_weeks) / (max_weeks - ideal_max_weeks))

        # Correction quality
        if ideal_correction_low <= correction <= ideal_correction_high:
            correction_quality = 1.0
        elif correction < ideal_correction_low:
            correction_quality = (correction - correction_abs_min) / (ideal_correction_low - correction_abs_min)
        else:
            correction_quality = 1.0 - (correction - ideal_correction_high) / (correction_abs_max - ideal_correction_high)

        correction_quality = float(np.clip(correction_quality, 0.0, 1.0))

        # Combined quality: 35% duration + 35% correction + 30% tightness
        quality = 0.35 * duration_quality + 0.35 * correction_quality + 0.30 * tightness

        if quality < 0.2:
            continue

        # --- Proximity Score (-1 to +1) ---
        # Breakout point = ceiling price (base ceiling / resistance)
        current_price = close[-1]
        breakout_level = ceiling

        # Normalize distance by the base range
        base_range = ceiling - base_low
        if base_range <= 0:
            continue

        price_vs_breakout = (current_price - breakout_level) / base_range

        # Map: below breakout → negative, at breakout → 0, above → positive
        # Well below (-1 base_range) → -1.0
        # At breakout → 0.0
        # One base_range above → +1.0
        proximity = float(np.clip(price_vs_breakout, -1.0, 1.0))

        # Time-based component: how far through the base are we?
        # If price is still below breakout, factor in temporal progress
        if proximity < 0:
            # Scale by how far through the expected duration we are
            time_progress = duration_bars / (ideal_max_weeks * 5)
            # Early in base = more negative, late in base = approaching 0
            temporal_factor = min(time_progress, 1.0)
            proximity = -1.0 + temporal_factor * (1.0 + proximity)

        if quality > best_quality:
            best_quality = quality
            best_score = FlatBaseScore(
                score=float(np.clip(proximity, -1.0, 1.0)),
                quality=float(np.clip(quality, 0.0, 1.0)),
                details={
                    "ceiling": float(ceiling),
                    "base_low": float(base_low),
                    "correction_pct": float(correction),
                    "duration_weeks": float(duration_weeks),
                    "tightness": float(tightness),
                    "breakout_level": float(breakout_level),
                    "current_price": float(current_price),
                    "base_start_idx": base_region_start,
                },
            )

    if best_score is None:
        return FlatBaseScore(-1.0, 0.0, None)

    return best_score


@dataclass
class MovingAverageSupportScore:
    """10-week moving average support bounce score.

    Attributes:
        score: -1.0 to +1.0 — temporal proximity to the bounce.
            -1.0 = well above the 10-week MA (not pulling back yet)
            -0.5 = pulling back toward the MA
             0.0 = at the MA, testing support
            +0.5 = bouncing off MA with volume confirmation
            +1.0 = confirmed bounce, strong support hold
        quality: 0.0 to 1.0 — quality of the support test.
            Evaluates: volume on bounce, clean touch vs break,
            prior trend strength, number of successful prior bounces.
        details: Dict with measurements, or None.
    """

    score: float
    quality: float
    details: dict | None


def score_ma_support(
    df: pd.DataFrame,
    ma_period: int = 50,
    touch_tolerance: float = 0.02,
    volume_surge_threshold: float = 1.3,
    volume_avg_period: int = 50,
    bounce_confirm_bars: int = 3,
) -> MovingAverageSupportScore:
    """Score how closely the stock is bouncing off its 10-week moving average.

    The 10-week MA (≈50-day SMA on daily charts) is a key institutional
    support level. A pullback to this line that holds on increased volume
    is a valid pyramiding entry.

    Score interpretation:
        -1.0 = price well above MA, no pullback opportunity
        -0.5 = price pulling back toward MA, approaching support
         0.0 = price is at or touching the MA line
        +0.5 = initial bounce off MA with some volume
        +1.0 = confirmed bounce with volume surge, strong support

    Args:
        df: OHLCV DataFrame. Daily data, needs 60+ bars.
        ma_period: Moving average period (50 ≈ 10 weeks on daily).
        touch_tolerance: How close to MA counts as "touching" (2% default).
        volume_surge_threshold: Volume ratio for confirmation (1.3 = 30% above avg).
        volume_avg_period: Bars for average volume calculation.
        bounce_confirm_bars: Bars after touch to look for bounce confirmation.

    Returns:
        MovingAverageSupportScore with proximity, quality, and details.
    """
    if len(df) < ma_period + 20:
        return MovingAverageSupportScore(-1.0, 0.0, None)

    close = df["Close"].values
    low = df["Low"].values
    volume = df["Volume"].values if "Volume" in df.columns else None

    # Compute 50-day SMA (≈10-week MA)
    ma = pd.Series(close).rolling(window=ma_period).mean().values

    current_price = close[-1]
    current_ma = ma[-1]

    if np.isnan(current_ma) or current_ma <= 0:
        return MovingAverageSupportScore(-1.0, 0.0, None)

    # --- Check prior trend: MA should be rising (bullish context) ---
    ma_slope_period = 20
    if len(ma) > ma_slope_period:
        ma_recent = ma[-1]
        ma_earlier = ma[-ma_slope_period]
        if not np.isnan(ma_earlier) and ma_earlier > 0:
            ma_slope = (ma_recent - ma_earlier) / ma_earlier
        else:
            ma_slope = 0.0
    else:
        ma_slope = 0.0

    # If MA is declining, this isn't a support setup
    if ma_slope < -0.01:
        return MovingAverageSupportScore(-1.0, 0.0, None)

    # --- Find the touch/pullback ---
    # Look at last 10 bars for the closest approach to the MA
    lookback = min(10, len(df) - ma_period)
    recent_lows = low[-lookback:]
    recent_ma = ma[-lookback:]
    recent_close = close[-lookback:]

    # Distance from MA (as fraction of price)
    distances = (recent_lows - recent_ma) / recent_ma

    # Find the closest touch (minimum distance, preferably touching or slightly below)
    min_dist_idx = np.argmin(np.abs(distances))
    min_distance = distances[min_dist_idx]
    touch_bar = len(df) - lookback + min_dist_idx

    # Was there a valid touch? (within tolerance)
    touched = abs(min_distance) <= touch_tolerance
    undercut = min_distance < -touch_tolerance  # briefly dipped below

    if not touched and min_distance > touch_tolerance:
        # Price hasn't pulled back near MA — still above
        # Score based on how far above we are
        above_pct = (current_price - current_ma) / current_ma
        if above_pct > 0.10:
            proximity = -1.0  # well above, no pullback
        else:
            proximity = -1.0 + (1.0 - above_pct / 0.10) * 0.5  # approaching

        return MovingAverageSupportScore(
            score=float(np.clip(proximity, -1.0, 0.0)),
            quality=0.0,
            details={
                "current_price": float(current_price),
                "ma_value": float(current_ma),
                "distance_pct": float(above_pct),
                "ma_slope": float(ma_slope),
                "touch_bar": None,
            },
        )

    # --- We have a touch. Now evaluate the bounce. ---
    bars_since_touch = len(df) - 1 - touch_bar

    # Proximity score: at touch = 0, confirmed bounce = positive
    if bars_since_touch == 0:
        # Currently at the MA
        proximity = 0.0
    elif bars_since_touch <= bounce_confirm_bars:
        # In the confirmation window
        # Is price moving up from the touch?
        price_at_touch = recent_close[min_dist_idx]
        bounce_pct = (current_price - price_at_touch) / price_at_touch if price_at_touch > 0 else 0.0
        if bounce_pct > 0:
            proximity = min(bounce_pct / 0.03, 1.0) * 0.5  # ramp toward +0.5
        else:
            proximity = 0.0  # still at MA, not bouncing yet
    else:
        # Past confirmation window
        price_at_touch_close = recent_close[min_dist_idx]
        bounce_pct = (current_price - price_at_touch_close) / price_at_touch_close if price_at_touch_close > 0 else 0.0
        if bounce_pct > 0.02 and current_price > current_ma:
            proximity = 0.5 + min(bounce_pct / 0.05, 1.0) * 0.5  # confirmed bounce
        elif current_price < current_ma:
            proximity = -0.5  # broke below — failed support
        else:
            proximity = 0.2

    # --- Quality Score (0-1) ---
    # Components: volume on bounce, clean touch, trend strength, prior bounces

    # Volume quality: was there a volume surge on the bounce day(s)?
    volume_quality = 0.5  # neutral default
    volume_ratio = None
    if volume is not None and len(volume) > volume_avg_period:
        avg_vol = np.mean(volume[-(volume_avg_period + 1):-1])
        if avg_vol > 0:
            # Check volume in the bars around/after the touch
            bounce_start = max(touch_bar, 0)
            bounce_end = min(touch_bar + bounce_confirm_bars + 1, len(df))
            bounce_volumes = volume[bounce_start:bounce_end]
            if len(bounce_volumes) > 0:
                max_bounce_vol = np.max(bounce_volumes)
                volume_ratio = max_bounce_vol / avg_vol
                if volume_ratio >= volume_surge_threshold:
                    volume_quality = min(volume_ratio / 2.0, 1.0)
                else:
                    volume_quality = (volume_ratio - 1.0) / (volume_surge_threshold - 1.0) * 0.5

    volume_quality = float(np.clip(volume_quality, 0.0, 1.0))

    # Touch cleanliness: slight touch is better than big undercut
    if undercut:
        undercut_pct = abs(min_distance)
        # Small undercut (<1%) is fine, larger is worse
        touch_quality = max(0.0, 1.0 - undercut_pct / 0.03)
    elif touched:
        touch_quality = 1.0
    else:
        touch_quality = 0.5

    # Trend quality: rising MA = good context
    trend_quality = float(np.clip(ma_slope / 0.05, 0.0, 1.0))

    # Prior bounces: count successful touches in the last 100 bars
    prior_bounces = 0
    if len(df) > ma_period + 50:
        scan_start = max(ma_period, len(df) - 100)
        for i in range(scan_start, len(df) - lookback):
            if not np.isnan(ma[i]) and ma[i] > 0:
                bar_dist = abs(low[i] - ma[i]) / ma[i]
                if bar_dist <= touch_tolerance:
                    # Check if price recovered in next 5 bars
                    recovery_end = min(i + 5, len(df))
                    if np.max(close[i:recovery_end]) > ma[i] * 1.01:
                        prior_bounces += 1

    bounce_history_quality = min(prior_bounces / 3.0, 1.0)  # 3+ prior bounces = max

    # Combined quality
    quality = (
        0.35 * volume_quality
        + 0.25 * touch_quality
        + 0.20 * trend_quality
        + 0.20 * bounce_history_quality
    )

    return MovingAverageSupportScore(
        score=float(np.clip(proximity, -1.0, 1.0)),
        quality=float(np.clip(quality, 0.0, 1.0)),
        details={
            "current_price": float(current_price),
            "ma_value": float(current_ma),
            "distance_pct": float(min_distance),
            "ma_slope": float(ma_slope),
            "touch_bar": int(touch_bar),
            "bars_since_touch": int(bars_since_touch),
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "prior_bounces": int(prior_bounces),
            "undercut": bool(undercut),
        },
    )
