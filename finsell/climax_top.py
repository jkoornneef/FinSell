"""Climax top detection scoring.

Identifies exhaustion moves where a stock surges vertically after a
prolonged advance — a classic sell/avoid signal.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ClimaxTopScore:
    """Climax top pattern score.

    Attributes:
        score: -1.0 to +1.0 — temporal proximity to the climax.
            -1.0 = extended advance but no vertical surge yet
            -0.5 = acceleration starting (steepening trend)
             0.0 = in the climax surge right now
            +0.5 = surge peaking / exhaustion signs appearing
            +1.0 = climax confirmed (reversal underway)
        quality: 0.0 to 1.0 — how well the move fits climax top criteria.
            Evaluates: prior advance duration, surge magnitude, surge speed,
            volume climax, widening daily ranges.
        details: Dict with pattern measurements, or None.
    """

    score: float
    quality: float
    details: dict | None


def score_climax_top(
    df: pd.DataFrame,
    min_prior_advance_months: int = 4,
    min_prior_advance_pct: float = 0.30,
    surge_min_pct: float = 0.25,
    surge_max_weeks: int = 3,
    volume_climax_threshold: float = 2.0,
) -> ClimaxTopScore:
    """Score how closely current price action matches a climax top.

    A climax top occurs when a stock that has advanced for many months
    suddenly surges 25-50%+ in just 1-3 weeks on huge volume with
    widening daily ranges — signaling exhaustion and institutional selling
    into strength.

    Score interpretation:
        -1.0 = stock is in a prolonged advance but no vertical surge
        -0.5 = trend accelerating, early warning
         0.0 = in the middle of the climax surge
        +0.5 = surge appears to be peaking (exhaustion bars)
        +1.0 = climax top confirmed (price reversing from peak)

    Args:
        df: OHLCV DataFrame. Daily data, needs 100+ bars (covers months of advance).
        min_prior_advance_months: Minimum months of advance before climax qualifies.
        min_prior_advance_pct: Minimum total advance before the surge (30% default).
        surge_min_pct: Minimum surge gain to qualify (25%).
        surge_max_weeks: Maximum surge duration in weeks (3 weeks = 15 bars).
        volume_climax_threshold: Volume ratio vs average for climax confirmation.

    Returns:
        ClimaxTopScore with proximity score, quality, and details.
    """
    min_bars = min_prior_advance_months * 21  # ~21 trading days per month
    if len(df) < min_bars + 10:
        return ClimaxTopScore(-1.0, 0.0, None)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values if "Volume" in df.columns else None

    current_price = close[-1]
    surge_max_bars = surge_max_weeks * 5

    # --- Find the peak (highest close in recent data) ---
    # Look at last 30 bars for the peak
    recent_window = min(30, len(df) - min_bars)
    recent_highs = close[-recent_window:]
    peak_offset = np.argmax(recent_highs)
    peak_pos = len(df) - recent_window + peak_offset
    peak_price = close[peak_pos]

    # --- Identify the surge start ---
    # Walk backward from peak to find where the steep move began
    # The surge start is where the slope dramatically increases
    best_surge_start = None
    best_surge_pct = 0.0

    for lookback in range(5, surge_max_bars + 1):
        start_pos = peak_pos - lookback
        if start_pos < min_bars:
            break
        start_price = close[start_pos]
        if start_price <= 0:
            continue
        gain = (peak_price - start_price) / start_price
        if gain >= surge_min_pct and gain > best_surge_pct:
            best_surge_pct = gain
            best_surge_start = start_pos

    # Also check if we're currently IN a surge (peak might be today)
    if best_surge_start is None:
        # Check if current price IS the surge — look back up to surge_max_bars
        for lookback in range(5, surge_max_bars + 1):
            start_pos = len(df) - 1 - lookback
            if start_pos < min_bars:
                break
            start_price = close[start_pos]
            if start_price <= 0:
                continue
            gain = (current_price - start_price) / start_price
            if gain >= surge_min_pct and gain > best_surge_pct:
                best_surge_pct = gain
                best_surge_start = start_pos
                peak_pos = len(df) - 1
                peak_price = current_price

    # --- Check prior advance (before the surge) ---
    if best_surge_start is not None and best_surge_start >= min_bars:
        # Prior advance: from the lowest low in the months before surge start
        advance_region = close[:best_surge_start]
        advance_low = np.min(advance_region)
        advance_low_pos = np.argmin(advance_region)
        surge_start_price = close[best_surge_start]

        prior_advance_pct = (surge_start_price - advance_low) / advance_low if advance_low > 0 else 0.0
        prior_advance_months = (best_surge_start - advance_low_pos) / 21.0

        has_prior_advance = (
            prior_advance_pct >= min_prior_advance_pct
            and prior_advance_months >= min_prior_advance_months
        )
    elif best_surge_start is None:
        # No surge found — check if we're in an extended advance (pre-climax)
        advance_low = np.min(close[:min_bars])
        prior_advance_pct = (current_price - advance_low) / advance_low if advance_low > 0 else 0.0

        if prior_advance_pct >= min_prior_advance_pct:
            # In extended advance but no vertical surge yet
            # Check if trend is accelerating (slope increasing)
            slope_recent = (close[-1] - close[-10]) / close[-10] if close[-10] > 0 else 0.0
            slope_earlier = (close[-10] - close[-20]) / close[-20] if len(close) > 20 and close[-20] > 0 else 0.0

            acceleration = slope_recent - slope_earlier

            if acceleration > 0.02:
                # Trend steepening — early warning
                return ClimaxTopScore(
                    score=-0.5,
                    quality=0.3,
                    details={
                        "prior_advance_pct": float(prior_advance_pct),
                        "acceleration": float(acceleration),
                        "surge_pct": None,
                        "surge_weeks": None,
                        "peak_price": float(current_price),
                        "current_vs_peak_pct": 0.0,
                        "volume_ratio": None,
                        "widening_ranges": False,
                    },
                )

        return ClimaxTopScore(-1.0, 0.0, None)
    else:
        has_prior_advance = False

    if not has_prior_advance:
        return ClimaxTopScore(-1.0, 0.0, None)

    # --- We have a valid surge with prior advance. Score it. ---
    surge_bars = peak_pos - best_surge_start
    surge_weeks = surge_bars / 5.0

    # --- Volume analysis ---
    volume_ratio = None
    has_volume_climax = False
    if volume is not None:
        avg_vol_start = max(0, best_surge_start - 50)
        avg_vol = np.mean(volume[avg_vol_start:best_surge_start])
        if avg_vol > 0:
            surge_vol = np.mean(volume[best_surge_start:peak_pos + 1])
            volume_ratio = surge_vol / avg_vol
            # Also check for individual climax days (huge single-day volume)
            max_surge_vol = np.max(volume[best_surge_start:peak_pos + 1])
            if max_surge_vol / avg_vol >= volume_climax_threshold:
                has_volume_climax = True

    # --- Widening daily ranges ---
    # Compare average daily range during surge vs before
    range_before = np.mean(high[best_surge_start - 20:best_surge_start] -
                           low[best_surge_start - 20:best_surge_start])
    range_during = np.mean(high[best_surge_start:peak_pos + 1] -
                           low[best_surge_start:peak_pos + 1])
    widening_ranges = range_during > range_before * 1.5 if range_before > 0 else False

    # --- Proximity Score (-1 to +1) ---
    bars_past_peak = len(df) - 1 - peak_pos
    current_vs_peak = (current_price - peak_price) / peak_price if peak_price > 0 else 0.0

    if bars_past_peak == 0 and current_price >= peak_price * 0.98:
        # At or near the peak right now
        # If still surging upward (within surge window), score 0.0
        # If showing exhaustion signs, score toward +0.5
        last_3_ranges = high[-3:] - low[-3:]
        if len(last_3_ranges) == 3 and range_before > 0:
            if np.mean(last_3_ranges) > range_before * 2.0:
                proximity = 0.3  # exhaustion bars
            else:
                proximity = 0.0  # mid-surge
        else:
            proximity = 0.0
    elif bars_past_peak <= 5 and current_vs_peak > -0.05:
        # Just past peak, small pullback — peaking
        proximity = 0.5
    elif bars_past_peak > 5 or current_vs_peak < -0.05:
        # Confirmed reversal from peak
        reversal_strength = min(abs(current_vs_peak) / 0.10, 1.0)
        proximity = 0.5 + 0.5 * reversal_strength
    else:
        # In the surge, before peak
        surge_progress = (len(df) - 1 - best_surge_start) / max(surge_bars, 1)
        proximity = -1.0 + surge_progress

    # --- Quality Score (0-1) ---
    # Surge magnitude quality (25%→0.5, 50%→1.0, beyond saturates)
    surge_mag_quality = float(np.clip((best_surge_pct - 0.20) / 0.30, 0.0, 1.0))

    # Surge speed quality (1 week → 1.0, 3 weeks → 0.5, slower → lower)
    speed_quality = float(np.clip(1.0 - (surge_weeks - 1.0) / 3.0, 0.3, 1.0))

    # Volume quality
    vol_quality = 0.5
    if volume_ratio is not None:
        vol_quality = float(np.clip((volume_ratio - 1.0) / 1.5, 0.0, 1.0))
    if has_volume_climax:
        vol_quality = min(vol_quality + 0.2, 1.0)

    # Range widening quality
    range_quality = 0.7 if widening_ranges else 0.3

    # Prior advance quality (longer/stronger prior advance = more climax-like)
    advance_quality = float(np.clip(prior_advance_pct / 1.0, 0.3, 1.0))

    quality = (
        0.30 * surge_mag_quality
        + 0.20 * speed_quality
        + 0.20 * vol_quality
        + 0.15 * range_quality
        + 0.15 * advance_quality
    )

    return ClimaxTopScore(
        score=float(np.clip(proximity, -1.0, 1.0)),
        quality=float(np.clip(quality, 0.0, 1.0)),
        details={
            "prior_advance_pct": float(prior_advance_pct),
            "prior_advance_months": float(prior_advance_months),
            "surge_pct": float(best_surge_pct),
            "surge_weeks": float(surge_weeks),
            "peak_price": float(peak_price),
            "current_vs_peak_pct": float(current_vs_peak),
            "volume_ratio": float(volume_ratio) if volume_ratio is not None else None,
            "has_volume_climax": bool(has_volume_climax),
            "widening_ranges": bool(widening_ranges),
        },
    )
