"""Double bottom with undercut buy signal scoring.

The undercut variant is where the second low dips slightly below the first low
(shaking out weak holders) before reversing. The buy point triggers when price
crosses the middle peak (pivot/neckline) on volume.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from finsell.patterns import find_swing_highs, find_swing_lows


@dataclass
class DoubleBottomUndercutScore:
    """Double bottom with undercut buy signal score.

    Attributes:
        score: -1.0 to +1.0 — proximity to the buy signal.
            -1.0 = first low forming, pattern far from complete
            -0.5 = second low forming / undercut in progress
             0.0 = at the buy point (price crossing neckline)
            +0.5 = just past breakout, still actionable
            +1.0 = confirmed breakout with volume
        quality: 0.0 to 1.0 — how well the pattern fits ideal criteria.
            Evaluates: undercut depth, time between lows, volume on breakout,
            neckline clarity.
        details: Dict with pattern measurements, or None.
    """

    score: float
    quality: float
    details: dict | None


def score_double_bottom_undercut(
    df: pd.DataFrame,
    order: int = 8,
    undercut_min_pct: float = 0.005,
    undercut_max_pct: float = 0.05,
    min_bars_between_lows: int = 15,
    max_bars_between_lows: int = 100,
    volume_surge_threshold: float = 1.4,
    volume_avg_period: int = 50,
    pivot_offset: float = 0.10,
) -> DoubleBottomUndercutScore:
    """Score a double bottom with undercut pattern and buy signal proximity.

    Pattern requirements:
      1. First swing low establishes initial support
      2. Rally to a middle peak (neckline/pivot)
      3. Second decline that UNDERCUTS the first low (goes 0.5-5% below)
      4. Reversal from undercut back above first low
      5. Buy signal: price crosses above neckline + $0.10 on volume 40%+ above avg

    The undercut is bullish because it:
      - Shakes out stop-loss sellers at obvious support
      - Traps bears who short the "breakdown"
      - Creates a vacuum for the reversal move

    Score interpretation:
        -1.0 = only one low exists, pattern incomplete
        -0.5 = second low / undercut forming
         0.0 = at the buy point (neckline crossing)
        +0.5 = just above neckline, initial breakout
        +1.0 = confirmed breakout with volume surge

    Args:
        df: OHLCV DataFrame. Daily data, needs 60+ bars.
        order: Swing detection sensitivity.
        undercut_min_pct: Minimum undercut below first low (0.5%).
        undercut_max_pct: Maximum undercut below first low (5%).
        min_bars_between_lows: Minimum separation between the two lows.
        max_bars_between_lows: Maximum separation between the two lows.
        volume_surge_threshold: Volume ratio for breakout confirmation (1.4 = 40%).
        volume_avg_period: Bars for average volume.
        pivot_offset: Dollar offset above neckline for buy trigger.

    Returns:
        DoubleBottomUndercutScore with score, quality, and details.
    """
    if len(df) < 40:
        return DoubleBottomUndercutScore(-1.0, 0.0, None)

    close = df["Close"].values
    low = df["Low"].values
    high = df["High"].values
    volume = df["Volume"].values if "Volume" in df.columns else None

    current_price = close[-1]
    current_pos = len(df) - 1

    swing_lows = find_swing_lows(df, order=order)
    swing_highs = find_swing_highs(df, order=order)

    low_indices = df.index[swing_lows]
    low_prices = df.loc[low_indices, "Low"]
    high_indices = df.index[swing_highs]
    high_prices = df.loc[high_indices, "High"]

    if len(low_prices) < 2:
        return DoubleBottomUndercutScore(-1.0, 0.0, None)

    best_result = None
    best_combined = -999.0

    # Search for double bottom with undercut patterns
    for i in range(len(low_prices) - 1):
        first_low_idx = low_indices[i]
        first_low_price = low_prices.iloc[i]
        first_low_pos = df.index.get_loc(first_low_idx)

        for j in range(i + 1, len(low_prices)):
            second_low_idx = low_indices[j]
            second_low_price = low_prices.iloc[j]
            second_low_pos = df.index.get_loc(second_low_idx)

            bars_between = second_low_pos - first_low_pos
            if bars_between < min_bars_between_lows or bars_between > max_bars_between_lows:
                continue

            # --- Undercut check: second low must be BELOW first low ---
            undercut_amount = (first_low_price - second_low_price) / first_low_price
            if undercut_amount < undercut_min_pct or undercut_amount > undercut_max_pct:
                continue

            # --- Find neckline: highest high between the two lows ---
            between_highs = high[first_low_pos:second_low_pos + 1]
            if len(between_highs) == 0:
                continue
            neckline = float(np.max(between_highs))
            neckline_pos = first_low_pos + np.argmax(between_highs)

            # Neckline must be meaningfully above the lows
            neckline_height = (neckline - first_low_price) / first_low_price
            if neckline_height < 0.03:
                continue  # Too shallow

            # --- Check for reversal from undercut ---
            # After second low, price should recover above first low level
            bars_after_second = current_pos - second_low_pos
            if bars_after_second < 1:
                continue

            # Did price recover above the first low after the undercut?
            recovery_region = close[second_low_pos:]
            recovered = np.any(recovery_region > first_low_price)
            if not recovered:
                # Still below first low — undercut hasn't reversed yet
                proximity = -0.5
                quality = 0.3
                volume_ratio = None
            else:
                # --- Buy point: neckline + offset ---
                buy_point = neckline + pivot_offset

                price_vs_buy = (current_price - buy_point) / (neckline - first_low_price)

                # Proximity score
                if current_price < first_low_price:
                    # Still below first low (in the undercut zone)
                    proximity = -0.5
                elif current_price < neckline * 0.98:
                    # Between lows and neckline — approaching buy point
                    progress = (current_price - first_low_price) / (neckline - first_low_price)
                    proximity = -0.5 + 0.5 * progress
                elif current_price < buy_point:
                    # Near neckline but below buy trigger
                    proximity = 0.0 - 0.1 * ((buy_point - current_price) / pivot_offset)
                elif current_price <= buy_point * 1.05:
                    # At or just above buy point — actionable
                    proximity = 0.0 + 0.5 * min((current_price - buy_point) / (buy_point * 0.03), 1.0)
                else:
                    # Extended above buy point
                    extension = (current_price - buy_point) / buy_point
                    if extension < 0.05:
                        proximity = 0.5 + 0.5 * (extension / 0.05)
                    else:
                        proximity = 1.0

                # --- Quality assessment ---
                # Undercut quality: ideal is 1-3% below first low
                ideal_undercut_mid = 0.02
                undercut_quality = 1.0 - min(abs(undercut_amount - ideal_undercut_mid) / 0.03, 1.0)

                # Symmetry: time from first low to neckline vs neckline to second low
                time_first_half = neckline_pos - first_low_pos
                time_second_half = second_low_pos - neckline_pos
                if time_first_half > 0 and time_second_half > 0:
                    time_ratio = min(time_first_half, time_second_half) / max(time_first_half, time_second_half)
                else:
                    time_ratio = 0.5
                symmetry_quality = time_ratio

                # Volume on breakout (if price is near/above buy point)
                volume_quality = 0.5
                volume_ratio = None
                if volume is not None and current_price >= neckline * 0.98:
                    avg_vol_start = max(0, len(volume) - 1 - volume_avg_period)
                    avg_vol = np.mean(volume[avg_vol_start:-1])
                    if avg_vol > 0:
                        current_vol = volume[-1]
                        volume_ratio = current_vol / avg_vol
                        if volume_ratio >= volume_surge_threshold:
                            volume_quality = min(volume_ratio / 2.0, 1.0)
                        elif volume_ratio >= 1.0:
                            volume_quality = 0.3 + 0.2 * ((volume_ratio - 1.0) / (volume_surge_threshold - 1.0))
                        else:
                            volume_quality = 0.2

                    # Volume confirmation bonus for proximity
                    if volume_ratio and volume_ratio >= volume_surge_threshold and current_price >= buy_point:
                        proximity = min(proximity + 0.2, 1.0)

                # Neckline clarity: sharper peak between lows is better
                # (less rounded = clearer resistance level)
                peak_region = high[max(neckline_pos - 3, first_low_pos):min(neckline_pos + 4, second_low_pos)]
                if len(peak_region) > 0:
                    peak_sharpness = (neckline - np.mean(peak_region)) / (neckline - first_low_price + 1e-10)
                    neckline_quality = float(np.clip(peak_sharpness * 5, 0.3, 1.0))
                else:
                    neckline_quality = 0.5

                quality = (
                    0.30 * undercut_quality
                    + 0.25 * volume_quality
                    + 0.25 * symmetry_quality
                    + 0.20 * neckline_quality
                )

            # Rank this pattern
            combined = quality + (1.0 - abs(proximity)) * 0.5
            if combined > best_combined:
                best_combined = combined
                best_result = DoubleBottomUndercutScore(
                    score=float(np.clip(proximity, -1.0, 1.0)),
                    quality=float(np.clip(quality, 0.0, 1.0)),
                    details={
                        "first_low": float(first_low_price),
                        "second_low": float(second_low_price),
                        "undercut_pct": float(undercut_amount),
                        "neckline": float(neckline),
                        "buy_point": float(neckline + pivot_offset),
                        "current_price": float(current_price),
                        "bars_between_lows": int(bars_between),
                        "volume_ratio": float(volume_ratio) if volume_ratio else None,
                        "first_low_idx": str(first_low_idx),
                        "second_low_idx": str(second_low_idx),
                    },
                )

    if best_result is None:
        return DoubleBottomUndercutScore(-1.0, 0.0, None)

    return best_result
