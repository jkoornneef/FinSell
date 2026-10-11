"""Cup and handle pattern scoring with proximity awareness."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from finsell.patterns import find_swing_highs, find_swing_lows


@dataclass
class CupAndHandleScore:
    """Three-dimensional cup and handle score.

    Attributes:
        depth_score: 0.0 to 1.0 — shallow to deep cup.
            0.0 = barely any retracement, 1.0 = ideal depth (~12-20% range).
            Cups too shallow or too deep score lower.
        shape_score: 0.0 to 1.0 — V-shaped (sharp) to U-shaped (curved).
            0.0 = sharp V-recovery, 1.0 = perfectly rounded base.
        proximity_score: -3.0 to +3.0 — temporal distance to the breakout.
            -3.0 = very early formation, pattern far from completing
            -1.0 = handle forming, watch closely
             0.0 = at the breakout point (right at rim after handle)
            +1.0 = just past breakout, still actionable
            +3.0 = well past breakout, too late
        best_pattern: Dict of the best-scoring pattern details, or None.
    """

    depth_score: float
    shape_score: float
    proximity_score: float
    best_pattern: dict | None


def score_cup_and_handle(
    df: pd.DataFrame,
    order: int = 10,
    ideal_depth_low: float = 0.12,
    ideal_depth_high: float = 0.20,
    depth_abs_min: float = 0.05,
    depth_abs_max: float = 0.35,
    rim_tolerance: float = 0.05,
) -> CupAndHandleScore:
    """Score how close the current price action is to a cup and handle pattern.

    Returns three scores:
        1. depth_score (0-1): Quality of cup depth. Peaks at ideal range.
        2. shape_score (0-1): How rounded/U-shaped vs sharp/V-shaped.
        3. proximity_score (-3 to +3): Where we are in time relative to breakout.

    Args:
        df: OHLCV DataFrame (needs enough history — 60+ bars recommended)
        order: Swing detection sensitivity
        ideal_depth_low: Lower bound of ideal depth range (12%)
        ideal_depth_high: Upper bound of ideal depth range (20%)
        depth_abs_min: Minimum depth to consider (5%)
        depth_abs_max: Maximum depth to consider (35%)
        rim_tolerance: How close rims need to be (relaxed for scoring vs detection)

    Returns:
        CupAndHandleScore with all three dimensions.
    """
    if len(df) < 20:
        return CupAndHandleScore(0.0, 0.0, -3.0, None)

    swing_highs = find_swing_highs(df, order=order)
    swing_lows = find_swing_lows(df, order=order)

    high_indices = df.index[swing_highs]
    high_prices = df.loc[high_indices, "High"]
    low_indices = df.index[swing_lows]
    low_prices = df.loc[low_indices, "Low"]

    if len(high_prices) < 2 or len(low_prices) < 1:
        return CupAndHandleScore(0.0, 0.0, -3.0, None)

    current_pos = len(df) - 1
    current_price = df["Close"].iloc[-1]

    best_score = None
    best_total = -1.0

    for i in range(len(high_prices) - 1):
        left_rim_idx = high_indices[i]
        left_rim = high_prices.iloc[i]
        left_rim_pos = df.index.get_loc(left_rim_idx)

        for j in range(i + 1, len(high_prices)):
            right_rim_idx = high_indices[j]
            right_rim = high_prices.iloc[j]
            right_rim_pos = df.index.get_loc(right_rim_idx)

            # Relaxed rim check for scoring (we grade it, not filter it)
            rim_diff = abs(right_rim - left_rim) / left_rim
            if rim_diff > rim_tolerance * 2:
                continue

            # Find cup bottom between rims
            cup_lows = low_prices[
                (low_indices > left_rim_idx) & (low_indices < right_rim_idx)
            ]
            if cup_lows.empty:
                continue

            cup_bottom = cup_lows.min()
            bottom_idx = cup_lows.idxmin()
            bottom_pos = df.index.get_loc(bottom_idx)
            rim_avg = (left_rim + right_rim) / 2
            cup_depth_pct = (rim_avg - cup_bottom) / rim_avg
            cup_len = right_rim_pos - left_rim_pos

            if cup_len < 5:
                continue

            # --- DEPTH SCORE (0-1) ---
            if cup_depth_pct < depth_abs_min or cup_depth_pct > depth_abs_max:
                depth_score = 0.0
            elif ideal_depth_low <= cup_depth_pct <= ideal_depth_high:
                # In the sweet spot
                depth_score = 1.0
            elif cup_depth_pct < ideal_depth_low:
                # Too shallow — ramp up from abs_min
                depth_score = (cup_depth_pct - depth_abs_min) / (ideal_depth_low - depth_abs_min)
            else:
                # Too deep — ramp down toward abs_max
                depth_score = 1.0 - (cup_depth_pct - ideal_depth_high) / (depth_abs_max - ideal_depth_high)

            depth_score = float(np.clip(depth_score, 0.0, 1.0))

            # --- SHAPE SCORE (0-1) ---
            # Roundedness: fraction of bars in lower third of the cup range
            cup_closes = df.iloc[left_rim_pos:right_rim_pos + 1]["Close"].values
            price_range = rim_avg - cup_bottom
            if price_range <= 0:
                continue

            lower_third_threshold = cup_bottom + price_range / 3
            bars_in_lower_third = np.sum(cup_closes <= lower_third_threshold)
            roundedness = bars_in_lower_third / cup_len

            # Symmetry: how centered is the bottom
            relative_bottom = (bottom_pos - left_rim_pos) / cup_len
            symmetry = 1.0 - abs(relative_bottom - 0.5) * 2  # 1.0 = perfectly centered

            # Rim evenness contributes to shape quality
            rim_evenness = 1.0 - min(rim_diff / rim_tolerance, 1.0)

            # Combine: roundedness dominates, symmetry and rim evenness contribute
            shape_score = float(np.clip(
                0.60 * min(roundedness / 0.5, 1.0) +  # roundedness peaks ~0.5
                0.25 * symmetry +
                0.15 * rim_evenness,
                0.0, 1.0
            ))

            # --- PROXIMITY SCORE (-3 to +3) ---
            # Define key time segments relative to the cup length:
            #   Expected handle zone: right_rim_pos to right_rim_pos + cup_len*0.3
            #   Breakout point: end of handle zone
            handle_expected_len = int(cup_len * 0.25)
            breakout_pos = right_rim_pos + handle_expected_len

            # Distance from current position to breakout, normalized by cup_len
            bars_from_breakout = current_pos - breakout_pos
            time_segment = bars_from_breakout / (cup_len / 3)  # 1 segment = 1/3 of cup length

            proximity_score = float(np.clip(time_segment, -3.0, 3.0))

            # --- Look for actual handle ---
            handle_low = None
            handle_low_idx = None
            if current_pos > right_rim_pos:
                handle_search_end = min(right_rim_pos + int(cup_len * 0.5), len(df))
                handle_region = df.iloc[right_rim_pos + 1:handle_search_end]
                if len(handle_region) > 0:
                    handle_low_pos = handle_region["Low"].idxmin()
                    handle_low = df.loc[handle_low_pos, "Low"]
                    handle_low_idx = handle_low_pos
                    # Handle shouldn't retrace more than 50% of cup depth
                    handle_depth = right_rim - handle_low
                    if handle_depth > price_range * 0.5:
                        # Handle too deep — penalize proximity (pattern failing)
                        proximity_score = min(proximity_score, -2.0)

            # Combined quality for ranking
            total = depth_score * 0.3 + shape_score * 0.4 + (1.0 - abs(proximity_score) / 3.0) * 0.3

            if total > best_total:
                best_total = total
                best_score = CupAndHandleScore(
                    depth_score=depth_score,
                    shape_score=shape_score,
                    proximity_score=proximity_score,
                    best_pattern={
                        "left_rim_idx": left_rim_idx,
                        "right_rim_idx": right_rim_idx,
                        "bottom_idx": bottom_idx,
                        "handle_low_idx": handle_low_idx,
                        "left_rim": left_rim,
                        "right_rim": right_rim,
                        "cup_bottom": cup_bottom,
                        "handle_low": handle_low,
                        "cup_depth_pct": cup_depth_pct,
                        "roundedness": roundedness,
                        "symmetry": symmetry,
                        "cup_len_bars": cup_len,
                        "breakout_pos": breakout_pos,
                        "current_pos": current_pos,
                    },
                )

    if best_score is None:
        return CupAndHandleScore(0.0, 0.0, -3.0, None)

    return best_score


@dataclass
class PivotBuyScore:
    """Pivot point buy signal strength.

    Attributes:
        score: 0.0 to 2.0 — strength of the buy pivot.
            0.0 = no valid pivot or conditions not met
            1.0 = price is ~$0.10 above handle peak on ~45% above-average volume
            2.0 = strong breakout (tighter price, much higher volume)
        price_score: 0.0 to 2.0 — how well price is positioned above the pivot.
        volume_score: 0.0 to 2.0 — volume surge strength.
        pivot_price: The calculated pivot price (handle peak + $0.10), or None.
        current_price: Current close price.
        volume_ratio: Current volume relative to average.
    """

    score: float
    price_score: float
    volume_score: float
    pivot_price: float | None
    current_price: float
    volume_ratio: float | None


def score_pivot_buy(
    df: pd.DataFrame,
    order: int = 10,
    pivot_offset: float = 0.10,
    volume_target_low: float = 1.40,
    volume_target_high: float = 1.50,
    volume_avg_period: int = 50,
    price_tolerance_below: float = 0.50,
    price_tolerance_above: float = 1.00,
) -> PivotBuyScore:
    """Score how strongly price is breaking out of a cup-and-handle pivot point.

    The ideal pivot buy is when price crosses $0.10 above the handle's peak
    on volume at least 40-50% above average. Scores:

        0.0 = no pattern, or price/volume far from pivot conditions
        1.0 = price is right at pivot ($0.10 above handle peak) with
               volume 40-50% above average — textbook entry
        2.0 = exceptional breakout (price tight above pivot, volume 100%+ above avg)

    Args:
        df: OHLCV DataFrame with enough history for cup & handle detection.
        order: Swing detection window.
        pivot_offset: Dollar amount above handle peak for pivot ($0.10 default).
        volume_target_low: Low end of target volume ratio (1.40 = 40% above avg).
        volume_target_high: High end of target volume ratio (1.50 = 50% above avg).
        volume_avg_period: Lookback bars for average volume calculation.
        price_tolerance_below: Max $ below pivot that still gets partial score.
        price_tolerance_above: Max $ above pivot before score decays (chasing).

    Returns:
        PivotBuyScore with combined score and components.
    """
    # First, find the cup and handle pattern
    cup_score = score_cup_and_handle(df, order=order)

    if cup_score.best_pattern is None:
        return PivotBuyScore(0.0, 0.0, 0.0, None, df["Close"].iloc[-1], None)

    pattern = cup_score.best_pattern
    handle_low_idx = pattern["handle_low_idx"]
    right_rim = pattern["right_rim"]

    # Determine handle peak: highest high in the handle region
    right_rim_idx = pattern["right_rim_idx"]
    right_rim_pos = df.index.get_loc(right_rim_idx)

    # Handle region: from right rim onward (up to 50% of cup length)
    cup_len = pattern["cup_len_bars"]
    handle_end_pos = min(right_rim_pos + int(cup_len * 0.5), len(df) - 1)

    if handle_low_idx is not None:
        # Handle exists — peak is the highest High in handle region BEFORE current bar
        handle_region = df.iloc[right_rim_pos:min(handle_end_pos + 1, len(df) - 1)]
    else:
        # No handle yet — use right rim area (excluding current bar)
        handle_region = df.iloc[right_rim_pos:min(handle_end_pos + 1, len(df) - 1)]

    if len(handle_region) == 0:
        return PivotBuyScore(0.0, 0.0, 0.0, None, df["Close"].iloc[-1], None)

    handle_peak = handle_region["High"].max()
    pivot_price = handle_peak + pivot_offset

    current_price = df["Close"].iloc[-1]
    current_volume = df["Volume"].iloc[-1] if "Volume" in df.columns else None

    # --- PRICE SCORE (0-2) ---
    price_diff = current_price - pivot_price

    if price_diff < -price_tolerance_below:
        # Too far below pivot — not triggered
        price_score = 0.0
    elif price_diff < 0:
        # Approaching pivot from below — ramp 0 to 1.0 as we near pivot
        price_score = (price_tolerance_below + price_diff) / price_tolerance_below
    elif price_diff <= pivot_offset * 2:
        # At pivot or just above — sweet spot. Exactly at pivot = 1.0,
        # tight above (within 2x offset) drifts toward 2.0
        price_score = 1.0 + min(price_diff / (pivot_offset * 2), 1.0)
    elif price_diff <= price_tolerance_above:
        # Extended above pivot — starts decaying (chasing)
        overshoot_pct = (price_diff - pivot_offset * 2) / (price_tolerance_above - pivot_offset * 2)
        price_score = 2.0 * (1.0 - overshoot_pct)
    else:
        # Way above pivot — too late
        price_score = 0.0

    price_score = float(np.clip(price_score, 0.0, 2.0))

    # --- VOLUME SCORE (0-2) ---
    volume_score = 0.0
    volume_ratio = None

    if current_volume is not None and "Volume" in df.columns:
        avg_start = max(0, len(df) - 1 - volume_avg_period)
        avg_volume = df["Volume"].iloc[avg_start:-1].mean()

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume

            target_mid = (volume_target_low + volume_target_high) / 2  # 1.45

            if volume_ratio < 1.0:
                # Below average volume — no conviction
                volume_score = 0.0
            elif volume_ratio < volume_target_low:
                # Between average and target — ramp up to 1.0
                volume_score = (volume_ratio - 1.0) / (volume_target_low - 1.0)
            elif volume_ratio <= volume_target_high:
                # In the sweet spot — score is 1.0
                volume_score = 1.0
            else:
                # Above target — exceptional, score climbs toward 2.0
                # Caps at 2.0 around 2x average (100% above)
                excess = volume_ratio - volume_target_high
                max_excess = 2.0 - volume_target_high  # ~0.5
                volume_score = 1.0 + min(excess / max_excess, 1.0)

        volume_score = float(np.clip(volume_score, 0.0, 2.0))

    # --- COMBINED SCORE ---
    # Both price and volume must contribute; weak on either kills the signal
    if price_score == 0.0 or volume_score == 0.0:
        combined = 0.0
    else:
        # Geometric-ish blend: both must be strong
        combined = (price_score * 0.5 + volume_score * 0.5)
        # Penalize if cup/handle quality is poor
        pattern_quality = (cup_score.depth_score + cup_score.shape_score) / 2
        combined *= max(pattern_quality, 0.3)  # floor at 30% to not completely zero out

    combined = float(np.clip(combined, 0.0, 2.0))

    return PivotBuyScore(
        score=combined,
        price_score=price_score,
        volume_score=volume_score,
        pivot_price=pivot_price,
        current_price=current_price,
        volume_ratio=volume_ratio,
    )
