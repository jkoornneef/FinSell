"""Custom pattern detection algorithms."""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def find_swing_highs(df: pd.DataFrame, order: int = 5, col: str = "High") -> pd.Series:
    """Find local maxima (swing highs) in price data.

    Args:
        df: OHLCV DataFrame
        order: Number of bars on each side to compare
        col: Column to analyze

    Returns:
        Boolean Series where True marks a swing high
    """
    highs = argrelextrema(df[col].values, np.greater_equal, order=order)[0]
    result = pd.Series(False, index=df.index)
    result.iloc[highs] = True
    return result


def find_swing_lows(df: pd.DataFrame, order: int = 5, col: str = "Low") -> pd.Series:
    """Find local minima (swing lows) in price data."""
    lows = argrelextrema(df[col].values, np.less_equal, order=order)[0]
    result = pd.Series(False, index=df.index)
    result.iloc[lows] = True
    return result


def detect_double_bottom(df: pd.DataFrame, tolerance: float = 0.02, order: int = 10) -> list[dict]:
    """Detect double bottom patterns.

    Args:
        df: OHLCV DataFrame
        tolerance: Max % difference between the two bottoms
        order: Swing detection window size

    Returns:
        List of dicts with keys: first_low, second_low, neckline, first_idx, second_idx
    """
    swing_lows = find_swing_lows(df, order=order)
    low_indices = df.index[swing_lows]
    low_prices = df.loc[low_indices, "Low"]

    patterns = []
    for i in range(len(low_prices) - 1):
        for j in range(i + 1, min(i + 5, len(low_prices))):
            price_a = low_prices.iloc[i]
            price_b = low_prices.iloc[j]
            pct_diff = abs(price_a - price_b) / price_a

            if pct_diff <= tolerance:
                # Neckline is the highest high between the two lows
                between = df.loc[low_indices[i] : low_indices[j], "High"]
                neckline = between.max()
                patterns.append(
                    {
                        "first_idx": low_indices[i],
                        "second_idx": low_indices[j],
                        "first_low": price_a,
                        "second_low": price_b,
                        "neckline": neckline,
                    }
                )
    return patterns


def detect_cup_and_handle(
    df: pd.DataFrame,
    order: int = 10,
    cup_depth_min: float = 0.05,
    cup_depth_max: float = 0.35,
    handle_depth_max: float = 0.5,
    handle_length_max_pct: float = 0.5,
    rim_tolerance: float = 0.03,
    roundedness_min: float = 0.4,
) -> list[dict]:
    """Detect cup and handle patterns.

    Looks for a U-shaped consolidation (the cup) followed by a smaller
    downward drift (the handle) near the prior highs.

    Args:
        df: OHLCV DataFrame
        order: Swing detection window size
        cup_depth_min: Min retracement from rim to cup bottom (5% default)
        cup_depth_max: Max retracement from rim to cup bottom (35% default)
        handle_depth_max: Max handle depth as fraction of cup depth (50%)
        handle_length_max_pct: Max handle length as fraction of cup length
        rim_tolerance: Max % difference between left and right rim heights
        roundedness_min: Min fraction of cup bars within the bottom third of the
            cup's price range. Higher = more U-shaped (0.4 default). A V-shape
            concentrates bars near the rims, so this value will be low (~0.1-0.2).

    Returns:
        List of dicts with: left_rim_idx, bottom_idx, right_rim_idx,
        handle_low_idx, left_rim, right_rim, cup_bottom, handle_low, cup_depth_pct
    """
    swing_highs = find_swing_highs(df, order=order)
    swing_lows = find_swing_lows(df, order=order)

    high_indices = df.index[swing_highs]
    high_prices = df.loc[high_indices, "High"]
    low_indices = df.index[swing_lows]
    low_prices = df.loc[low_indices, "Low"]

    patterns = []

    for i in range(len(high_prices) - 1):
        left_rim_idx = high_indices[i]
        left_rim = high_prices.iloc[i]

        # Find candidate right rims (subsequent swing highs near the same level)
        for j in range(i + 1, len(high_prices)):
            right_rim_idx = high_indices[j]
            right_rim = high_prices.iloc[j]

            # Rims should be roughly equal
            rim_diff = abs(right_rim - left_rim) / left_rim
            if rim_diff > rim_tolerance:
                continue

            # Find the lowest swing low between the two rims (cup bottom)
            cup_lows = low_prices[
                (low_indices > left_rim_idx) & (low_indices < right_rim_idx)
            ]
            if cup_lows.empty:
                continue

            cup_bottom = cup_lows.min()
            bottom_idx = cup_lows.idxmin()
            rim_avg = (left_rim + right_rim) / 2
            cup_depth_pct = (rim_avg - cup_bottom) / rim_avg

            # Validate cup depth
            if not (cup_depth_min <= cup_depth_pct <= cup_depth_max):
                continue

            # Check cup is roughly U-shaped (bottom is in middle third)
            cup_start_pos = df.index.get_loc(left_rim_idx)
            cup_end_pos = df.index.get_loc(right_rim_idx)
            bottom_pos = df.index.get_loc(bottom_idx)
            cup_len = cup_end_pos - cup_start_pos
            relative_bottom = (bottom_pos - cup_start_pos) / cup_len
            if not (0.25 <= relative_bottom <= 0.75):
                continue

            # Roundedness check: reject V-shapes.
            # Measure what fraction of bars in the cup sit in the lower third
            # of the price range. A U-shape has many bars near the bottom
            # (flat base), while a V-shape spends most time near the rims.
            cup_closes = df.iloc[cup_start_pos:cup_end_pos + 1]["Close"].values
            price_range = rim_avg - cup_bottom
            lower_third_threshold = cup_bottom + price_range / 3
            bars_in_lower_third = np.sum(cup_closes <= lower_third_threshold)
            roundedness = bars_in_lower_third / cup_len
            if roundedness < roundedness_min:
                continue

            # Look for the handle: a smaller dip after the right rim
            handle_search_end = min(
                cup_end_pos + int(cup_len * handle_length_max_pct), len(df) - 1
            )
            handle_region_lows = low_prices[
                (low_indices > right_rim_idx)
                & (low_indices <= df.index[handle_search_end])
            ]

            if handle_region_lows.empty:
                # No handle yet — could be forming; still record cup
                patterns.append(
                    {
                        "left_rim_idx": left_rim_idx,
                        "bottom_idx": bottom_idx,
                        "right_rim_idx": right_rim_idx,
                        "handle_low_idx": None,
                        "left_rim": left_rim,
                        "right_rim": right_rim,
                        "cup_bottom": cup_bottom,
                        "handle_low": None,
                        "cup_depth_pct": cup_depth_pct,
                        "roundedness": roundedness,
                    }
                )
                continue

            handle_low = handle_region_lows.min()
            handle_low_idx = handle_region_lows.idxmin()
            handle_depth = right_rim - handle_low
            max_handle_depth = cup_depth_pct * rim_avg * handle_depth_max

            if handle_depth > max_handle_depth:
                continue

            patterns.append(
                {
                    "left_rim_idx": left_rim_idx,
                    "bottom_idx": bottom_idx,
                    "right_rim_idx": right_rim_idx,
                    "handle_low_idx": handle_low_idx,
                    "left_rim": left_rim,
                    "right_rim": right_rim,
                    "cup_bottom": cup_bottom,
                    "handle_low": handle_low,
                    "cup_depth_pct": cup_depth_pct,
                    "roundedness": roundedness,
                }
            )

    return patterns


def detect_higher_highs_higher_lows(df: pd.DataFrame, order: int = 5, min_count: int = 3) -> bool:
    """Check if recent price action shows an uptrend (higher highs + higher lows)."""
    swing_highs = df.loc[find_swing_highs(df, order=order), "High"]
    swing_lows = df.loc[find_swing_lows(df, order=order), "Low"]

    if len(swing_highs) < min_count or len(swing_lows) < min_count:
        return False

    recent_highs = swing_highs.iloc[-min_count:]
    recent_lows = swing_lows.iloc[-min_count:]

    hh = all(recent_highs.iloc[i] > recent_highs.iloc[i - 1] for i in range(1, len(recent_highs)))
    hl = all(recent_lows.iloc[i] > recent_lows.iloc[i - 1] for i in range(1, len(recent_lows)))
    return hh and hl


def detect_dead_cat_bounce(
    df: pd.DataFrame,
    drop_pct_min: float = 0.15,
    bounce_retrace_min: float = 0.10,
    bounce_retrace_max: float = 0.50,
    continuation_drop_pct: float = 0.05,
    lookback: int = 60,
    volume_spike_factor: float = 1.5,
    bounce_volume_fade: float = 0.7,
) -> list[dict]:
    """Detect dead cat bounce patterns.

    A dead cat bounce is a brief recovery rally in a strong downtrend that
    fails to hold and resumes the decline. The sequence is:
    1. Sharp initial drop (>= drop_pct_min)
    2. A bounce that retraces part of the drop (but not too much)
    3. Price rolls over and breaks below the bounce low, continuing down

    Args:
        df: OHLCV DataFrame
        drop_pct_min: Min initial decline to qualify (15% default)
        bounce_retrace_min: Min bounce as fraction of the drop (10%)
        bounce_retrace_max: Max bounce as fraction of the drop (50%)
            — a larger retrace suggests genuine recovery, not a dead cat
        continuation_drop_pct: How far below the initial low price must
            fall to confirm the pattern (5% default)
        lookback: Max bars to scan for the full pattern sequence
        volume_spike_factor: If > 0, require initial drop to have elevated
            volume (avg volume * this factor). Set to 0 to disable.
        bounce_volume_fade: Max ratio of bounce avg volume to drop avg volume.
            A dead cat bounce typically has fading volume (weak conviction).
            Default 0.7 means bounce volume must be ≤70% of drop volume.
            Set to 0 to disable this check.

    Returns:
        List of dicts with: pre_drop_high_idx, drop_low_idx, bounce_high_idx,
        continuation_low_idx, drop_pct, bounce_retrace_pct, confirmed
    """
    patterns = []
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None

    i = 0
    while i < len(df) - lookback:
        # 1. Find a significant drop: highest high followed by a lower low
        window_end = min(i + lookback, len(df))
        segment_high_idx = i + np.argmax(highs[i:window_end])
        segment_high = highs[segment_high_idx]

        # Look for the drop low after the high
        if segment_high_idx >= window_end - 5:
            i += 1
            continue

        post_high_lows = lows[segment_high_idx + 1 : window_end]
        if len(post_high_lows) == 0:
            i += 1
            continue

        drop_low_offset = np.argmin(post_high_lows)
        drop_low_idx = segment_high_idx + 1 + drop_low_offset
        drop_low = lows[drop_low_idx]

        drop_pct = (segment_high - drop_low) / segment_high
        if drop_pct < drop_pct_min:
            i += 1
            continue

        # Optional: check for volume spike during the drop
        if volume_spike_factor > 0 and volumes is not None:
            avg_vol = np.mean(volumes[max(0, i - 20) : i]) if i > 20 else np.mean(volumes[:i + 1])
            drop_vol = np.mean(volumes[segment_high_idx : drop_low_idx + 1])
            if avg_vol > 0 and drop_vol < avg_vol * volume_spike_factor:
                i += 1
                continue

        # 2. Find the bounce high after the drop low
        bounce_search_end = min(drop_low_idx + lookback // 2, len(df))
        post_drop_highs = highs[drop_low_idx + 1 : bounce_search_end]
        if len(post_drop_highs) == 0:
            i += 1
            continue

        bounce_high_offset = np.argmax(post_drop_highs)
        bounce_high_idx = drop_low_idx + 1 + bounce_high_offset
        bounce_high = highs[bounce_high_idx]

        drop_size = segment_high - drop_low
        bounce_size = bounce_high - drop_low
        bounce_retrace_pct = bounce_size / drop_size if drop_size > 0 else 0

        if not (bounce_retrace_min <= bounce_retrace_pct <= bounce_retrace_max):
            i += 1
            continue

        # Volume fade check: bounce should have lower volume than the drop
        bounce_vol_ratio = None
        if bounce_volume_fade > 0 and volumes is not None:
            drop_vol = np.mean(volumes[segment_high_idx : drop_low_idx + 1])
            bounce_vol = np.mean(volumes[drop_low_idx + 1 : bounce_high_idx + 1])
            if drop_vol > 0:
                bounce_vol_ratio = bounce_vol / drop_vol
                if bounce_vol_ratio > bounce_volume_fade:
                    i += 1
                    continue

        # 3. Look for continuation: price breaks below the initial drop low
        continuation_search_end = min(bounce_high_idx + lookback // 2, len(df))
        post_bounce_lows = lows[bounce_high_idx + 1 : continuation_search_end]

        confirmed = False
        continuation_low_idx = None
        if len(post_bounce_lows) > 0:
            cont_low_offset = np.argmin(post_bounce_lows)
            cont_low = post_bounce_lows[cont_low_offset]
            threshold = drop_low * (1 - continuation_drop_pct)
            if cont_low <= threshold:
                confirmed = True
                continuation_low_idx = bounce_high_idx + 1 + cont_low_offset

        patterns.append(
            {
                "pre_drop_high_idx": df.index[segment_high_idx],
                "drop_low_idx": df.index[drop_low_idx],
                "bounce_high_idx": df.index[bounce_high_idx],
                "continuation_low_idx": df.index[continuation_low_idx] if continuation_low_idx else None,
                "pre_drop_high": segment_high,
                "drop_low": drop_low,
                "bounce_high": bounce_high,
                "drop_pct": drop_pct,
                "bounce_retrace_pct": bounce_retrace_pct,
                "bounce_vol_ratio": bounce_vol_ratio,
                "confirmed": confirmed,
            }
        )

        # Skip past this pattern to avoid overlapping detections
        i = bounce_high_idx + 1
        continue

    return patterns
