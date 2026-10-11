"""Purchase timing score.

Combines multiple pyramiding/add-on signals into a single -1 to +1
timing indicator for when to add to an existing position.

Inputs: OHLCV data + weighted average cost basis.

Sub-signals:
  1. Pullback to 10-week (50-day) MA — bounce on volume
  2. Three-Weeks-Tight pattern — tight weekly closes (accumulation)
  3. Flat Base — sideways consolidation (5+ weeks, <15% correction)
  4. Ascending Base — three pullbacks with rising lows
  5. Auto add-on — stock 2-2.5% above entry (momentum confirmation)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class ThreeWeeksTightScore:
    """Three-Weeks-Tight pattern score.

    Attributes:
        score: -1.0 to +1.0 — proximity to pattern completion/breakout.
            -1.0 = no tight action
            -0.5 = one tight week so far
             0.0 = three tight weeks complete, at breakout
            +0.5 = breaking above the tight range
            +1.0 = confirmed breakout with volume
        quality: 0.0 to 1.0 — how tight the weekly closes are.
        details: Dict with measurements.
    """

    score: float
    quality: float
    details: dict | None


def _get_weekly_closes(df: pd.DataFrame) -> pd.Series:
    """Convert daily OHLCV to weekly closes."""
    if hasattr(df.index, 'freq') or hasattr(df.index, 'to_period'):
        weekly = df["Close"].resample("W-FRI").last().dropna()
    else:
        # Fallback: group every 5 bars
        close = df["Close"].values
        weekly_closes = [close[min(i + 4, len(close) - 1)] for i in range(0, len(close), 5)]
        weekly = pd.Series(weekly_closes)
    return weekly


def _get_weekly_volumes(df: pd.DataFrame) -> pd.Series:
    """Convert daily volumes to weekly totals."""
    if "Volume" not in df.columns:
        return pd.Series(dtype=float)
    if hasattr(df.index, 'freq') or hasattr(df.index, 'to_period'):
        weekly = df["Volume"].resample("W-FRI").sum().dropna()
    else:
        volume = df["Volume"].values
        weekly_vols = [np.sum(volume[i:i+5]) for i in range(0, len(volume), 5)]
        weekly = pd.Series(weekly_vols)
    return weekly


def score_three_weeks_tight(
    df: pd.DataFrame,
    tight_threshold: float = 0.015,
    volume_avg_period: int = 50,
    volume_surge_threshold: float = 1.4,
) -> ThreeWeeksTightScore:
    """Score the Three-Weeks-Tight pattern.

    Three consecutive weekly closes within ~1-1.5% of each other indicates
    quiet institutional accumulation. The buy trigger is when price breaks
    above the top of the tight range on volume.

    Args:
        df: OHLCV DataFrame. Daily data, needs 30+ bars.
        tight_threshold: Max % variation between weekly closes (1.5%).
        volume_avg_period: Bars for avg volume (for breakout confirmation).
        volume_surge_threshold: Volume ratio for confirmation.

    Returns:
        ThreeWeeksTightScore with score, quality, and details.
    """
    if len(df) < 25:
        return ThreeWeeksTightScore(-1.0, 0.0, None)

    weekly_closes = _get_weekly_closes(df)
    weekly_volumes = _get_weekly_volumes(df)

    if len(weekly_closes) < 4:
        return ThreeWeeksTightScore(-1.0, 0.0, None)

    close = df["Close"].values
    volume = df["Volume"].values if "Volume" in df.columns else None
    current_price = close[-1]

    # Check the most recent weeks for tight pattern
    # Look at last 6 weeks to find the best 3-week tight sequence
    best_tight_score = None
    search_depth = min(6, len(weekly_closes) - 1)

    for start in range(len(weekly_closes) - 3, max(len(weekly_closes) - 3 - search_depth, -1), -1):
        if start < 0:
            break

        w1 = weekly_closes.iloc[start]
        w2 = weekly_closes.iloc[start + 1]
        w3 = weekly_closes.iloc[start + 2]

        if w1 <= 0:
            continue

        # Calculate tightness: max deviation from w1
        dev2 = abs(w2 - w1) / w1
        dev3 = abs(w3 - w1) / w1
        dev23 = abs(w3 - w2) / w2 if w2 > 0 else 1.0

        max_deviation = max(dev2, dev3, dev23)

        if max_deviation > tight_threshold * 2:
            continue  # Not tight enough even for partial score

        # Tightness quality: 0→1 based on how tight
        if max_deviation <= tight_threshold * 0.5:
            tightness = 1.0
        elif max_deviation <= tight_threshold:
            tightness = 0.7 + 0.3 * (1.0 - max_deviation / tight_threshold)
        else:
            tightness = 0.3 * (1.0 - (max_deviation - tight_threshold) / tight_threshold)

        tightness = float(np.clip(tightness, 0.0, 1.0))

        # Volume during tight weeks should be LOW (quiet accumulation)
        vol_quality = 0.5
        if len(weekly_volumes) > start + 2 and len(weekly_volumes) > 3:
            tight_vol = np.mean(weekly_volumes.iloc[start:start + 3])
            prior_vol = np.mean(weekly_volumes.iloc[max(0, start - 4):start])
            if prior_vol > 0:
                vol_ratio_during = tight_vol / prior_vol
                # Low volume during tight = good (below 0.8x = ideal)
                if vol_ratio_during <= 0.7:
                    vol_quality = 1.0
                elif vol_ratio_during <= 1.0:
                    vol_quality = 0.5 + 0.5 * (1.0 - vol_ratio_during) / 0.3
                else:
                    vol_quality = 0.3

        # Top of tight range (resistance/breakout level)
        tight_high = max(w1, w2, w3)

        # Where is current price relative to the tight range?
        weeks_since = len(weekly_closes) - 1 - (start + 2)

        if weeks_since == 0 and max_deviation <= tight_threshold:
            # Currently IN the third tight week
            proximity = -0.1  # Pattern complete, waiting for breakout
        elif current_price > tight_high * 1.005:
            # Above the tight range — breakout
            breakout_pct = (current_price - tight_high) / tight_high

            # Volume confirmation on breakout
            breakout_vol_confirmed = False
            if volume is not None and len(volume) > volume_avg_period:
                avg_vol = np.mean(volume[-(volume_avg_period + 1):-1])
                if avg_vol > 0 and volume[-1] / avg_vol >= volume_surge_threshold:
                    breakout_vol_confirmed = True

            if breakout_vol_confirmed:
                proximity = 0.5 + min(breakout_pct / 0.03, 0.5)
            else:
                proximity = 0.0 + min(breakout_pct / 0.03, 0.5)
        elif weeks_since <= 1:
            # Just completing the pattern
            proximity = -0.2
        else:
            # Pattern formed a while ago, may be stale
            proximity = -0.5

        quality = 0.50 * tightness + 0.30 * vol_quality + 0.20 * (1.0 - min(weeks_since / 3.0, 1.0))

        combined = quality + (1.0 - abs(proximity))
        if best_tight_score is None or combined > best_tight_score[0]:
            best_tight_score = (combined, proximity, quality, {
                "week1_close": float(w1),
                "week2_close": float(w2),
                "week3_close": float(w3),
                "max_deviation_pct": float(max_deviation),
                "tight_high": float(tight_high),
                "current_price": float(current_price),
                "weeks_since_pattern": int(weeks_since),
            })

    if best_tight_score is None:
        return ThreeWeeksTightScore(-1.0, 0.0, None)

    _, proximity, quality, details = best_tight_score
    return ThreeWeeksTightScore(
        score=float(np.clip(proximity, -1.0, 1.0)),
        quality=float(np.clip(quality, 0.0, 1.0)),
        details=details,
    )


@dataclass
class AscendingBaseScore:
    """Ascending base pattern score.

    Attributes:
        score: -1.0 to +1.0 — proximity to breakout from ascending base.
            -1.0 = first pullback only, pattern forming
            -0.5 = second pullback complete
             0.0 = third pullback complete, at breakout level
            +0.5 = breaking above the ascending resistance
            +1.0 = confirmed breakout
        quality: 0.0 to 1.0 — how well the pullbacks fit ascending base criteria.
        details: Dict with the three pullback measurements.
    """

    score: float
    quality: float
    details: dict | None


def score_ascending_base(
    df: pd.DataFrame,
    order: int = 8,
    pullback_min_pct: float = 0.08,
    pullback_max_pct: float = 0.22,
    min_bars_between: int = 10,
) -> AscendingBaseScore:
    """Score an ascending base pattern.

    An ascending base consists of three pullbacks (10-20% each) where
    each successive low is higher than the previous one. It forms midway
    through a move, indicating orderly profit-taking without breaking the trend.

    Args:
        df: OHLCV DataFrame. Daily data, needs 80+ bars.
        order: Swing detection sensitivity.
        pullback_min_pct: Minimum pullback depth (8%).
        pullback_max_pct: Maximum pullback depth (22%).
        min_bars_between: Minimum bars between successive lows.

    Returns:
        AscendingBaseScore with proximity, quality, and details.
    """
    from finsell.patterns import find_swing_lows, find_swing_highs

    if len(df) < 60:
        return AscendingBaseScore(-1.0, 0.0, None)

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    current_price = close[-1]

    swing_lows = find_swing_lows(df, order=order)
    swing_highs = find_swing_highs(df, order=order)

    low_indices = df.index[swing_lows]
    low_prices = df.loc[low_indices, "Low"]
    high_indices = df.index[swing_highs]
    high_prices = df.loc[high_indices, "High"]

    if len(low_prices) < 3 or len(high_prices) < 3:
        return AscendingBaseScore(-1.0, 0.0, None)

    best_result = None
    best_combined = -999.0

    # Find three ascending lows with pullbacks from local highs
    for i in range(len(low_prices) - 2):
        low1_idx = low_indices[i]
        low1_price = low_prices.iloc[i]
        low1_pos = df.index.get_loc(low1_idx)

        for j in range(i + 1, len(low_prices) - 1):
            low2_idx = low_indices[j]
            low2_price = low_prices.iloc[j]
            low2_pos = df.index.get_loc(low2_idx)

            if low2_pos - low1_pos < min_bars_between:
                continue
            if low2_price <= low1_price:
                continue  # Must be ascending

            for k in range(j + 1, len(low_prices)):
                low3_idx = low_indices[k]
                low3_price = low_prices.iloc[k]
                low3_pos = df.index.get_loc(low3_idx)

                if low3_pos - low2_pos < min_bars_between:
                    continue
                if low3_price <= low2_price:
                    continue  # Must be ascending

                # Find the high before each low (the peak it pulled back from)
                high_before_1 = np.max(high[max(0, low1_pos - 30):low1_pos + 1])
                high_before_2 = np.max(high[low1_pos:low2_pos + 1])
                high_before_3 = np.max(high[low2_pos:low3_pos + 1])

                # Calculate pullback depths
                pb1 = (high_before_1 - low1_price) / high_before_1 if high_before_1 > 0 else 0
                pb2 = (high_before_2 - low2_price) / high_before_2 if high_before_2 > 0 else 0
                pb3 = (high_before_3 - low3_price) / high_before_3 if high_before_3 > 0 else 0

                # At least 2 of 3 pullbacks should be in range
                valid_pbs = sum(1 for pb in [pb1, pb2, pb3]
                                if pullback_min_pct <= pb <= pullback_max_pct)
                if valid_pbs < 2:
                    continue

                # Resistance line: connect the highs before each pullback
                resistance = max(high_before_1, high_before_2, high_before_3)

                # Quality: ascending nature, pullback consistency, depth
                # Ascending quality: how evenly spaced are the lows?
                rise1 = (low2_price - low1_price) / low1_price
                rise2 = (low3_price - low2_price) / low2_price
                ascending_quality = 1.0 - min(abs(rise1 - rise2) / max(rise1, rise2, 0.01), 1.0)

                # Pullback depth quality (10-20% ideal)
                ideal_mid = 0.15
                pb_quality = 1.0 - np.mean([
                    min(abs(pb - ideal_mid) / 0.10, 1.0) for pb in [pb1, pb2, pb3]
                ])
                pb_quality = float(np.clip(pb_quality, 0.0, 1.0))

                # Proximity: where is current price vs resistance?
                price_vs_resistance = (current_price - resistance) / resistance

                if current_price < low3_price:
                    proximity = -0.8  # Below the pattern
                elif price_vs_resistance < -0.05:
                    # Well below resistance
                    progress = (current_price - low3_price) / (resistance - low3_price) if resistance > low3_price else 0
                    proximity = -0.5 + 0.5 * progress
                elif price_vs_resistance < 0:
                    # Approaching resistance
                    proximity = -0.1 + 0.1 * (1.0 + price_vs_resistance / 0.05)
                elif price_vs_resistance < 0.03:
                    # At or just above resistance — breakout zone
                    proximity = 0.0 + 0.5 * (price_vs_resistance / 0.03)
                else:
                    # Above resistance
                    proximity = min(0.5 + 0.5 * (price_vs_resistance / 0.05), 1.0)

                quality = 0.40 * ascending_quality + 0.35 * pb_quality + 0.25 * (valid_pbs / 3.0)

                combined = quality + (1.0 - abs(proximity)) * 0.3
                if combined > best_combined:
                    best_combined = combined
                    best_result = AscendingBaseScore(
                        score=float(np.clip(proximity, -1.0, 1.0)),
                        quality=float(np.clip(quality, 0.0, 1.0)),
                        details={
                            "low1": float(low1_price),
                            "low2": float(low2_price),
                            "low3": float(low3_price),
                            "pullback1_pct": float(pb1),
                            "pullback2_pct": float(pb2),
                            "pullback3_pct": float(pb3),
                            "resistance": float(resistance),
                            "current_price": float(current_price),
                        },
                    )

    if best_result is None:
        return AscendingBaseScore(-1.0, 0.0, None)

    return best_result


@dataclass
class PurchaseTimingScore:
    """Combined purchase timing score for adding to positions.

    Attributes:
        score: -1.0 to +1.0 — overall timing signal.
            -1.0 = terrible time to add (no patterns, extended, etc.)
            -0.5 = neutral, no clear signal
             0.0 = conditions forming, watch closely
            +0.5 = good add-on opportunity
            +1.0 = strong add signal (pattern + momentum + volume)
        best_signal: Name of the strongest contributing signal.
        sub_scores: Dict of all sub-signal scores.
        details: Dict with combined analysis.
    """

    score: float
    best_signal: str
    sub_scores: dict
    details: dict | None


def score_purchase_timing(
    df: pd.DataFrame,
    cost_basis: float,
    volume_avg_period: int = 50,
) -> PurchaseTimingScore:
    """Score overall purchase timing for adding to an existing position.

    Combines five sub-signals into a single -1 to +1 timing score:
      1. Pullback to 10-week MA (bounce on volume)
      2. Three-Weeks-Tight (quiet accumulation)
      3. Flat Base (sideways consolidation)
      4. Ascending Base (three rising pullbacks)
      5. Auto add-on (stock 2-2.5% above cost basis)

    The final score uses the BEST sub-signal (not average) since any one
    valid pattern is sufficient to justify adding. Quality of the best
    pattern weights the final score.

    Args:
        df: OHLCV DataFrame. Daily data, needs 60+ bars for all patterns.
        cost_basis: Weighted average cost per share of existing position.
        volume_avg_period: Bars for average volume calculations.

    Returns:
        PurchaseTimingScore with combined score, best signal name, and all sub-scores.
    """
    from finsell.pyramiding import score_flat_base, score_ma_support

    close = df["Close"].values
    current_price = close[-1]

    sub_scores = {}
    signal_details = {}

    # --- 1. Pullback to 10-week (50-day) MA ---
    ma_result = score_ma_support(df)
    sub_scores["ma_support"] = ma_result.score
    signal_details["ma_support"] = {"quality": ma_result.quality}

    # --- 2. Three-Weeks-Tight ---
    twt_result = score_three_weeks_tight(df)
    sub_scores["three_weeks_tight"] = twt_result.score
    signal_details["three_weeks_tight"] = {"quality": twt_result.quality}

    # --- 3. Flat Base ---
    fb_result = score_flat_base(df)
    sub_scores["flat_base"] = fb_result.score
    signal_details["flat_base"] = {"quality": fb_result.quality}

    # --- 4. Ascending Base ---
    ab_result = score_ascending_base(df)
    sub_scores["ascending_base"] = ab_result.score
    signal_details["ascending_base"] = {"quality": ab_result.quality}

    # --- 5. Auto Add-On (2-2.5% above cost basis) ---
    if cost_basis > 0:
        gain_from_entry = (current_price - cost_basis) / cost_basis

        if gain_from_entry < 0:
            # Below cost basis — not a good add time
            addon_score = -1.0
        elif gain_from_entry < 0.02:
            # Between 0-2% above entry — approaching add zone
            addon_score = -0.5 + 0.5 * (gain_from_entry / 0.02)
        elif gain_from_entry <= 0.025:
            # 2-2.5% above entry — ideal automatic add zone
            addon_score = 0.5 + 0.5 * ((gain_from_entry - 0.02) / 0.005)
        elif gain_from_entry <= 0.05:
            # 2.5-5% above — still ok but less ideal
            addon_score = 0.5 * (1.0 - (gain_from_entry - 0.025) / 0.025)
        else:
            # 5%+ above entry — extended, don't chase
            addon_score = -0.3
    else:
        addon_score = 0.0

    sub_scores["auto_addon"] = addon_score
    signal_details["auto_addon"] = {
        "cost_basis": cost_basis,
        "current_price": float(current_price),
        "gain_pct": float((current_price - cost_basis) / cost_basis) if cost_basis > 0 else 0.0,
    }

    # --- Combine: use the BEST signal (any one pattern is sufficient) ---
    # Weight each signal by its quality where available
    qualities = {
        "ma_support": ma_result.quality,
        "three_weeks_tight": twt_result.quality,
        "flat_base": fb_result.quality,
        "ascending_base": ab_result.quality,
        "auto_addon": 0.7 if 0.02 <= (current_price - cost_basis) / cost_basis <= 0.025 else 0.3,
    }

    # Effective score = sub_score weighted by quality
    effective_scores = {
        name: sub_scores[name] * max(qualities[name], 0.3)
        for name in sub_scores
    }

    # Best signal drives the output
    best_signal = max(effective_scores, key=effective_scores.get)
    best_effective = effective_scores[best_signal]

    # The auto-addon is additive — if price is in the 2-2.5% sweet spot
    # AND a pattern is active, boost the score
    pattern_signals = {k: v for k, v in effective_scores.items() if k != "auto_addon"}
    best_pattern = max(pattern_signals, key=pattern_signals.get)
    best_pattern_score = pattern_signals[best_pattern]

    if addon_score > 0.3 and best_pattern_score > 0:
        # Both momentum AND pattern confirm — strong add signal
        combined = 0.6 * best_pattern_score + 0.4 * addon_score
        best_signal = best_pattern + " + auto_addon"
    elif best_effective > 0:
        combined = best_effective
    else:
        # No positive signal — use the least negative
        combined = best_effective

    final_score = float(np.clip(combined, -1.0, 1.0))

    return PurchaseTimingScore(
        score=final_score,
        best_signal=best_signal,
        sub_scores=sub_scores,
        details={
            "current_price": float(current_price),
            "cost_basis": float(cost_basis),
            "gain_from_entry_pct": float((current_price - cost_basis) / cost_basis) if cost_basis > 0 else 0.0,
            "signal_qualities": qualities,
            "effective_scores": effective_scores,
        },
    )
