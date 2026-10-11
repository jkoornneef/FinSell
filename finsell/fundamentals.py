"""Fundamental growth scoring."""

import numpy as np


def earnings_growth_score(
    quarterly_earnings: list[float],
    annual_earnings: list[float],
) -> float:
    """Score a stock's earnings growth on a 0.0 to 2.0 scale.

    Evaluates quarterly YoY earnings growth and multi-year annual growth
    consistency. The scoring bands are:

        0.0  — Below 15% quarterly growth OR less than 1.5 years of annual data
        0.0–1.0 — Between the floor (15%) and the target (25% quarterly + 3yr annual)
        1.0  — Meets 25% quarterly growth with a solid 3-year annual record
        1.0–2.0 — Exceeds the target; stronger growth drifts toward 2.0

    Args:
        quarterly_earnings: Quarterly EPS values, oldest first.
            Needs at least 5 quarters (4 YoY comparisons).
        annual_earnings: Annual EPS values, oldest first.
            Ideally 3+ years for full scoring.

    Returns:
        Float score between 0.0 and 2.0.
    """
    # --- Quarterly growth ---
    if len(quarterly_earnings) < 5:
        return 0.0

    # Calculate YoY quarterly growth rates (each quarter vs same quarter prior year)
    # With 5 quarters we get 1 YoY comparison; with 8 we get 4 (full year of YoY)
    yoy_growths = []
    for i in range(4, len(quarterly_earnings)):
        prior = quarterly_earnings[i - 4]
        current = quarterly_earnings[i]
        if prior > 0:
            yoy_growths.append((current - prior) / prior)

    if not yoy_growths:
        return 0.0

    avg_quarterly_growth = np.mean(yoy_growths)

    # --- Annual growth ---
    years_of_data = len(annual_earnings)

    if years_of_data < 2:
        return 0.0

    # Calculate annual growth rates
    annual_growths = []
    for i in range(1, len(annual_earnings)):
        prior = annual_earnings[i - 1]
        current = annual_earnings[i]
        if prior > 0:
            annual_growths.append((current - prior) / prior)

    if not annual_growths:
        return 0.0

    avg_annual_growth = np.mean(annual_growths)
    # Consistency: penalize volatile annual growth
    annual_consistency = 1.0 - min(np.std(annual_growths) / (abs(avg_annual_growth) + 1e-9), 1.0)

    # --- Floor check: hard zero below minimums ---
    if avg_quarterly_growth < 0.15 or years_of_data < 2:
        return 0.0

    # Years factor: scales 0→1 from 1.5yr to 3yr of annual data
    years_factor = np.clip((years_of_data - 1.5) / 1.5, 0.0, 1.0)

    # If below the floor thresholds after interpolation, return 0
    if years_factor == 0.0:
        return 0.0

    # --- Score calculation ---
    # Quarterly component: 15% → 0.0, 25% → 1.0, 50%+ → approaching 2.0
    if avg_quarterly_growth <= 0.15:
        q_score = 0.0
    elif avg_quarterly_growth <= 0.25:
        # Linear ramp from 0 to 1.0 between 15% and 25%
        q_score = (avg_quarterly_growth - 0.15) / 0.10
    else:
        # Above 25%: ramp toward 2.0 with diminishing returns
        excess = avg_quarterly_growth - 0.25
        q_score = 1.0 + min(excess / 0.25, 1.0)  # caps at 2.0 around 50%

    # Annual component: rewards strong multi-year track record
    if avg_annual_growth <= 0.15:
        a_score = 0.0
    elif avg_annual_growth <= 0.25:
        a_score = (avg_annual_growth - 0.15) / 0.10
    else:
        excess = avg_annual_growth - 0.25
        a_score = 1.0 + min(excess / 0.25, 1.0)

    # Blend: 60% quarterly, 30% annual, 10% consistency
    raw_score = 0.60 * q_score + 0.30 * a_score + 0.10 * (annual_consistency * 2.0)

    # Apply years factor (not enough history drags score toward 0)
    score = raw_score * years_factor

    return float(np.clip(score, 0.0, 2.0))
