"""Market top and distribution day detection."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DistributionSignal:
    """Distribution day analysis result.

    Attributes:
        distribution_days: DataFrame marking each distribution day with details.
        cluster_count: Number of distribution days in the lookback window.
        signal_strength: 0.0 to 2.0 — likelihood of a market top.
            0.0 = no distribution pressure
            1.0 = elevated distribution (4-5 days in 25 sessions — caution)
            2.0 = heavy distribution (6+ days — potential market top / bubble burst)
        is_stalling: Whether the most recent bar is a stalling day.
        latest_cluster_start: Index of when the current cluster began, or None.
    """

    distribution_days: pd.DataFrame
    cluster_count: int
    signal_strength: float
    is_stalling: bool
    latest_cluster_start: pd.Timestamp | None


def detect_distribution_days(
    df: pd.DataFrame,
    lookback: int = 25,
    min_decline_pct: float = 0.002,
    stall_range_pct: float = 0.004,
    volume_increase_pct: float = 0.0,
    cluster_warning: int = 4,
    cluster_danger: int = 6,
) -> DistributionSignal:
    """Detect distribution days and score market top risk.

    A distribution day occurs when a stock/index closes down (or stalls)
    on volume higher than the previous session. Multiple distribution days
    clustered in a short period signals institutional selling pressure —
    a potential market top or bursting bubble.

    Args:
        df: OHLCV DataFrame (needs at least lookback+1 bars).
        lookback: Rolling window to count distribution days (25 trading days default).
        min_decline_pct: Minimum close-to-close decline to qualify (0.2% default).
        stall_range_pct: Max gain that still counts as "stalling" if on higher volume.
            A stall day is when price barely moves up despite heavy volume —
            institutions distributing into buying pressure.
        volume_increase_pct: Min volume increase over prior day (0% = any increase).
        cluster_warning: Number of dist. days in lookback that triggers caution (4).
        cluster_danger: Number of dist. days in lookback that signals danger (6).

    Returns:
        DistributionSignal with day-by-day breakdown and overall signal.
    """
    if len(df) < 2:
        return DistributionSignal(
            distribution_days=pd.DataFrame(),
            cluster_count=0,
            signal_strength=0.0,
            is_stalling=False,
            latest_cluster_start=None,
        )

    closes = df["Close"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.ones(len(df))

    # Calculate daily returns and volume changes
    returns = np.diff(closes) / closes[:-1]
    vol_changes = np.diff(volumes) / np.where(volumes[:-1] > 0, volumes[:-1], 1)

    # Build distribution day flags (starting from index 1 since we need prior bar)
    n = len(returns)
    is_distribution = np.zeros(n, dtype=bool)
    is_stall = np.zeros(n, dtype=bool)
    day_type = [""] * n

    for i in range(n):
        higher_volume = vol_changes[i] > volume_increase_pct

        if not higher_volume:
            continue

        if returns[i] <= -min_decline_pct:
            # Classic distribution: closes down on higher volume
            is_distribution[i] = True
            day_type[i] = "decline"
        elif -min_decline_pct < returns[i] <= stall_range_pct:
            # Stalling: barely moves (or tiny gain) on higher volume
            # Institutions selling into buying pressure
            is_distribution[i] = True
            is_stall[i] = True
            day_type[i] = "stall"

    # Build result DataFrame (aligned to df index, skipping first bar)
    dist_df = pd.DataFrame(
        {
            "is_distribution": is_distribution,
            "is_stall": is_stall,
            "type": day_type,
            "return_pct": returns * 100,
            "volume_change_pct": vol_changes * 100,
        },
        index=df.index[1:],
    )

    # Count distribution days in the lookback window
    recent = dist_df.iloc[-lookback:] if len(dist_df) >= lookback else dist_df
    cluster_count = int(recent["is_distribution"].sum())

    # Find when the cluster started
    latest_cluster_start = None
    if cluster_count > 0:
        dist_in_window = recent[recent["is_distribution"]]
        latest_cluster_start = dist_in_window.index[0]

    # Is the most recent bar a stalling day?
    current_is_stalling = bool(is_stall[-1]) if len(is_stall) > 0 else False

    # --- SIGNAL STRENGTH (0-2) ---
    if cluster_count < 2:
        signal_strength = 0.0
    elif cluster_count < cluster_warning:
        # Ramp from 0 to 1.0 between 2 and warning threshold
        signal_strength = (cluster_count - 2) / (cluster_warning - 2) * 0.5
    elif cluster_count < cluster_danger:
        # Between warning and danger: 0.5 to 1.0
        progress = (cluster_count - cluster_warning) / (cluster_danger - cluster_warning)
        signal_strength = 0.5 + progress * 0.5
    else:
        # At or above danger: 1.0 to 2.0 based on how extreme
        excess = cluster_count - cluster_danger
        signal_strength = 1.0 + min(excess / 3.0, 1.0)

    # Boost signal if recent days are consecutive or tightly clustered
    if cluster_count >= cluster_warning:
        last_5 = dist_df.iloc[-5:]
        recent_density = last_5["is_distribution"].sum() / 5.0
        if recent_density >= 0.6:  # 3+ of last 5 bars are distribution
            signal_strength = min(signal_strength + 0.3, 2.0)

    # Boost if stalling days are present (more ominous than simple declines)
    stall_count = int(recent["is_stall"].sum())
    if stall_count >= 2:
        signal_strength = min(signal_strength + 0.2, 2.0)

    signal_strength = float(np.clip(signal_strength, 0.0, 2.0))

    return DistributionSignal(
        distribution_days=dist_df,
        cluster_count=cluster_count,
        signal_strength=signal_strength,
        is_stalling=current_is_stalling,
        latest_cluster_start=latest_cluster_start,
    )
