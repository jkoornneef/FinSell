"""Combined decision signals: Buy, Sell, Hold, Buy More, and Market Pulse.

Implements the CAN SLIM-based algorithmic approach:
- Market Pulse (Master Switch): Green/Yellow/Red based on distribution day count on indices.
- Buy Signal (0-1): Only fires when market is green; considers breakouts, RS, fundamentals.
- Sell Signal (0-1): Emergency exit based on stop-loss, trend breaks, climax tops.
- Hold Signal (0-1): Confidence that the position should be maintained.
- Buy More Signal (0-1): Pyramiding signal for secondary entries on existing winners.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketPulse:
    """Market direction master switch.

    Attributes:
        stage: 'green' (confirmed uptrend), 'yellow' (under pressure), 'red' (correction)
        multiplier: 1.0, 0.5, or 0.0
        distribution_count: Number of active distribution days on primary index
        stalling_count: Number of stalling days counted as distribution
        last_signal: Description of the most recent signal event
        index_name: Which index this pulse was computed from
        ftd_detected: Whether a follow-through day was detected
    """

    stage: str
    multiplier: float
    distribution_count: int
    stalling_count: int
    last_signal: str
    index_name: str
    ftd_detected: bool


@dataclass
class DualMarketPulse:
    """Combined market pulse from two indices (US + Canadian).

    The effective multiplier is the minimum of the two (conservative).
    """

    us: MarketPulse
    tsx: MarketPulse
    effective_multiplier: float
    effective_stage: str


@dataclass
class DecisionSignals:
    """Combined decision output.

    All scores are 0.0 to 1.0.
    """

    buy: float
    sell: float
    hold: float
    buy_more: float
    market_pulse: DualMarketPulse
    details: dict


def compute_market_pulse(index_df: pd.DataFrame, lookback: int = 25,
                         index_name: str = "SP500") -> MarketPulse:
    """Compute market pulse from an index DataFrame with full IBD rules.

    Distribution day rules:
        - Close down >= 0.2% on volume higher than prior day's volume
        - Stalling day: close up but gain < 0.2% on volume 25%+ above 50-day avg
        - Distribution days expire after 25 trading sessions
        - A distribution day is REMOVED if the index rallies 5%+ from that day's close
        - 0-2 active dist days → Green (Confirmed Uptrend), M=1.0
        - 3-4 active dist days → Yellow (Under Pressure), M=0.5
        - 5+  active dist days → Red (Correction), M=0.0

    Follow-Through Day (FTD) rules:
        - After market hits a new low and attempts a rally
        - On day 4-7 of rally attempt: index up >= 1.5% on volume > prior day
        - FTD resets market to Green if currently Yellow/Red
    """
    if index_df is None or len(index_df) < max(lookback + 1, 55):
        return MarketPulse(stage="green", multiplier=1.0, distribution_count=0,
                           stalling_count=0, last_signal="Insufficient data — defaulting to green",
                           index_name=index_name, ftd_detected=False)

    df = index_df.tail(lookback + 55).copy()  # extra buffer for 50-day vol avg
    df["pct_change"] = df["Close"].pct_change()
    df["vol_avg_50"] = df["Volume"].rolling(50).mean()

    # Work with the last `lookback` sessions for counting
    window = df.tail(lookback + 1).copy()
    current_close = window["Close"].iloc[-1]

    # --- Count distribution days (with 5% rally removal) ---
    dist_days = []
    stalling_days = []

    for i in range(1, len(window)):
        row = window.iloc[i]
        prev_row = window.iloc[i - 1]

        pct = row["pct_change"] if not pd.isna(row["pct_change"]) else 0
        vol_vs_prev = row["Volume"] > prev_row["Volume"]
        vol_avg = row["vol_avg_50"] if not pd.isna(row["vol_avg_50"]) else row["Volume"]
        vol_vs_avg = row["Volume"] > vol_avg * 1.25 if vol_avg > 0 else False

        # Standard distribution day: down >= 0.2% on higher volume than prior day
        if pct <= -0.002 and vol_vs_prev:
            dist_days.append({"idx": i, "date": window.index[i], "close": row["Close"]})

        # Stalling day: up but < 0.2% gain, volume 25%+ above 50-day average
        elif 0 <= pct < 0.002 and vol_vs_avg:
            stalling_days.append({"idx": i, "date": window.index[i], "close": row["Close"]})

    # Remove distribution days where index has since rallied 5% from that day's close
    active_dist = [d for d in dist_days if (current_close - d["close"]) / d["close"] < 0.05]
    active_stall = [d for d in stalling_days if (current_close - d["close"]) / d["close"] < 0.05]

    dist_count = len(active_dist) + len(active_stall)
    stall_count = len(active_stall)

    # --- Follow-Through Day detection ---
    # Find the most recent swing low, then check if day 4-7 after it has FTD criteria
    ftd_detected = False
    closes = window["Close"].values
    # Simple swing low: lowest close in last 15+ sessions
    if len(closes) >= 10:
        low_idx = np.argmin(closes[-15:]) if len(closes) >= 15 else np.argmin(closes[-10:])
        # Offset from end
        days_since_low = (len(closes) - 1) - (len(closes) - 15 + low_idx) if len(closes) >= 15 else (len(closes) - 1) - (len(closes) - 10 + low_idx)

        # Check if we're in the FTD window (day 4-10 of rally from low)
        if 4 <= days_since_low <= 10:
            # Look for a day in the rally attempt with >= 1.5% gain on higher volume
            rally_start = len(window) - days_since_low
            for j in range(rally_start + 3, min(rally_start + 8, len(window))):
                if j < len(window) and j > 0:
                    day_pct = window.iloc[j]["pct_change"]
                    day_vol = window.iloc[j]["Volume"]
                    prev_vol = window.iloc[j - 1]["Volume"]
                    if not pd.isna(day_pct) and day_pct >= 0.015 and day_vol > prev_vol:
                        ftd_detected = True
                        break

    # --- Determine stage ---
    if ftd_detected and dist_count < 5:
        # FTD can upgrade from yellow to green
        stage, mult = "green", 1.0
        signal = f"Follow-Through Day confirmed — {dist_count} dist days (reset)"
    elif dist_count >= 5:
        stage, mult = "red", 0.0
        signal = f"{dist_count} distribution days ({stall_count} stalling) — Market in Correction"
    elif dist_count >= 3:
        stage, mult = "yellow", 0.5
        signal = f"{dist_count} distribution days ({stall_count} stalling) — Uptrend Under Pressure"
    else:
        stage, mult = "green", 1.0
        signal = f"Confirmed Uptrend ({dist_count} dist days)"

    return MarketPulse(
        stage=stage, multiplier=mult, distribution_count=dist_count,
        stalling_count=stall_count, last_signal=signal,
        index_name=index_name, ftd_detected=ftd_detected,
    )


def compute_dual_market_pulse(
    us_index_df: pd.DataFrame | None,
    tsx_index_df: pd.DataFrame | None,
) -> DualMarketPulse:
    """Compute market pulse for both US (SPY) and Canadian (TSX) markets.

    The effective multiplier is the MINIMUM of the two — conservative approach.
    If a Canadian stock is in a TSX correction, even a green US market won't help.
    """
    us_pulse = compute_market_pulse(us_index_df, index_name="SP500")
    tsx_pulse = compute_market_pulse(tsx_index_df, index_name="TSX")

    eff_mult = min(us_pulse.multiplier, tsx_pulse.multiplier)
    # Effective stage from the multiplier
    if eff_mult >= 1.0:
        eff_stage = "green"
    elif eff_mult >= 0.5:
        eff_stage = "yellow"
    else:
        eff_stage = "red"

    return DualMarketPulse(
        us=us_pulse,
        tsx=tsx_pulse,
        effective_multiplier=eff_mult,
        effective_stage=eff_stage,
    )


def _score_buy(df: pd.DataFrame, market_mult: float) -> tuple[float, dict]:
    # OVERRIDE so technical recession doesn't kill all buy signals. 
    market_mult = 1.0
    """Compute raw buy signal (0-1) and return details."""
    if market_mult == 0.0:
        return 0.0, {"reason": "Market in Correction — no buys"}
    #print(f"[DEBUG] Calc Buy : MarketMult {market_mult}")

    details = {}
    components = []

    # 1. Technical breakout proximity (cup-handle, double bottom pivot)
    # Use proximity to recent high as simple proxy
    if "High" in df.columns and len(df) >= 20:
        recent_high = df["High"].rolling(20).max().iloc[-1]
        current = df["Close"].iloc[-1]
        proximity = current / recent_high if recent_high > 0 else 0
        # Score: 1.0 if within 5% of breakout, tapering to 0
        breakout_score = max(0, min(1.0, (proximity - 0.90) / 0.10))
        components.append(("breakout_proximity", breakout_score, 0.35))
        details["breakout_proximity"] = round(breakout_score, 4)
        print(f"[DEBUG] Calc Buy : breakout {breakout_score}")

    # 2. Volume surge (above-average volume on up days)
    if "Volume" in df.columns and len(df) >= 50:
        avg_vol = df["Volume"].rolling(50).mean().iloc[-1]
        recent_vol = df["Volume"].iloc[-1]
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        vol_score = max(0, min(1.0, (vol_ratio - 1.0) / 0.5))  # 1.0 at 50% above avg
        components.append(("volume_surge", vol_score, 0.20))
        details["volume_surge"] = round(vol_score, 4)
        print(f"[DEBUG] Calc Buy : volume {vol_score}")

    # 3. Price above key MAs (bullish structure)
    if len(df) >= 200:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        sma200 = df["Close"].rolling(200).mean().iloc[-1]
        price = df["Close"].iloc[-1]
        ma_score = 0.0
        if price > sma50:
            ma_score += 0.5
        if price > sma200:
            ma_score += 0.5
        components.append(("ma_structure", ma_score, 0.25))
        details["ma_structure"] = round(ma_score, 4)
        print(f"[DEBUG] Calc Buy : ma_structure {ma_score}")
    elif len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        price = df["Close"].iloc[-1]
        ma_score = 1.0 if price > sma50 else 0.0
        components.append(("ma_structure", ma_score, 0.25))
        details["ma_structure"] = round(ma_score, 4)
        print(f"[DEBUG] Calc Buy : ma_structure {ma_score}")

    # 4. RS momentum (price vs 3 months ago)
    if len(df) >= 63:
        price_now = df["Close"].iloc[-1]
        price_3mo = df["Close"].iloc[-63]
        rs_gain = (price_now - price_3mo) / price_3mo if price_3mo > 0 else 0
        rs_score = max(0, min(1.0, rs_gain / 0.20))  # 1.0 at 20% gain
        components.append(("relative_strength", rs_score, 0.20))
        details["relative_strength"] = round(rs_score, 4)
        print(f"[DEBUG] Calc Buy : relative_strength {rs_score}")

    if not components:
        return 0.0, {"reason": "Insufficient data"}

    # Weighted average
    total_weight = sum(w for _, _, w in components)
    raw = sum(s * w for _, s, w in components) / total_weight if total_weight > 0 else 0
    print(f"[DEBUG] Calc Buy : raw {raw} market_mult {market_mult}")

    return round(float(raw * market_mult), 4), details


def _score_sell(df: pd.DataFrame, wac: float | None, market_mult: float) -> tuple[float, dict]:
    """Compute sell signal (0-1)."""
    details = {}
    signals = []

    price = df["Close"].iloc[-1] if len(df) > 0 else 0
    # OVERRIDE so technical recession doesn't kill all buy signals. 
    #market_mult = 0.0
    #print(f"[DEBUG] Calc Sell : MarketMult {market_mult}")

    # 1. Stop-loss trigger (1.0 weight — the Golden Rule)
    if wac is not None and wac > 0:
        loss_pct = (price - wac) / wac
        print(f"[DEBUG] Calc Sell : loss_pct {price} {wac} {loss_pct}")
        # Standard: 7-8% below WAC. In weak market: 3-4%.
        threshold = -0.04 if market_mult < 1.0 else -0.07
        if loss_pct <= threshold:
            signals.append(("stop_loss", 1.0, 1.0))
            details["stop_loss"] = 1.0
            details["loss_pct"] = round(loss_pct, 4)
            print(f"[DEBUG] Calc Sell : stop_loss 1")
            print(f"[DEBUG] Calc Sell : loss_pct {loss_pct}")
        elif loss_pct < 0:
            # Partial: scale from 0 to 1 as loss approaches threshold
            partial = abs(loss_pct) / abs(threshold)
            signals.append(("stop_loss", min(1.0, partial), 1.0))
            details["stop_loss"] = round(min(1.0, partial), 4)
            print(f"[DEBUG] Calc Sell : stop_loss {round(min(1.0, partial), 4)}")
            print(f"[DEBUG] Calc Sell : loss_pct {loss_pct}")

    # 2. Trend break — below 50-day MA on heavy volume (0.75 weight)
    if len(df) >= 50 and "Volume" in df.columns:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        avg_vol = df["Volume"].rolling(50).mean().iloc[-1]
        recent_vol = df["Volume"].iloc[-1]
        below_50 = price < sma50
        heavy_vol = recent_vol > avg_vol * 1.25 if avg_vol > 0 else False
        if below_50 and heavy_vol:
            signals.append(("trend_break", 1.0, 0.75))
            details["trend_break"] = 1.0
            print(f"[DEBUG] Calc Sell : trend_break 1")
        elif below_50:
            signals.append(("trend_break", 0.5, 0.75))
            details["trend_break"] = 0.5
            print(f"[DEBUG] Calc Sell : trend_break 0.5")

    # 3. Climax top — large surge in short time (0.75 weight)
    if len(df) >= 15:
        gain_3wk = (price - df["Close"].iloc[-15]) / df["Close"].iloc[-15] if df["Close"].iloc[-15] > 0 else 0
        if gain_3wk >= 0.25:
            signals.append(("climax_top", 1.0, 0.75))
            details["climax_top"] = 1.0
            print(f"[DEBUG] Calc Sell : climax_top 1")
        elif gain_3wk >= 0.15:
            signals.append(("climax_top", 0.5, 0.75))
            details["climax_top"] = 0.5
            print(f"[DEBUG] Calc Sell : climax_top 0.5")

    # 4. Distribution on the stock itself (0.5 weight)
    if len(df) >= 25 and "Volume" in df.columns:
        recent_25 = df.tail(26).copy()
        recent_25["pct"] = recent_25["Close"].pct_change()
        recent_25["vol_up"] = recent_25["Volume"] > recent_25["Volume"].shift(1)
        dist_days = ((recent_25["pct"] <= -0.002) & recent_25["vol_up"]).sum()
        if dist_days >= 5:
            signals.append(("distribution", 1.0, 0.5))
            details["distribution_days"] = int(dist_days)
            print(f"[DEBUG] Calc Sell : distribution 1")
            print(f"[DEBUG] Calc Sell : distribution_days {dist_days}")
        elif dist_days >= 3:
            signals.append(("distribution", 0.5, 0.5))
            details["distribution_days"] = int(dist_days)
            print(f"[DEBUG] Calc Sell : distribution 0.5")
            print(f"[DEBUG] Calc Sell : distribution_days {dist_days}")

    if not signals:
        print(f"[DEBUG] No Signals for Sell")
        return 0.0, details

    # Take max weighted signal (sell is "worst case wins")
    max_score = max(s * w for _, s, w in signals) / max(w for _, _, w in signals)
    print(f"[DEBUG] Calc Sell : max_score {max_score}")
    return round(float(min(1.0, max_score)), 4), details


def _score_hold(df: pd.DataFrame, wac: float | None, market_mult: float) -> tuple[float, dict]:
    """Compute hold signal (0-1). High = strong reason to maintain position."""
    details = {}
    components = []

    print(f"[DEBUG] Calc Hold : MarketMult {market_mult}")

    price = df["Close"].iloc[-1] if len(df) > 0 else 0

    # 1. 8-week rule: if up 20%+ from WAC within first 3 weeks, mandatory hold
    if wac is not None and wac > 0:
        gain_pct = (price - wac) / wac
        if gain_pct >= 0.20:
            components.append(("profit_cushion", 1.0, 0.4))
            details["profit_cushion"] = 1.0
        elif gain_pct > 0:
            components.append(("profit_cushion", gain_pct / 0.20, 0.4))
            details["profit_cushion"] = round(gain_pct / 0.20, 4)
        else:
            components.append(("profit_cushion", 0.0, 0.4))
            details["profit_cushion"] = 0.0

    # 2. Above 10-week (50-day) moving average on light volume = normal consolidation
    if len(df) >= 50 and "Volume" in df.columns:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        avg_vol = df["Volume"].rolling(50).mean().iloc[-1]
        recent_vol = df["Volume"].tail(5).mean()
        above_ma = price >= sma50
        light_vol = recent_vol < avg_vol * 0.9 if avg_vol > 0 else True
        if above_ma and light_vol:
            components.append(("ma_support", 1.0, 0.35))
            details["ma_support"] = 1.0
        elif above_ma:
            components.append(("ma_support", 0.7, 0.35))
            details["ma_support"] = 0.7
        else:
            components.append(("ma_support", 0.0, 0.35))
            details["ma_support"] = 0.0

    # 3. Market is supportive
    components.append(("market_support", market_mult, 0.25))
    details["market_support"] = market_mult

    if not components:
        return 0.5, details

    total_weight = sum(w for _, _, w in components)
    raw = sum(s * w for _, s, w in components) / total_weight if total_weight > 0 else 0.5
    return round(float(raw), 4), details


def _score_buy_more(df: pd.DataFrame, wac: float | None, market_mult: float) -> tuple[float, dict]:
    """Compute buy-more/pyramiding signal (0-1). Only fires in green market with existing profit."""
    details = {}

    print(f"[DEBUG] Calc BuyMore : MarketMult {market_mult}")

    if market_mult < 1.0:
        market_mult = .10
        print(f"[DEBUG] Calc BuyMore : Market not in confirmed uptrend — reducing buy-more multiplier to {market_mult}")
        
#    if market_mult < 1.0:
#        return 0.0, {"reason": "Market not in confirmed uptrend — no pyramiding"}

    if wac is None or wac <= 0:
        print(f"[DEBUG] Calc BuyMore : Watching")
        return 0.0, {"reason": "No WAC — use Buy signal instead"}

    price = df["Close"].iloc[-1] if len(df) > 0 else 0
    gain_pct = (price - wac) / wac

    print(f"[DEBUG] Calc BuyMore : gain_pct {gain_pct}")

    # Must be in profit (at least 2-3% above last purchase)
    if gain_pct < 0.02:
        return 0.0, {"reason": f"Not yet 2% above WAC (gain: {gain_pct*100:.1f}%)"}

    components = []

    # 1. Profit threshold met (2-3% above WAC)
    profit_score = min(1.0, gain_pct / 0.05)  # maxes at 5% above
    components.append(("profit_above_wac", profit_score, 0.30))
    details["profit_above_wac"] = round(profit_score, 4)
    print(f"[DEBUG] Calc BuyMore : profit_above_wac {profit_score}")

    # 2. 10-week MA support bounce
    if len(df) >= 50 and "Volume" in df.columns:
        sma50 = df["Close"].rolling(50).mean().iloc[-1]
        # Within 2% of 50-day MA and holding above = support bounce
        dist_to_ma = (price - sma50) / sma50 if sma50 > 0 else 0
        if 0 <= dist_to_ma <= 0.03:
            # Near MA and above — ideal buy-more zone
            avg_vol = df["Volume"].rolling(50).mean().iloc[-1]
            recent_vol = df["Volume"].iloc[-1]
            vol_spike = recent_vol > avg_vol * 1.4 if avg_vol > 0 else False
            bounce_score = 0.8 if vol_spike else 0.5
            components.append(("ma_bounce", bounce_score, 0.35))
            details["ma_bounce"] = round(bounce_score, 4)
            print(f"[DEBUG] Calc BuyMore : ma_bounce {bounce_score}")
        elif dist_to_ma > 0.03:
            components.append(("ma_bounce", 0.3, 0.35))
            details["ma_bounce"] = 0.3
            print(f"[DEBUG] Calc BuyMore : ma_bounce {0.3}")
        else:
            components.append(("ma_bounce", 0.0, 0.35))
            details["ma_bounce"] = 0.0
            print(f"[DEBUG] Calc BuyMore : ma_bounce {0.0}")

    # 3. Tight consolidation (low volatility in recent 3 weeks)
    if len(df) >= 15:
        recent_15 = df["Close"].tail(15)
        pct_range = (recent_15.max() - recent_15.min()) / recent_15.mean() if recent_15.mean() > 0 else 1
        # Tight if range < 5%
        tight_score = max(0, 1.0 - (pct_range / 0.05))
        components.append(("consolidation_tight", tight_score, 0.35))
        details["consolidation_tight"] = round(tight_score, 4)
        print(f"[DEBUG] Calc BuyMore : consolidation_tight {tight_score}")


    if not components:
        return 0.0, details

    total_weight = sum(w for _, _, w in components)
    raw = sum(s * w for _, s, w in components) / total_weight if total_weight > 0 else 0
    print(f"[DEBUG] Calc BuyMore : raw {raw} market_mult {market_mult}")
    return round(float(raw), 4), details


def compute_decision_signals(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
    tsx_index_df: pd.DataFrame | None = None,
    wac: float | None = None,
    ticker: str = "",
    market_override: str = "auto",
) -> DecisionSignals:
    """Compute all decision signals for a stock.

    Args:
        stock_df: Daily OHLCV DataFrame for the stock.
        index_df: Daily OHLCV DataFrame for US market index (SPY).
        tsx_index_df: Daily OHLCV DataFrame for TSX index (^GSPTSE).
        wac: Weighted average cost basis. If None, Buy More signal is disabled.
        ticker: Stock ticker symbol. Used to determine which market pulse applies.
        market_override: Which market pulse to use for scoring.
            "auto" — US for US stocks, dual (conservative min) for .TO/.V stocks.
            "us" — Force US market pulse only.
            "tsx" — Force TSX market pulse only.
            "dual" — Force conservative min of both markets.

    Returns:
        DecisionSignals with buy, sell, hold, buy_more scores and market_pulse.
    """
    # Dual market pulse
    dual_pulse = compute_dual_market_pulse(index_df, tsx_index_df)

    print(f"[DEBUG] Ticker: {ticker} Market Override: {market_override}")

    # Pick the relevant multiplier
    if market_override == "us":
        market_mult = dual_pulse.us.multiplier
    elif market_override == "tsx":
        market_mult = dual_pulse.tsx.multiplier
    elif market_override == "dual":
        market_mult = dual_pulse.effective_multiplier
    else:  # "auto"
        is_canadian = ticker.upper().endswith(".TO") or ticker.upper().endswith(".V")
        if is_canadian:
            market_mult = dual_pulse.effective_multiplier
        else:
            market_mult = dual_pulse.us.multiplier

    # Individual signals
    buy, buy_details = _score_buy(stock_df, market_mult)
    sell, sell_details = _score_sell(stock_df, wac, market_mult)
    hold, hold_details = _score_hold(stock_df, wac, market_mult)
    buy_more, bm_details = _score_buy_more(stock_df, wac, market_mult)

    details = {
        "buy": buy_details,
        "sell": sell_details,
        "hold": hold_details,
        "buy_more": bm_details,
        "market_us": {
            "stage": dual_pulse.us.stage,
            "dist_days": dual_pulse.us.distribution_count,
            "stalling": dual_pulse.us.stalling_count,
            "ftd": dual_pulse.us.ftd_detected,
        },
        "market_tsx": {
            "stage": dual_pulse.tsx.stage,
            "dist_days": dual_pulse.tsx.distribution_count,
            "stalling": dual_pulse.tsx.stalling_count,
            "ftd": dual_pulse.tsx.ftd_detected,
        },
    }

    return DecisionSignals(
        buy=buy,
        sell=sell,
        hold=hold,
        buy_more=buy_more,
        market_pulse=dual_pulse,
        details=details,
    )
