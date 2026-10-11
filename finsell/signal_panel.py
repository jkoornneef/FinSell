"""Signal panel — organized view of all signals by decision category.

Groups signals into three decision frameworks:
  1. ENTER: Should I buy if I don't own it? (initial position)
  2. ADD:   Should I buy more if I already own it? (pyramiding)
  3. EXIT:  Should I sell if I own it?

Each category returns individual signal scores plus a combined read.
Some signals contribute to multiple categories differently.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field


@dataclass
class SignalReading:
    """A single signal's output."""

    name: str
    score: float
    scale: str  # e.g. "-1 to +1", "0 to 2", "-3 to +3"
    quality: float | None = None
    interpretation: str = ""
    details: dict | None = None


@dataclass
class CategoryPanel:
    """All signals for one decision category."""

    category: str  # "enter", "add", "exit"
    signals: list[SignalReading] = field(default_factory=list)
    combined_score: float = 0.0
    recommendation: str = ""


@dataclass
class SignalPanel:
    """Complete signal panel for a stock.

    Attributes:
        ticker: Stock symbol.
        current_price: Current close price.
        enter: CategoryPanel for initial entry decision.
        add: CategoryPanel for pyramiding decision.
        exit: CategoryPanel for sell decision.
    """

    ticker: str
    current_price: float
    enter: CategoryPanel = field(default_factory=lambda: CategoryPanel("enter"))
    add: CategoryPanel = field(default_factory=lambda: CategoryPanel("add"))
    exit: CategoryPanel = field(default_factory=lambda: CategoryPanel("exit"))


def _interpret_proximity(score: float, name: str) -> str:
    """Convert a -1 to +1 proximity score to a human label."""
    if score <= -0.7:
        return f"{name}: too early / not forming"
    elif score <= -0.3:
        return f"{name}: forming, watch"
    elif score <= 0.1:
        return f"{name}: at trigger point"
    elif score <= 0.5:
        return f"{name}: just triggered, actionable"
    else:
        return f"{name}: confirmed / extended"


def generate_panel(
    ticker: str,
    cost_basis: float | None = None,
    period: str = "1y",
    composite_universe: list[str] | None = None,
) -> SignalPanel:
    """Generate the full signal panel for a stock.

    Args:
        ticker: Stock symbol.
        cost_basis: Weighted average cost (required for ADD signals).
            If None, ADD signals will use current price as reference.
        period: Data fetch period (default "1y"; use "2y" for climax top).
        composite_universe: Optional ticker list for RS rating.

    Returns:
        SignalPanel with enter, add, and exit categories populated.
    """
    # Fetch data
    print(f"[DEBUG] Download : {ticker}")
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    print(f"[DEBUG] Download complete : {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel("Ticker")

    if df.empty or len(df) < 50:
        panel = SignalPanel(ticker=ticker, current_price=0.0)
        panel.enter.recommendation = "Insufficient data"
        panel.add.recommendation = "Insufficient data"
        panel.exit.recommendation = "Insufficient data"
        return panel

    current_price = float(df["Close"].iloc[-1])
    panel = SignalPanel(ticker=ticker, current_price=current_price)

    # ===================================================================
    # ENTER: Should I buy if I don't own it?
    # ===================================================================
    _populate_enter(panel.enter, df, ticker, composite_universe)

    # ===================================================================
    # ADD: Should I buy more if I already own it?
    # ===================================================================
    effective_cost = cost_basis if cost_basis else current_price
    _populate_add(panel.add, df, effective_cost)

    # ===================================================================
    # EXIT: Should I sell if I own it?
    # ===================================================================
    _populate_exit(panel.exit, df, ticker)

    return panel


def _populate_enter(cat: CategoryPanel, df: pd.DataFrame, ticker: str, universe):
    """Populate initial entry signals."""
    from finsell.cup_handle_score import score_cup_and_handle, score_pivot_buy
    from finsell.double_bottom import score_double_bottom_undercut
    from finsell.composite import composite_rating_score
    from finsell.indicators import golden_cross_indicator

    # --- Cup & Handle (timeline: -3 to +3) ---
    try:
        ch = score_cup_and_handle(df)
        cat.signals.append(SignalReading(
            name="Cup & Handle",
            score=ch.proximity_score,
            scale="-3 to +3",
            quality=(ch.depth_score + ch.shape_score) / 2.0,
            interpretation=_describe_cup_handle(ch.proximity_score),
            details=ch.best_pattern,
        ))
    except Exception:
        pass

    # --- Pivot Buy (0-2) ---
    try:
        pb = score_pivot_buy(df)
        cat.signals.append(SignalReading(
            name="Pivot Breakout",
            score=pb.score,
            scale="0 to 2",
            quality=None,
            interpretation=_describe_pivot(pb.score),
            details={"pivot_price": pb.pivot_price, "volume_ratio": pb.volume_ratio},
        ))
    except Exception:
        pass

    # --- Double Bottom Undercut (-1 to +1) ---
    try:
        db = score_double_bottom_undercut(df)
        cat.signals.append(SignalReading(
            name="Double Bottom Undercut",
            score=db.score,
            scale="-1 to +1",
            quality=db.quality,
            interpretation=_interpret_proximity(db.score, "Dbl Bottom"),
            details=db.details,
        ))
    except Exception:
        pass

    # --- Composite Rating (0-2; RS≥80 + Composite≥90) ---
    try:
        comp = composite_rating_score(ticker, universe)
        cat.signals.append(SignalReading(
            name="Composite Rating",
            score=comp,
            scale="0 to 2",
            quality=None,
            interpretation=_describe_composite(comp),
            details=None,
        ))
    except Exception:
        pass

    # --- Golden Cross (-1 to +1) ---
    try:
        gc = golden_cross_indicator(df)
        gc_current = float(gc.iloc[-1]) if not gc.empty else 0.0
        cat.signals.append(SignalReading(
            name="Golden Cross",
            score=gc_current,
            scale="-1 to +1",
            quality=None,
            interpretation=_interpret_proximity(gc_current, "Golden Cross"),
            details=None,
        ))
    except Exception:
        pass

    # --- Combined enter score ---
    _score_category(cat)


def _populate_add(cat: CategoryPanel, df: pd.DataFrame, cost_basis: float):
    """Populate pyramiding/add-on signals."""
    from finsell.pyramiding import score_flat_base, score_ma_support
    from finsell.purchase_timing import (
        score_three_weeks_tight,
        score_ascending_base,
    )

    current_price = float(df["Close"].iloc[-1])

    # --- MA Support Bounce (-1 to +1) ---
    try:
        ma = score_ma_support(df)
        cat.signals.append(SignalReading(
            name="10-Week MA Support",
            score=ma.score,
            scale="-1 to +1",
            quality=ma.quality,
            interpretation=_interpret_proximity(ma.score, "MA Bounce"),
            details=ma.details,
        ))
    except Exception:
        pass

    # --- Three-Weeks-Tight (-1 to +1) ---
    try:
        twt = score_three_weeks_tight(df)
        cat.signals.append(SignalReading(
            name="Three-Weeks-Tight",
            score=twt.score,
            scale="-1 to +1",
            quality=twt.quality,
            interpretation=_interpret_proximity(twt.score, "3-Wk Tight"),
            details=twt.details,
        ))
    except Exception:
        pass

    # --- Flat Base (-1 to +1) ---
    try:
        fb = score_flat_base(df)
        cat.signals.append(SignalReading(
            name="Flat Base",
            score=fb.score,
            scale="-1 to +1",
            quality=fb.quality,
            interpretation=_interpret_proximity(fb.score, "Flat Base"),
            details=fb.details,
        ))
    except Exception:
        pass

    # --- Ascending Base (-1 to +1) ---
    try:
        ab = score_ascending_base(df)
        cat.signals.append(SignalReading(
            name="Ascending Base",
            score=ab.score,
            scale="-1 to +1",
            quality=ab.quality,
            interpretation=_interpret_proximity(ab.score, "Ascending Base"),
            details=ab.details,
        ))
    except Exception:
        pass

    # --- Auto Add-On (2-2.5% above cost) ---
    if cost_basis > 0:
        gain = (current_price - cost_basis) / cost_basis
        if 0.02 <= gain <= 0.025:
            addon_score = 0.8
            interp = f"Auto add: +{gain*100:.1f}% (in sweet spot)"
        elif 0.015 <= gain < 0.02:
            addon_score = 0.3
            interp = f"Auto add: +{gain*100:.1f}% (approaching)"
        elif 0.025 < gain <= 0.05:
            addon_score = 0.2
            interp = f"Auto add: +{gain*100:.1f}% (slightly extended)"
        elif gain > 0.05:
            addon_score = -0.3
            interp = f"Auto add: +{gain*100:.1f}% (too extended, don't chase)"
        elif gain < 0:
            addon_score = -0.8
            interp = f"Auto add: {gain*100:.1f}% (underwater)"
        else:
            addon_score = -0.1
            interp = f"Auto add: +{gain*100:.1f}% (not yet at 2%)"

        cat.signals.append(SignalReading(
            name="Auto Add-On (2-2.5%)",
            score=addon_score,
            scale="-1 to +1",
            quality=None,
            interpretation=interp,
            details={"cost_basis": cost_basis, "gain_pct": gain},
        ))

    _score_category(cat)


def _populate_exit(cat: CategoryPanel, df: pd.DataFrame, ticker: str):
    """Populate sell/exit signals."""
    from finsell.climax_top import score_climax_top
    from finsell.distribution_days import score_distribution_days
    from finsell.broken_trend import score_broken_trend
    from finsell.overbought import score_overbought

    # --- Climax Top (-1 to +1) ---
    try:
        ct = score_climax_top(df)
        cat.signals.append(SignalReading(
            name="Climax Top",
            score=ct.score,
            scale="-1 to +1",
            quality=ct.quality,
            interpretation=_describe_climax(ct.score),
            details=ct.details,
        ))
    except Exception:
        pass

    # --- Distribution Days (-1 to +1) ---
    try:
        dd = score_distribution_days(df)
        cat.signals.append(SignalReading(
            name="Distribution Days",
            score=dd.score,
            scale="-1 to +1",
            quality=None,
            interpretation=_describe_distribution(dd.score, dd.count),
            details=dd.details,
        ))
    except Exception:
        pass

    # --- Broken Trend (-1 to +1) ---
    try:
        bt = score_broken_trend(df)
        cat.signals.append(SignalReading(
            name="Broken Trend",
            score=bt.score,
            scale="-1 to +1",
            quality=None,
            interpretation=_describe_broken_trend(bt),
            details=bt.details,
        ))
    except Exception:
        pass

    # --- Overbought / RSI (-1 to +1) ---
    try:
        ob = score_overbought(df)
        cat.signals.append(SignalReading(
            name="Overbought (RSI)",
            score=ob.score,
            scale="-1 to +1",
            quality=None,
            interpretation=_describe_overbought(ob),
            details=ob.details,
        ))
    except Exception:
        pass

    _score_category(cat)


def _score_category(cat: CategoryPanel):
    """Compute combined score and recommendation for a category."""
    if not cat.signals:
        cat.combined_score = 0.0
        cat.recommendation = "No data"
        return

    # Normalize all scores to -1 to +1 for combining
    normalized = []
    for sig in cat.signals:
        if sig.scale == "-3 to +3":
            norm = sig.score / 3.0
        elif sig.scale == "0 to 2":
            norm = sig.score - 1.0  # 0→-1, 1→0, 2→+1
        elif sig.scale == "-1 to +1":
            norm = sig.score
        else:
            norm = sig.score
        normalized.append(float(np.clip(norm, -1.0, 1.0)))

    # For ENTER and ADD: use the best signal (any single pattern justifies action)
    # For EXIT: use the worst signal (any single sell signal warrants caution)
    if cat.category in ("enter", "add"):
        # Best positive signal, but average if multiple are positive
        positive = [n for n in normalized if n > 0]
        negative = [n for n in normalized if n <= 0]
        if positive:
            cat.combined_score = float(np.mean(positive))
        else:
            cat.combined_score = float(np.max(normalized)) if normalized else 0.0
    else:
        # EXIT: highest sell signal dominates
        cat.combined_score = float(np.max(normalized)) if normalized else 0.0

    # Recommendation text
    score = cat.combined_score
    if cat.category == "enter":
        if score >= 0.5:
            cat.recommendation = "BUY — strong entry signals"
        elif score >= 0.2:
            cat.recommendation = "LEAN BUY — pattern forming, watch for trigger"
        elif score >= -0.2:
            cat.recommendation = "NEUTRAL — no clear entry"
        else:
            cat.recommendation = "WAIT — conditions not favorable"
    elif cat.category == "add":
        if score >= 0.5:
            cat.recommendation = "ADD — strong pyramiding signal"
        elif score >= 0.2:
            cat.recommendation = "LEAN ADD — opportunity developing"
        elif score >= -0.2:
            cat.recommendation = "HOLD — no add signal"
        else:
            cat.recommendation = "HOLD / WAIT — not a good add point"
    else:  # exit
        if score >= 0.5:
            cat.recommendation = "SELL — multiple warning signals"
        elif score >= 0.2:
            cat.recommendation = "TRIM — some caution warranted"
        elif score >= -0.2:
            cat.recommendation = "HOLD — no sell signal"
        else:
            cat.recommendation = "HOLD — trend healthy"


# --- Interpretation helpers ---

def _describe_cup_handle(score: float) -> str:
    if score <= -2.0:
        return "Cup & Handle: very early formation"
    elif score <= -1.0:
        return "Cup & Handle: forming, handle not yet"
    elif score <= 0.5:
        return "Cup & Handle: near breakout"
    elif score <= 1.5:
        return "Cup & Handle: just broke out"
    else:
        return "Cup & Handle: well past breakout"


def _describe_pivot(score: float) -> str:
    if score < 0.3:
        return "Pivot: not triggered"
    elif score < 1.0:
        return "Pivot: approaching buy point"
    elif score < 1.5:
        return "Pivot: at buy point with volume"
    else:
        return "Pivot: strong breakout"


def _describe_composite(score: float) -> str:
    if score < 0.5:
        return "Composite: weak (RS<80, Comp<90)"
    elif score < 1.0:
        return "Composite: one threshold met"
    elif score < 1.5:
        return "Composite: RS≥80 AND Comp≥90"
    else:
        return "Composite: elite (RS≥90, Comp≥95)"


def _describe_climax(score: float) -> str:
    if score <= -0.5:
        return "Climax Top: no exhaustion signs"
    elif score <= 0.0:
        return "Climax Top: acceleration (early warning)"
    elif score <= 0.5:
        return "Climax Top: surge in progress"
    else:
        return "Climax Top: CONFIRMED — take profits"


def _describe_distribution(score: float, count: int) -> str:
    if count <= 2:
        return f"Distribution: {count} days (normal)"
    elif count <= 4:
        return f"Distribution: {count} days (watch)"
    else:
        return f"Distribution: {count} days (HEAVY selling)"


def _describe_broken_trend(bt) -> str:
    parts = []
    if bt.trendline_break:
        parts.append("trendline broken")
    if bt.ma200_declining:
        parts.append("MA200 declining")
    if not parts:
        return "Trend: intact"
    return f"Trend: {' + '.join(parts)}"


def _describe_overbought(ob) -> str:
    if ob.rsi >= 80:
        return f"RSI {ob.rsi:.0f} — EXTREME, lock profits"
    elif ob.rsi >= 70:
        return f"RSI {ob.rsi:.0f} — overbought, consider trimming"
    elif ob.rsi >= 60:
        return f"RSI {ob.rsi:.0f} — elevated but ok"
    else:
        return f"RSI {ob.rsi:.0f} — not overbought"


def print_panel(panel: SignalPanel):
    """Print a formatted signal panel to console."""
    print(f"\n{'='*60}")
    print(f"  SIGNAL PANEL: {panel.ticker}  |  Price: ${panel.current_price:.2f}")
    print(f"{'='*60}")

    for cat in [panel.enter, panel.add, panel.exit]:
        label = {"enter": "ENTER (Buy New)", "add": "ADD (Buy More)", "exit": "EXIT (Sell)"}
        print(f"\n  [{label[cat.category]}]  Combined: {cat.combined_score:+.2f}")
        print(f"  >>> {cat.recommendation}")
        print(f"  {'-'*50}")

        for sig in cat.signals:
            qual_str = f" (q:{sig.quality:.2f})" if sig.quality is not None else ""
            print(f"    {sig.name:<24} {sig.score:+.2f} [{sig.scale}]{qual_str}")
            print(f"      {sig.interpretation}")

    print(f"\n{'='*60}\n")
