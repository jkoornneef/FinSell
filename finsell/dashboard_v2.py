"""Finsell visual dashboard — multi-mode: Holdings, Research, Compare, Account."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from finsell.data_cache import CACHE_DIR, fetch_all, fetch_combined, invalidate_cache
from finsell.portfolio import Holding, Portfolio
from finsell.composite import composite_rating_score

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Finsell Dashboard")
_parser.add_argument("-t", "--ticker", default="ENB.TO", help="Initial ticker")
_parser.add_argument("-s", "--sector", default=None, help="Sector override")
_parser.add_argument("--portfolio", type=str, default=None, help="Override portfolio JSON file path (e.g., data/portfolio2.json or C:/temp/test_portfolio.json)")
_parser.add_argument("-p", "--port", type=int, default=8050, help="Port")
_args = _parser.parse_known_args()[0]

_default_ticker = _args.ticker.upper()
_default_sector = _args.sector
_port = _args.port

# ---------------------------------------------------------------------------
# Portfolio singleton (with override support)
# ---------------------------------------------------------------------------

if _args.portfolio:
    # Use the provided portfolio file path, converting backslashes to forward slashes for cross-platform compatibility
    _portfolio_path = _args.portfolio.replace('\\', '/')
else:
    _portfolio_path = None

_portfolio = Portfolio.load(_portfolio_path)

# ---------------------------------------------------------------------------
# Gauge helpers
# ---------------------------------------------------------------------------


def gauge_0_to_1(value, title, thresholds=None):
    thresholds = thresholds or {"green": 0.7, "yellow": 0.4}
    try:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    value = round(value, 4)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number", value=value,
            title={"text": title},
            number={"valueformat": ".4f"},
            domain={"x": [0.1, 0.9], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1]},
                "bar": {"color": "#1F77B4"},
                "steps": [
                    {"range": [0, thresholds["yellow"]], "color": "#FFCCCC"},
                    {"range": [thresholds["yellow"], thresholds["green"]], "color": "#FFF3CD"},
                    {"range": [thresholds["green"], 1], "color": "#D4EDDA"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8, "value": value},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def gauge_neg1_to_pos1(value, title):
    try:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    value = round(value, 4)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number", value=value,
            title={"text": title},
            number={"valueformat": ".4f"},
            domain={"x": [0.1, 0.9], "y": [0, 1]},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar": {"color": "#444"},
                "steps": [
                    {"range": [-1, -0.5], "color": "#D62728"},
                    {"range": [-0.5, 0], "color": "#FFCCCC"},
                    {"range": [0, 0.5], "color": "#D4EDDA"},
                    {"range": [0.5, 1], "color": "#2CA02C"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8, "value": value},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def mini_gauge(value, title, range_type="neg1_pos1"):
    try:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    value = round(value, 4)
    if range_type == "0_1":
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number={"font": {"size": 14}, "valueformat": ".4f"},
            title={"text": title, "font": {"size": 11}},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.5, 1], "tickfont": {"size": 8}},
                "bar": {"color": "#1F77B4", "thickness": 0.6},
                "steps": [
                    {"range": [0, 0.4], "color": "#FFCCCC"},
                    {"range": [0.4, 0.7], "color": "#FFF3CD"},
                    {"range": [0.7, 1], "color": "#D4EDDA"},
                ],
            },
        ))
    else:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number={"font": {"size": 14}, "valueformat": ".4f"},
            title={"text": title, "font": {"size": 11}},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, 0, 1], "tickfont": {"size": 8}},
                "bar": {"color": "#444", "thickness": 0.6},
                "steps": [
                    {"range": [-1, -0.3], "color": "#D62728"},
                    {"range": [-0.3, 0.3], "color": "#F0F0F0"},
                    {"range": [0.3, 1], "color": "#2CA02C"},
                ],
            },
        ))
    fig.update_layout(height=160, margin=dict(t=40, b=5, l=15, r=15))
    return fig


def decision_gauge(value, title, color_scheme="green"):
    try:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
    except (TypeError, ValueError):
        value = 0.0
    value = round(value, 4)
    colors = {
        "green": {"bar": "#2CA02C", "steps": [
            {"range": [0, 0.3], "color": "#F0F0F0"},
            {"range": [0.3, 0.7], "color": "#D4EDDA"},
            {"range": [0.7, 1], "color": "#2CA02C"},
        ]},
        "red": {"bar": "#D62728", "steps": [
            {"range": [0, 0.3], "color": "#F0F0F0"},
            {"range": [0.3, 0.7], "color": "#FFF3CD"},
            {"range": [0.7, 1], "color": "#D62728"},
        ]},
        "blue": {"bar": "#1F77B4", "steps": [
            {"range": [0, 0.3], "color": "#F0F0F0"},
            {"range": [0.3, 0.7], "color": "#CCE5FF"},
            {"range": [0.7, 1], "color": "#1F77B4"},
        ]},
    }
    scheme = colors.get(color_scheme, colors["green"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"font": {"size": 16}, "valueformat": ".4f"},
        title={"text": title, "font": {"size": 12}},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, 0.5, 1], "tickfont": {"size": 9}},
            "bar": {"color": scheme["bar"], "thickness": 0.7},
            "steps": scheme["steps"],
        },
    ))
    fig.update_layout(height=180, margin=dict(t=45, b=10, l=15, r=15))
    return fig


_GAUGE_TOOLTIP_TEXT = {
    "composite score": "Composite Score\nApproximates IBD-style Composite Rating.\nHigher is better. Range: 0 to 1.",
    "trend strength": "Trend Strength\nCalculates momentum remapped from the signal average into a 0 to 1 strength gauge.\nHigher is better. Range: 0 to 1.",
    "momentum": "Momentum\nCalculates the average of the active pattern and timing signals.\nSpan: -1 to 1. Negative is bearish, positive is bullish.",
    "cup handle": "Cup & Handle\nCalculates how well price matches a cup-with-handle base.\nHigher is better. Range: 0 to 1.",
    "distribution": "Distribution\nCalculates recent distribution-day selling pressure from down days on elevated volume.\nSpan: -1 to 1. Lower is better, higher is worse.",
    "climax top": "Climax Top\nCalculates how close price action is to a blow-off or climax-top move.\nSpan: -1 to 1. Lower is better, higher is worse.",
    "broken trend": "Broken Trend\nCalculates whether the stock has broken key trend support or uptrend structure.\nSpan: -1 to 1. Lower is better, higher is worse.",
    "overbought": "Overbought\nCalculates RSI-based overbought pressure for profit-taking risk.\nSpan: -1 to 1. Lower is better, higher is worse.",
    "double bottom": "Double Bottom\nCalculates proximity to a double-bottom undercut breakout.\nSpan: -1 to 1. Higher is better.",
    "3 wks tight": "3 Wks Tight\nCalculates whether price has formed a three-weeks-tight pattern and is near breakout.\nSpan: -1 to 1. Higher is better.",
    "three wks tight": "3 Wks Tight\nCalculates whether price has formed a three-weeks-tight pattern and is near breakout.\nSpan: -1 to 1. Higher is better.",
    "ascending base": "Ascending Base\nCalculates proximity to an ascending-base breakout after rising pullbacks.\nSpan: -1 to 1. Higher is better.",
    "flat base": "Flat Base\nCalculates proximity to a flat-base breakout after a tight sideways consolidation.\nSpan: -1 to 1. Higher is better.",
    "ma support": "MA Support\nCalculates whether price is finding support at a key moving average and bouncing.\nSpan: -1 to 1. Higher is better.",
    "golden cross": "Golden Cross\nCalculates proximity to the 50-day SMA crossing above the 200-day SMA.\nSpan: -1 to 1. Negative is before the cross, positive is after it.",
    "purchase timing": "Purchase Timing\nCalculates add-on timing from secondary buy signals like MA support, tight action, and bases.\nSpan: -1 to 1. Higher is better.",
    "buy": "BUY\nCalculates the weighted entry signal for a new position.\nHigher is better. Range: 0 to 1.",
    "sell": "SELL\nCalculates exit urgency from stop-loss, trend-break, and climax-top style signals.\nHigher is worse. Range: 0 to 1.",
    "hold": "HOLD\nCalculates confidence that the current position should still be held.\nHigher is better. Range: 0 to 1.",
    "buy more": "BUY MORE\nCalculates pyramiding or add-on strength for an existing winner.\nHigher is better. Range: 0 to 1.",
}


def _gauge_tooltip_text(title: str) -> str:
    key = " ".join(str(title or "").lower().split())
    return _GAUGE_TOOLTIP_TEXT.get(key, f"{title}\nGauge tooltip not defined.")


def _coerce_decision_scores(decisions) -> dict:
    """Normalize decision scores from either a dict or a score object."""
    if isinstance(decisions, dict):
        return decisions
    return {name: getattr(decisions, name, None) for name in ("buy", "sell", "hold", "buy_more")}


def _sanitize_pattern_scores(patterns: dict | None) -> dict:
    """Drop missing/NaN pattern values before using them in momentum or gauge math."""
    if not patterns:
        return {}
    cleaned = {}
    for key, value in patterns.items():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isnan(numeric):
            continue
        cleaned[key] = numeric
    return cleaned


def _pattern_momentum(patterns: dict | None) -> float:
    """Average valid pattern values while ignoring NaN/None entries."""
    valid = list(_sanitize_pattern_scores(patterns).values())
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


def _account_ticker_options(holdings) -> list[dict]:
    """Build checklist options for the Account tab from holdings/watchers."""
    if hasattr(holdings, "holdings"):
        holdings = holdings.holdings

    options = []
    for item in holdings or []:
        if isinstance(item, tuple):
            ticker, is_watcher = item
            ticker = str(ticker or "").upper().strip()
            is_watcher = bool(is_watcher)
        else:
            ticker = getattr(item, "ticker", "")
            ticker = str(ticker or "").upper().strip()
            is_watcher = getattr(item, "is_watcher", False)

        if not ticker:
            continue

        options.append({
            "label": f"{ticker} ({'Watcher' if is_watcher else 'Holding'})",
            "value": ticker,
        })

    return options


def _build_research_gauge_export_payload(ticker: str, patterns: dict, decisions) -> list[dict]:
    """Build a JSON-friendly gauge export payload for the current research stock."""
    pattern_labels = {
        "cup_handle": "Cup & Handle", 
        "distribution": "Distribution",
        "climax_top": "Climax Top",
        "broken_trend": "Broken Trend",
        "overbought": "Overbought",
        "double_bottom": "Double Bottom",
        "three_wks_tight": "3 Wks Tight",
        "ascending_base": "Ascending Base",
        "flat_base": "Flat Base",
        "ma_support": "MA Support",
        "golden_cross": "Golden Cross",
        "purchase_timing": "Purchase Timing",
    }

    entries = []

    def add_entry(name: str, value, description: str | None = None):
        entries.append({
            "name": name,
            "value": round(float(value), 6) if value is not None else None,
            "description": description or _gauge_tooltip_text(name),
        })

    # Main research gauges use the strongest pattern score as the composite score.
    composite_value = patterns.get("cup_handle") if patterns else None
    if composite_value is None:
        composite_value = composite_rating_score(ticker)


    add_entry("Composite Score", composite_value, _gauge_tooltip_text("Composite Score"))
    add_entry("Trend Strength", (sum(v for v in patterns.values() if v is not None) / len([v for v in patterns.values() if v is not None])) if patterns else 0,
              _gauge_tooltip_text("Trend Strength"))
    add_entry("Momentum", ((sum(v for v in patterns.values() if v is not None) / len([v for v in patterns.values() if v is not None])) - 0.5) * 2 if patterns else 0,
              _gauge_tooltip_text("Momentum"))

    for key, value in patterns.items():
        if value is None:
            continue
        label = pattern_labels.get(key, key.replace("_", " ").title())
        add_entry(label, value, _gauge_tooltip_text(label))

    decision_scores = _coerce_decision_scores(decisions)
    decision_labels = {
        "buy": "BUY",
        "sell": "SELL",
        "hold": "HOLD",
        "buy_more": "BUY MORE",
    }
    for key, value in decision_scores.items():
        if value is None:
            continue
        label = decision_labels.get(key, key.replace("_", " ").title())
        add_entry(label, value, _gauge_tooltip_text(label))

    return entries


def _gauge_graph(title: str, graph_id=None, figure=None, config=None):
    graph_kwargs = {}
    if graph_id is not None:
        graph_kwargs["id"] = graph_id
    if figure is not None:
        graph_kwargs["figure"] = figure
    if config is not None:
        graph_kwargs["config"] = config

    return html.Div(
        dcc.Graph(**graph_kwargs),
        title=_gauge_tooltip_text(title),
        style={"cursor": "help"},
    )

def signal_bar(value, max_val=1.0, color="blue"):
    """Inline horizontal bar (returns a Div)."""
    pct = max(0, min(100, (value / max_val) * 100)) if max_val else 0
    bar_color = {"green": "#2CA02C", "red": "#D62728", "blue": "#1F77B4", "orange": "#FF7F0E"}.get(color, color)
    return html.Div(
        html.Div(style={"width": f"{pct}%", "height": "100%", "background": bar_color, "borderRadius": "3px"}),
        style={"width": "100%", "height": "14px", "background": "#E9ECEF", "borderRadius": "3px"},
    )


_WINDOW_DAYS = {"3m": 90, "6m": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
_SCOPE_OPTIONS = [
    {"label": "Daily", "value": "daily"},
    {"label": "Weekly (TFSA)", "value": "weekly"},
    {"label": "Monthly", "value": "monthly"},
]


def _merge_history(monthly, daily):
    """Merge cached monthly + daily history into one OHLCV frame."""
    frames = []
    if monthly is not None and len(monthly) > 0:
        monthly = monthly.copy()
        monthly["_resolution"] = "monthly"
        frames.append(monthly)
    if daily is not None and len(daily) > 0:
        daily = daily.copy()
        daily["_resolution"] = "daily"
        frames.append(daily)
    if not frames:
        return None

    merged = frames[0] if len(frames) == 1 else None
    if len(frames) > 1:
        cutoff = daily.index.min()
        monthly_trimmed = monthly[monthly.index < cutoff] if monthly is not None and len(monthly) > 0 else None
        parts = [df for df in [monthly_trimmed, daily] if df is not None and len(df) > 0]
        merged = pd.concat(parts) if parts else None
    if merged is None or len(merged) == 0:
        return None
    merged = merged.sort_index()
    return merged


def _resample_ohlcv(df, rule: str):
    """Resample OHLCV history to weekly/monthly bars."""
    if df is None or len(df) == 0:
        return df
    agg = {}
    if "Open" in df.columns:
        agg["Open"] = "first"
    if "High" in df.columns:
        agg["High"] = "max"
    if "Low" in df.columns:
        agg["Low"] = "min"
    if "Close" in df.columns:
        agg["Close"] = "last"
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    resampled = df.resample(rule).agg(agg)
    if "Close" in resampled.columns:
        resampled = resampled.dropna(subset=["Close"])
    return resampled


def _select_scope_df(monthly, daily, scope: str):
    """Return history at the requested scope."""
    scope = (scope or "daily").lower()
    if scope == "daily":
        return daily.copy() if daily is not None else None
    merged = _merge_history(monthly, daily)
    if merged is None:
        return None
    if scope == "weekly":
        return _resample_ohlcv(merged, "W-FRI")
    if scope == "monthly":
        return _resample_ohlcv(merged, "ME")
    return daily.copy() if daily is not None else merged


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


def _compute_scores(ticker: str, wac: float | None = None, sector_override: str | None = None,
                    market_override: str = "auto", scope: str = "daily"):
    """Compute all scores for a ticker. Returns (decisions, pattern_scores, info) or None."""
    from finsell.cup_handle_score import score_cup_and_handle
    from finsell.distribution_days import score_distribution_days
    from finsell.climax_top import score_climax_top
    from finsell.broken_trend import score_broken_trend
    from finsell.overbought import score_overbought
    from finsell.double_bottom import score_double_bottom_undercut
    from finsell.purchase_timing import score_three_weeks_tight, score_ascending_base, score_purchase_timing
    from finsell.pyramiding import score_flat_base, score_ma_support
    from finsell.indicators import golden_cross_indicator
    from finsell.decision_signals import compute_decision_signals

    try:
        data = fetch_all(ticker, sector_override=sector_override)
    except Exception:
        return None

    daily = data.get("daily")
    monthly = data.get("monthly")
    score_df = _select_scope_df(monthly, daily, scope)
    if score_df is None or len(score_df) < 12:
        return None

    def safe(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception:
            return None

    cup = safe(score_cup_and_handle, score_df)
    dist = safe(score_distribution_days, score_df)
    climax = safe(score_climax_top, score_df)
    broken = safe(score_broken_trend, score_df)
    ob = safe(score_overbought, score_df)
    dbl = safe(score_double_bottom_undercut, score_df)
    twt = safe(score_three_weeks_tight, score_df)
    asc = safe(score_ascending_base, score_df)
    flat = safe(score_flat_base, score_df)
    ma_sup = safe(score_ma_support, score_df)
    gc_series = safe(golden_cross_indicator, score_df)
    pt = safe(score_purchase_timing, score_df, cost_basis=wac)

    patterns = {
        "cup_handle": cup.depth_score if cup else None,
        "distribution": dist.score if dist else None,
        "climax_top": climax.score if climax else None,
        "broken_trend": broken.score if broken else None,
        "overbought": ob.score if ob else None,
        "double_bottom": dbl.score if dbl else None,
        "three_wks_tight": twt.score if twt else None,
        "ascending_base": asc.score if asc else None,
        "flat_base": flat.score if flat else None,
        "ma_support": ma_sup.score if ma_sup else None,
        "golden_cross": float(gc_series.iloc[-1]) if gc_series is not None and len(gc_series) > 0 else None,
        "purchase_timing": pt.score if pt else None,
    }

    # Market pulse
    indices = data.get("indices", {})
    sp500_monthly, sp500_daily = indices.get("SP500", (None, None))
    tsx_monthly, tsx_daily = indices.get("TSX", (None, None))
    sp500 = _select_scope_df(sp500_monthly, sp500_daily, scope)
    tsx = _select_scope_df(tsx_monthly, tsx_daily, scope)

    decisions = compute_decision_signals(
        stock_df=score_df, index_df=sp500, tsx_index_df=tsx,
        wac=wac, ticker=ticker, market_override=market_override,
    )

    return decisions, patterns, data["info"]


# ---------------------------------------------------------------------------
# App + Layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Finsell Dashboard",
    suppress_callback_exceptions=True,
)


def _holdings_tab():
    """Layout for Holdings mode."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("All", id="filter-all", color="primary", size="sm", outline=True, active=True),
                    dbc.Button("Holdings", id="filter-holdings", color="primary", size="sm", outline=True),
                    dbc.Button("Watchers", id="filter-watchers", color="primary", size="sm", outline=True),
                ], className="me-3"),
                dcc.Dropdown(id="holdings-scope", options=_SCOPE_OPTIONS, value="daily", clearable=False,
                             style={"width": "180px", "display": "inline-block"}, className="me-2"),
                dbc.Button("+ Add", id="btn-add-holding", color="success", size="sm", className="me-2"),
                dbc.Button("Refresh Signals", id="btn-refresh-holdings", color="info", size="sm"),
            ], md=8),
            dbc.Col([
                html.Div(id="holdings-market-pulse", className="text-end"),
            ], md=4),
        ], className="mb-3"),

        # Attention summaries
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("🚨 Sell Alerts", className="bg-danger text-white py-1"),
                dbc.CardBody(html.Div(id="alert-sell"), className="p-2"),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("🚀 Buy More", className="bg-success text-white py-1"),
                dbc.CardBody(html.Div(id="alert-buy-more"), className="p-2"),
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardHeader("👀 Watcher Buy", className="bg-info text-white py-1"),
                dbc.CardBody(html.Div(id="alert-watcher-buy"), className="p-2"),
            ]), md=4),
        ], className="mb-3"),

        # Holdings grid
        html.Div(id="holdings-grid"),

        # Detail area (shown when user clicks a row)
        html.Hr(),
        html.Div(id="holdings-detail"),

        # Add holding modal
        dbc.Modal([
            dbc.ModalHeader("Add / Edit Holding"),
            dbc.ModalBody([
                dbc.Row([
                    dbc.Col([dbc.Label("Ticker"), dbc.Input(id="modal-ticker", placeholder="ENB.TO")], md=3),
                    dbc.Col([dbc.Label("Shares"), dbc.Input(id="modal-shares", type="number", value=0)], md=3),
                    dbc.Col([dbc.Label("Avg Cost"), dbc.Input(id="modal-wac", type="number", value=0, step=0.01)], md=3),
                    dbc.Col([dbc.Label("Market"), dcc.Dropdown(
                        id="modal-market",
                        options=[
                            {"label": "Auto", "value": "auto"},
                            {"label": "US", "value": "us"},
                            {"label": "TSX", "value": "tsx"},
                            {"label": "Both", "value": "dual"},
                        ], value="auto", clearable=False,
                    )], md=3),
                ]),
                dbc.Row([
                    dbc.Col([dbc.Label("Sector"), dbc.Input(id="modal-sector", placeholder="(auto)")], md=4),
                    dbc.Col([dbc.Label("Notes"), dbc.Input(id="modal-notes", placeholder="optional")], md=8),
                ], className="mt-2"),
                html.Small(id="modal-status", className="text-muted d-block mt-3"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Reload from yfinance", id="modal-reload-yf", color="info", outline=True, className="me-auto"),
                dbc.Button("Save", id="modal-save", color="primary"),
                dbc.Button("Delete", id="modal-delete", color="danger", outline=True),
                dbc.Button("Cancel", id="modal-cancel", color="secondary"),
            ]),
        ], id="modal-holding", is_open=False),

        # Hidden stores
        dcc.Store(id="holdings-data-store"),
        dcc.Store(id="selected-holding-ticker"),
        dcc.Store(id="holdings-refresh-token", data="init"),
    ], fluid=True)


def _research_tab():
    """Layout for Research mode (single stock deep-dive)."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Label("Ticker"),
                dbc.Input(id="research-ticker", type="text", value=_default_ticker, debounce=True),
            ], md=2),
            dbc.Col([
                dbc.Label("Time window"),
                dcc.Dropdown(id="research-window", options=[
                    {"label": "3 months", "value": "3m"},
                    {"label": "6 months", "value": "6m"},
                    {"label": "1 year", "value": "1y"},
                    {"label": "2 years", "value": "2y"},
                    {"label": "5 years", "value": "5y"},
                    {"label": "10 years", "value": "10y"},
                ], value="1y", clearable=False),
            ], md=2),
            dbc.Col([
                dbc.Label("Overlays"),
                dcc.Checklist(id="research-overlays", options=[
                    {"label": " High", "value": "high"},
                    {"label": " Low", "value": "low"},
                    {"label": " Volume", "value": "volume"},
                    {"label": " Dividends", "value": "dividends"},
                    {"label": " DRIP", "value": "drip"},
                    {"label": " Splits", "value": "splits"},
                ], value=["volume"], inline=True,
                    inputStyle={"marginRight": "4px", "marginLeft": "12px"}),
            ], md=4),
            dbc.Col([
                dbc.Label("Market Pulse"),
                dcc.Dropdown(id="research-pulse-override", options=[
                    {"label": "Auto (US/.TO)", "value": "auto"},
                    {"label": "US only", "value": "us"},
                    {"label": "TSX only", "value": "tsx"},
                    {"label": "Both (conservative)", "value": "dual"},
                ], value="auto", clearable=False),
            ], md=2),
            dbc.Col([
                dbc.Label("Signal Scope"),
                dcc.Dropdown(id="research-scope", options=_SCOPE_OPTIONS, value="daily", clearable=False),
            ], md=2),
            dbc.Col([
                dbc.Label("WAC"),
                dbc.Input(id="research-wac", type="number", placeholder="opt", step=0.01),
            ], md=2),
            dbc.Col([
                dbc.Button("Export Gauge Params", id="research-export-btn", color="secondary", outline=True, size="sm"),
            ], md=2, className="d-flex align-items-end"),
        ], className="mb-3"),
        dbc.Row(dbc.Col(html.Div(id="research-info-bar"), md=12), className="mb-2"),
        dcc.Download(id="research-export-download"),
        dcc.Store(id="research-export-data", data={}),
        dbc.Row(dbc.Col(dcc.Graph(id="research-chart"), md=12)),
        dbc.Row([
            dbc.Col(_gauge_graph("Composite Score", graph_id="research-gauge-score"), md=4),
            dbc.Col(_gauge_graph("Trend Strength", graph_id="research-gauge-strength"), md=4),
            dbc.Col(_gauge_graph("Momentum", graph_id="research-gauge-momentum"), md=4),
        ], className="mt-3"),
        html.Hr(),
        html.H5("Pattern & Signal Scores", className="text-center text-muted mb-2"),
        dbc.Row([
            dbc.Col(_gauge_graph("Cup Handle", graph_id="research-mg-cup-handle"), md=2),
            dbc.Col(_gauge_graph("Distribution", graph_id="research-mg-distribution"), md=2),
            dbc.Col(_gauge_graph("Climax Top", graph_id="research-mg-climax-top"), md=2),
            dbc.Col(_gauge_graph("Broken Trend", graph_id="research-mg-broken-trend"), md=2),
            dbc.Col(_gauge_graph("Overbought", graph_id="research-mg-overbought"), md=2),
            dbc.Col(_gauge_graph("Double Bottom", graph_id="research-mg-double-bottom"), md=2),
        ]),
        dbc.Row([
            dbc.Col(_gauge_graph("3 Wks Tight", graph_id="research-mg-three-weeks-tight"), md=2),
            dbc.Col(_gauge_graph("Ascending Base", graph_id="research-mg-ascending-base"), md=2),
            dbc.Col(_gauge_graph("Flat Base", graph_id="research-mg-flat-base"), md=2),
            dbc.Col(_gauge_graph("MA Support", graph_id="research-mg-ma-support"), md=2),
            dbc.Col(_gauge_graph("Golden Cross", graph_id="research-mg-golden-cross"), md=2),
            dbc.Col(_gauge_graph("Purchase Timing", graph_id="research-mg-purchase-timing"), md=2),
        ], className="mb-3"),
        html.Hr(),
        html.H5("Decision Signals", className="text-center mb-2"),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Market Pulse", className="text-center"),
                html.Div(id="research-pulse-indicator", className="text-center"),
            ])]), md=2),
            dbc.Col(_gauge_graph("BUY", graph_id="research-dg-buy"), md=2),
            dbc.Col(_gauge_graph("SELL", graph_id="research-dg-sell"), md=2),
            dbc.Col(_gauge_graph("HOLD", graph_id="research-dg-hold"), md=2),
            dbc.Col(_gauge_graph("BUY MORE", graph_id="research-dg-buy-more"), md=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Details", className="text-center"),
                html.Div(id="research-details", style={"fontSize": "11px"}),
            ])]), md=2),
        ], className="mb-3"),
    ], fluid=True)


def _account_tab():
    """Layout for Account mode (all holdings and watchers performance)."""
    account_tickers = [h.ticker for h in _portfolio.holdings if h.ticker]
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Label("Window"),
                dcc.Dropdown(id="account-window", options=[
                    {"label": "3 months", "value": "3m"},
                    {"label": "6 months", "value": "6m"},
                    {"label": "1 year", "value": "1y"},
                    {"label": "2 years", "value": "2y"},
                    {"label": "5 years", "value": "5y"},
                ], value="1y", clearable=False),
            ], md=3),
            dbc.Col([
                dbc.Label("Signal Scope"),
                dcc.Dropdown(id="account-scope", options=_SCOPE_OPTIONS, value="monthly", clearable=False),
            ], md=3),
            dbc.Col([
                dbc.Label("Tickers"),
                dcc.Checklist(
                    id="account-tickers",
                    options=_account_ticker_options(_portfolio.holdings),
                    value=account_tickers,
                    inline=False,
                    inputStyle={"marginRight": "6px"},
                    className="small",
                    style={"columnCount": 3, "columnGap": "1rem"},
                ),
            ], md=8),
        ], className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(id="account-chart"), md=12)),
        html.Hr(),
        dbc.Row(dbc.Col(html.Div(id="account-summary"), md=12)),
    ], fluid=True)


def _compare_tab():
    """Layout for Compare mode (two stocks side-by-side)."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Label("Stock A"),
                dbc.Input(id="compare-ticker-a", type="text", value=_default_ticker, debounce=True),
            ], md=2),
            dbc.Col([
                dbc.Label("Stock B"),
                dbc.Input(id="compare-ticker-b", type="text", value="", debounce=True, placeholder="e.g. MSFT"),
            ], md=2),
            dbc.Col([
                dbc.Label("Window"),
                dcc.Dropdown(id="compare-window", options=[
                    {"label": "3 months", "value": "3m"},
                    {"label": "6 months", "value": "6m"},
                    {"label": "1 year", "value": "1y"},
                    {"label": "2 years", "value": "2y"},
                ], value="1y", clearable=False),
            ], md=2),
            dbc.Col([
                dbc.Label("Normalize"),
                dcc.Dropdown(id="compare-normalize", options=[
                    {"label": "Percent change", "value": "pct"},
                    {"label": "Raw price", "value": "raw"},
                ], value="pct", clearable=False),
            ], md=2),
            dbc.Col([
                dbc.Label("Signal Scope"),
                dcc.Dropdown(id="compare-scope", options=_SCOPE_OPTIONS, value="daily", clearable=False),
            ], md=2),
        ], className="mb-3"),
        dbc.Row(dbc.Col(dcc.Graph(id="compare-chart"), md=12)),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.H5(id="compare-title-a", className="text-center"), html.Div(id="compare-signals-a")], md=6),
            dbc.Col([html.H5(id="compare-title-b", className="text-center"), html.Div(id="compare-signals-b")], md=6),
        ]),
    ], fluid=True)


app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Finsell Dashboard", className="text-center my-3"))),
    dbc.Tabs([
        dbc.Tab(_holdings_tab(), label="Holdings", tab_id="tab-holdings"),
        dbc.Tab(_research_tab(), label="Research", tab_id="tab-research"),
        dbc.Tab(_account_tab(), label="Account", tab_id="tab-account"),
        dbc.Tab(_compare_tab(), label="Compare", tab_id="tab-compare"),
    ], id="tabs", active_tab="tab-holdings"),
], fluid=True)


# ===========================================================================
# HOLDINGS CALLBACKS
# ===========================================================================


@callback(
    Output("holdings-data-store", "data"),
    Input("btn-refresh-holdings", "n_clicks"),
    Input("holdings-scope", "value"),
    Input("holdings-refresh-token", "data"),
    prevent_initial_call=False,
)
def compute_holdings_data(_, scope, __):
    """Compute signals for all holdings. Only runs on Refresh or initial load."""
    global _portfolio
    # Reload portfolio with same override path if provided during initialization
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", type=str, default=None)
    parsed_args = parser.parse_known_args()[0]
    
    reload_path = None
    if _args.portfolio:
        # Use the same override path that was passed when dashboard started
        reload_path = _args.portfolio.replace('\\', '/')
    elif hasattr(_portfolio, '_last_load_path') and _portfolio._last_load_path:
        # Fallback to last loaded path if available (from global args)
        reload_path = _portfolio._last_load_path
    
    if reload_path:
        _portfolio = Portfolio.load(reload_path)
    else:
        _portfolio = Portfolio.load()

    if not _portfolio.holdings:
        return {}

    store_data = {}
    for h in _portfolio.holdings:
        store_entry = {
            "buy": 0.0,
            "sell": 0.0,
            "hold": 0.0,
            "buy_more": 0.0,
            "patterns": {},
            "info": {"name": h.ticker},
            "shares": h.shares,
            "wac": h.wac,
            "is_watcher": h.is_watcher,
            "market_override": h.market_override,
        }

        res = _compute_scores(h.ticker, wac=h.wac or None,
                              sector_override=h.sector or None,
                              market_override=h.market_override,
                              scope=scope)
        if res is None:
            store_data[h.ticker] = store_entry
            continue

        decisions, patterns, info = res
        store_entry.update({
            "buy": decisions.buy,
            "sell": decisions.sell,
            "hold": decisions.hold,
            "buy_more": decisions.buy_more,
            "patterns": {k: v for k, v in patterns.items() if v is not None},
            "info": {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool, type(None)))},
        })
        store_data[h.ticker] = store_entry

        # Grab market pulse from first successful result
        if "_pulse" not in store_data:
            store_data["_pulse"] = {
                "us_stage": decisions.market_pulse.us.stage,
                "us_dist": decisions.market_pulse.us.distribution_count,
                "tsx_stage": decisions.market_pulse.tsx.stage,
                "tsx_dist": decisions.market_pulse.tsx.distribution_count,
                "scope": scope,
            }

    return store_data


@callback(
    Output("holdings-grid", "children"),
    Output("holdings-market-pulse", "children"),
    Output("alert-sell", "children"),
    Output("alert-buy-more", "children"),
    Output("alert-watcher-buy", "children"),
    Input("holdings-data-store", "data"),
    Input("filter-all", "n_clicks"),
    Input("filter-holdings", "n_clicks"),
    Input("filter-watchers", "n_clicks"),
    prevent_initial_call=False,
)
def render_holdings_grid(store_data, *_):
    """Render holdings grid from cached data. Fast — no fetching."""
    if not store_data:
        empty_msg = html.P("No holdings yet. Click '+ Add' to get started.", className="text-muted")
        return empty_msg, "", "—", "—", "—"

    # Determine filter
    trigger = ctx.triggered_id
    if trigger == "filter-holdings":
        show_filter = "holdings"
    elif trigger == "filter-watchers":
        show_filter = "watchers"
    else:
        show_filter = "all"

    # Filter stored data
    tickers = [k for k in store_data if not k.startswith("_")]
    if show_filter == "holdings":
        tickers = [t for t in tickers if not store_data[t].get("is_watcher", False)]
    elif show_filter == "watchers":
        tickers = [t for t in tickers if store_data[t].get("is_watcher", False)]

    if not tickers:
        empty_msg = html.P("No items match this filter.", className="text-muted")
        return empty_msg, "", "—", "—", "—"

    # Market pulse
    pulse_data = store_data.get("_pulse", {})
    pulse_colors = {"green": "success", "yellow": "warning", "red": "danger"}
    pulse_content = ""
    if pulse_data:
        pulse_content = html.Span([
            html.Small(f"Scope: {pulse_data.get('scope', 'daily').title()} ", className="me-2 text-muted"),
            dbc.Badge(f"US:{pulse_data.get('us_stage', '?').upper()}", color=pulse_colors.get(pulse_data.get("us_stage"), "secondary"), className="me-1"),
            dbc.Badge(f"TSX:{pulse_data.get('tsx_stage', '?').upper()}", color=pulse_colors.get(pulse_data.get("tsx_stage"), "secondary")),
        ])

    # Alerts
    sell_alerts = []
    buy_more_alerts = []
    watcher_buy_alerts = []
    for t in tickers:
        d = store_data[t]
        if d["sell"] >= 0.7:
            sell_alerts.append(f"{t} ({d['sell']:.2f})")
        if d["buy_more"] >= 0.7 and not d.get("is_watcher"):
            buy_more_alerts.append(f"{t} ({d['buy_more']:.2f})")
        if d["buy"] >= 0.7 and d.get("is_watcher"):
            watcher_buy_alerts.append(f"{t} ({d['buy']:.2f})")

    # Build grid
    header = dbc.Row([
        dbc.Col(html.Strong("Ticker"), width=1),
        dbc.Col(html.Strong("Type"), width=1),
        dbc.Col(html.Strong("Shares"), width=1),
        dbc.Col(html.Strong("WAC"), width=1),
        dbc.Col(html.Strong("Buy"), width=2),
        dbc.Col(html.Strong("Sell"), width=2),
        dbc.Col(html.Strong("Hold"), width=2),
        dbc.Col(html.Strong("Buy+"), width=1),
        dbc.Col(html.Strong(""), width=1),
    ], className="border-bottom pb-1 mb-1 small fw-bold")

    rows = [header]
    for t in tickers:
        d = store_data[t]
        buy_bar = signal_bar(d["buy"], color="green")
        sell_bar = signal_bar(d["sell"], color="red")
        hold_bar = signal_bar(d["hold"], color="blue")
        bm_bar = signal_bar(d["buy_more"], color="orange")

        shares = d.get("shares", 0)
        wac = d.get("wac", 0)
        is_watcher = d.get("is_watcher", False)

        row = dbc.Row([
            dbc.Col(html.A(t, href="#", id={"type": "holding-link", "ticker": t},
                           className="text-decoration-none fw-bold"), width=1),
            dbc.Col(html.Small("👀" if is_watcher else "💰"), width=1),
            dbc.Col(html.Small(f"{shares:.0f}" if shares else "—"), width=1),
            dbc.Col(html.Small(f"${wac:.2f}" if wac else "—"), width=1),
            dbc.Col(buy_bar, width=2),
            dbc.Col(sell_bar, width=2),
            dbc.Col(hold_bar, width=2),
            dbc.Col(bm_bar, width=1),
            dbc.Col(html.A("✏️", href="#", id={"type": "edit-btn", "ticker": t},
                           className="text-decoration-none", title="Edit / Delete"), width=1),
        ], className="border-bottom py-1 small align-items-center")
        rows.append(row)

    grid = html.Div(rows)
    sell_content = html.Ul([html.Li(s) for s in sell_alerts], className="mb-0 small") if sell_alerts else html.Small("None", className="text-muted")
    bm_content = html.Ul([html.Li(s) for s in buy_more_alerts], className="mb-0 small") if buy_more_alerts else html.Small("None", className="text-muted")
    wb_content = html.Ul([html.Li(s) for s in watcher_buy_alerts], className="mb-0 small") if watcher_buy_alerts else html.Small("None", className="text-muted")

    return grid, pulse_content, sell_content, bm_content, wb_content


@callback(
    Output("holdings-detail", "children"),
    Input({"type": "holding-link", "ticker": ALL}, "n_clicks"),
    State("holdings-data-store", "data"),
    prevent_initial_call=True,
)
def show_holding_detail(n_clicks_list, store_data):
    """Show detail for a clicked holding."""
    if not any(n_clicks_list):
        return no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update

    ticker = triggered["ticker"]
    if not store_data or ticker not in store_data:
        return html.P(f"No data for {ticker}")

    d = store_data[ticker]
    info = d.get("info", {})
    name = info.get("name", ticker)

    # Signal gauges (must wrap in dcc.Graph)
    gauges = dbc.Row([
        dbc.Col(_gauge_graph("BUY", figure=decision_gauge(d["buy"], "BUY", "green"), config={"displayModeBar": False}), md=3),
        dbc.Col(_gauge_graph("SELL", figure=decision_gauge(d["sell"], "SELL", "red"), config={"displayModeBar": False}), md=3),
        dbc.Col(_gauge_graph("HOLD", figure=decision_gauge(d["hold"], "HOLD", "blue"), config={"displayModeBar": False}), md=3),
        dbc.Col(_gauge_graph("BUY MORE", figure=decision_gauge(d["buy_more"], "BUY MORE", "green"), config={"displayModeBar": False}), md=3),
    ])

    # Pattern scores (must wrap in dcc.Graph)
    patterns = d.get("patterns", {})
    pattern_items = [
        dbc.Col(_gauge_graph(
            k.replace("_", " ").title(),
            figure=mini_gauge(v, k.replace("_", " ").title(),
                              "0_1" if k == "cup_handle" else "neg1_pos1"),
            config={"displayModeBar": False},
        ), md=2)
        for k, v in patterns.items()
    ]
    pattern_row = dbc.Row(pattern_items) if pattern_items else ""

    return html.Div([
        html.H5(f"{name} ({ticker})", className="mb-2"),
        html.Small(f"{info.get('sector', '')} | {info.get('industry', '')}",
                   className="text-muted d-block mb-2"),
        gauges,
        html.Hr(),
        pattern_row,
    ])


# --- Add/Edit modal ---
@callback(
    Output("modal-holding", "is_open"),
    Output("modal-ticker", "value"),
    Output("modal-shares", "value"),
    Output("modal-wac", "value"),
    Output("modal-market", "value"),
    Output("modal-sector", "value"),
    Output("modal-notes", "value"),
    Output("modal-status", "children"),
    Output("holdings-refresh-token", "data"),
    Input("btn-add-holding", "n_clicks"),
    Input({"type": "edit-btn", "ticker": ALL}, "n_clicks"),
    Input("modal-save", "n_clicks"),
    Input("modal-delete", "n_clicks"),
    Input("modal-reload-yf", "n_clicks"),
    Input("modal-cancel", "n_clicks"),
    State("modal-ticker", "value"),
    State("modal-shares", "value"),
    State("modal-wac", "value"),
    State("modal-market", "value"),
    State("modal-sector", "value"),
    State("modal-notes", "value"),
    State("holdings-refresh-token", "data"),
    prevent_initial_call=True,
)
def handle_modal(add_click, edit_clicks, save_click, delete_click, reload_click, cancel_click,
                 ticker, shares, wac, market, sector, notes, refresh_token):
    trigger = ctx.triggered_id

    def next_refresh_token() -> str:
        return datetime.now().isoformat()

    # Guard: ignore if no actual click happened (pattern-match init fires)
    if not trigger:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
    if isinstance(trigger, dict) and trigger.get("type") == "edit-btn":
        # Check that at least one edit button was actually clicked
        if not edit_clicks or not any(edit_clicks):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
        t = trigger["ticker"]
        h = _portfolio.get(t)
        if h:
            return (True, h.ticker, h.shares, h.wac,
                    h.market_override, h.sector, h.notes,
                    f"Cache: {CACHE_DIR / t.upper()}", refresh_token)
        return True, t, 0, 0, "auto", "", "", f"Cache: {CACHE_DIR / t.upper()}", refresh_token

    if trigger == "btn-add-holding":
        return True, "", 0, 0, "auto", "", "", f"Cache root: {Path(CACHE_DIR)}", refresh_token

    if trigger == "modal-cancel":
        return False, "", 0, 0, "auto", "", "", "", refresh_token

    if trigger == "modal-reload-yf" and ticker:
        normalized_ticker = ticker.upper().strip()
        try:
            invalidate_cache(normalized_ticker)
            data = fetch_all(normalized_ticker, sector_override=sector or None, force=True)
            info = data.get("info", {})
            status = (
                f"Reloaded {normalized_ticker} from yfinance. "
                f"Sector: {info.get('sector', 'Unknown')} | Industry: {info.get('industry', 'Unknown')} | "
                f"Cache: {CACHE_DIR / normalized_ticker}"
            )
            return True, normalized_ticker, shares, wac, market or "auto", sector or "", notes or "", status, next_refresh_token()
        except Exception as exc:
            status = f"Reload failed for {normalized_ticker}: {exc}"
            return True, normalized_ticker, shares, wac, market or "auto", sector or "", notes or "", status, refresh_token

    if trigger == "modal-save" and ticker:
        normalized_ticker = ticker.upper().strip()
        _portfolio.add(
            ticker=normalized_ticker,
            shares=float(shares or 0),
            wac=float(wac or 0),
            sector=sector or "",
            market_override=market or "auto",
            notes=notes or "",
        )
        _portfolio.save()
        status = "Saved holding."
        try:
            data = fetch_all(normalized_ticker, sector_override=sector or None, force=True)
            info = data.get("info", {})
            status = (
                f"Saved and refreshed {normalized_ticker}. "
                f"Sector: {info.get('sector', 'Unknown')} | Industry: {info.get('industry', 'Unknown')} | "
                f"Cache: {CACHE_DIR / normalized_ticker}"
            )
        except Exception as exc:
            status = f"Saved {normalized_ticker}, but refresh failed: {exc}"
        return False, "", 0, 0, "auto", "", "", status, next_refresh_token()

    if trigger == "modal-delete" and ticker:
        normalized_ticker = ticker.upper().strip()
        _portfolio.remove(normalized_ticker)
        _portfolio.save()
        return False, "", 0, 0, "auto", "", "", f"Deleted {normalized_ticker}.", next_refresh_token()

    return False, "", 0, 0, "auto", "", "", "", refresh_token


# ===========================================================================
# RESEARCH CALLBACKS
# ===========================================================================


@callback(
    Output("research-chart", "figure"),
    Output("research-info-bar", "children"),
    Output("research-gauge-score", "figure"),
    Output("research-gauge-strength", "figure"),
    Output("research-gauge-momentum", "figure"),
    Output("research-mg-cup-handle", "figure"),
    Output("research-mg-distribution", "figure"),
    Output("research-mg-climax-top", "figure"),
    Output("research-mg-broken-trend", "figure"),
    Output("research-mg-overbought", "figure"),
    Output("research-mg-double-bottom", "figure"),
    Output("research-mg-three-weeks-tight", "figure"),
    Output("research-mg-ascending-base", "figure"),
    Output("research-mg-flat-base", "figure"),
    Output("research-mg-ma-support", "figure"),
    Output("research-mg-golden-cross", "figure"),
    Output("research-mg-purchase-timing", "figure"),
    Output("research-pulse-indicator", "children"),
    Output("research-dg-buy", "figure"),
    Output("research-dg-sell", "figure"),
    Output("research-dg-hold", "figure"),
    Output("research-dg-buy-more", "figure"),
    Output("research-details", "children"),
    Output("research-export-data", "data"),
    Input("research-ticker", "value"),
    Input("research-window", "value"),
    Input("research-overlays", "value"),
    Input("research-pulse-override", "value"),
    Input("research-scope", "value"),
    Input("research-wac", "value"),
)
def update_research(ticker, window, overlays, market_override, scope, wac):
    """Full research view for a single stock."""
    empty = go.Figure()
    empty_mini = mini_gauge(0, "—")
    empty_decision = decision_gauge(0, "—")
    n_empty = (empty, "") + (empty,) * 2 + (empty,) + (empty_mini,) * 12 + ("",) + (empty_decision,) * 4 + ("",)

    if not ticker or len(ticker) < 1:
        return n_empty

    ticker = ticker.upper().strip()
    _portfolio.touch_lru(ticker)

    try:
        data = fetch_all(ticker, sector_override=_default_sector)
    except Exception:
        return n_empty

    chart_df = _select_scope_df(data.get("monthly"), data.get("daily"), scope)
    if chart_df is None or len(chart_df) == 0:
        return n_empty

    info = data["info"]
    wac_val = float(wac) if wac else None

    # Chart
    days = _WINDOW_DAYS.get(window, 365)
    cutoff = datetime.now() - timedelta(days=days)
    df = chart_df[chart_df.index >= cutoff].copy()
    if df.empty:
        return n_empty

    show_volume = "volume" in overlays
    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)
    else:
        fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines",
                             name="Close", line=dict(color="#1F77B4", width=2)), row=1, col=1)

    if "high" in overlays and "High" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["High"], mode="lines",
                                 name="High", line=dict(color="#2CA02C", width=1, dash="dot"), opacity=0.6), row=1, col=1)
    if "low" in overlays and "Low" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Low"], mode="lines",
                                 name="Low", line=dict(color="#D62728", width=1, dash="dot"), opacity=0.6), row=1, col=1)

    if "drip" in overlays:
        divs = data["dividends"]
        if divs is not None and len(divs) > 0:
            divs = divs.copy()
            divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
            divs_in_window = divs[divs.index >= cutoff]
            if len(divs_in_window) > 0:
                shares = 1.0
                drip_values = []
                for date, close in df["Close"].items():
                    last_date = drip_values[-1][0] if drip_values else cutoff
                    divs_on_date = divs_in_window[(divs_in_window.index <= date) & (divs_in_window.index > last_date)]
                    for _, div_row in divs_on_date.iterrows():
                        div_amount = div_row.iloc[0]
                        if close > 0:
                            shares += (div_amount * shares) / close
                    drip_values.append((date, shares * close))
                if drip_values:
                    drip_dates, drip_prices = zip(*drip_values)
                    fig.add_trace(go.Scatter(x=list(drip_dates), y=list(drip_prices), mode="lines",
                                            name="DRIP", line=dict(color="#FF7F0E", width=2, dash="dash")), row=1, col=1)

    if wac_val:
        fig.add_hline(y=wac_val, line_dash="dash", line_color="orange",
                      annotation_text=f"WAC ${wac_val:.2f}", row=1, col=1)

    if "dividends" in overlays:
        divs = data["dividends"]
        if divs is not None and len(divs) > 0:
            divs = divs.copy()
            divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
            divs = divs[divs.index >= cutoff]
            if len(divs) > 0:
                div_prices = []
                for d in divs.index:
                    mask = df.index <= d
                    div_prices.append(df.loc[mask, "Close"].iloc[-1] if mask.any() else df["Close"].iloc[0])
                fig.add_trace(go.Scatter(x=divs.index, y=div_prices, mode="markers", name="Dividend",
                                         marker=dict(symbol="diamond", size=10, color="green"),
                                         text=[f"${v:.3f}" for v in divs.iloc[:, 0]],
                                         hovertemplate="%{x}<br>Div: %{text}<extra></extra>"), row=1, col=1)

    if "splits" in overlays:
        splits = data["splits"]
        if splits is not None and len(splits) > 0:
            splits = splits.copy()
            splits.index = splits.index.tz_localize(None) if splits.index.tz else splits.index
            splits = splits[splits.index >= cutoff]
            if len(splits) > 0:
                split_prices = []
                for d in splits.index:
                    mask = df.index <= d
                    split_prices.append(df.loc[mask, "Close"].iloc[-1] if mask.any() else df["Close"].iloc[0])
                fig.add_trace(go.Scatter(x=splits.index, y=split_prices, mode="markers", name="Split",
                                         marker=dict(symbol="star", size=14, color="purple"),
                                         text=[f"{v:.0f}:1" for v in splits.iloc[:, 0]],
                                         hovertemplate="%{x}<br>Split: %{text}<extra></extra>"), row=1, col=1)

    if show_volume and "Volume" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                             marker_color="rgba(100,100,100,0.3)"), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    name = info.get("name", ticker)
    fig.update_layout(title=f"{name} ({ticker}) — {window}",
                      xaxis_rangeslider_visible=False,
                      height=550 if show_volume else 450,
                      margin=dict(t=40, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(rangeselector=dict(buttons=[
                          dict(count=1, label="1M", step="month", stepmode="backward"),
                          dict(count=3, label="3M", step="month", stepmode="backward"),
                          dict(count=6, label="6M", step="month", stepmode="backward"),
                          dict(count=1, label="1Y", step="year", stepmode="backward"),
                          dict(step="all", label="All"),
                      ])))
    fig.update_yaxes(title_text="Price", tickformat=".2f", row=1, col=1)

    # Info bar
    sector = _default_sector or info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    pe = info.get("trailingPE")
    pe_str = f"P/E {pe:.1f}" if pe else "P/E n/a"
    beta = info.get("beta")
    beta_str = f"β {beta:.2f}" if beta else "β n/a"
    info_text = f"{name} | {industry} | {sector} | {pe_str} | {beta_str} | Scope {scope.title()}"
    if wac_val:
        info_text += f" | WAC ${wac_val:.2f}"

    # Scores
    res = _compute_scores(ticker, wac=wac_val, sector_override=_default_sector,
                          market_override=market_override, scope=scope)
    if res is None:
        return n_empty

    decisions, patterns, _ = res

    valid_patterns = _sanitize_pattern_scores(patterns)
    all_scores = list(valid_patterns.values())
    momentum = _pattern_momentum(valid_patterns)
    composite = valid_patterns.get("cup_handle")
    composite = composite_rating_score(ticker) if composite is not None else None

    strength = (momentum + 1) / 2

    # Pulse display
    pulse = decisions.market_pulse
    pulse_colors = {"green": "success", "yellow": "warning", "red": "danger"}
    us_badge = dbc.Badge(f"US:{pulse.us.stage.upper()}", color=pulse_colors.get(pulse.us.stage, "secondary"), className="me-1")
    tsx_badge = dbc.Badge(f"TSX:{pulse.tsx.stage.upper()}", color=pulse_colors.get(pulse.tsx.stage, "secondary"), className="me-1")
    eff_badge = dbc.Badge(f"M={pulse.effective_multiplier}", color=pulse_colors.get(pulse.effective_stage, "secondary"), className="fs-6 p-2 mb-1")
    us_detail = html.Small(f"US: {pulse.us.distribution_count}d ({pulse.us.stalling_count}s)" + (" FTD✓" if pulse.us.ftd_detected else ""), className="text-muted d-block")
    tsx_detail = html.Small(f"TSX: {pulse.tsx.distribution_count}d ({pulse.tsx.stalling_count}s)" + (" FTD✓" if pulse.tsx.ftd_detected else ""), className="text-muted d-block")
    pulse_content = html.Div([eff_badge, html.Br(), us_badge, tsx_badge, us_detail, tsx_detail])

    # Details
    detail_items = []
    for signal_name, detail_dict in decisions.details.items():
        if isinstance(detail_dict, dict):
            for k, v in detail_dict.items():
                if k != "reason":
                    detail_items.append(f"{signal_name}.{k}: {v}")
                else:
                    detail_items.append(f"{signal_name}: {v}")
    details_content = html.Ul([html.Li(item) for item in detail_items[:8]], className="list-unstyled mb-0")

    export_payload = {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(),
        "gauges": _build_research_gauge_export_payload(ticker, patterns, decisions),
    }

    return (
        fig,
        html.Small(info_text, className="text-muted"),
        gauge_0_to_1(composite, "Composite Score"),
        gauge_0_to_1(strength, "Trend Strength"),
        gauge_neg1_to_pos1(momentum, "Momentum"),
        mini_gauge(patterns.get("cup_handle"), "Cup & Handle", "0_1"),
        mini_gauge(patterns.get("distribution"), "Distribution"),
        mini_gauge(patterns.get("climax_top"), "Climax Top"),
        mini_gauge(patterns.get("broken_trend"), "Broken Trend"),
        mini_gauge(patterns.get("overbought"), "Overbought"),
        mini_gauge(patterns.get("double_bottom"), "Double Bottom"),
        mini_gauge(patterns.get("three_wks_tight"), "3 Wks Tight"),
        mini_gauge(patterns.get("ascending_base"), "Ascending Base"),
        mini_gauge(patterns.get("flat_base"), "Flat Base"),
        mini_gauge(patterns.get("ma_support"), "MA Support"),
        mini_gauge(patterns.get("golden_cross"), "Golden Cross"),
        mini_gauge(patterns.get("purchase_timing"), "Purchase Timing"),
        pulse_content,
        decision_gauge(decisions.buy, "BUY", "green"),
        decision_gauge(decisions.sell, "SELL", "red"),
        decision_gauge(decisions.hold, "HOLD", "blue"),
        decision_gauge(decisions.buy_more, "BUY MORE", "green"),
        details_content,
        export_payload,
    )


@callback(
    Output("research-export-download", "data"),
    Input("research-export-btn", "n_clicks"),
    State("research-ticker", "value"),
    State("research-export-data", "data"),
    prevent_initial_call=True,
)
def download_research_gauge_export(n_clicks, ticker, export_data):
    """Download the current research gauges as JSON for AI-friendly inspection."""
    if not n_clicks or not export_data:
        raise PreventUpdate

    safe_ticker = (ticker or "stock").upper().replace(" ", "-").strip()
    return dcc.send_string(
        json.dumps(export_data, indent=2),
        filename=f"{safe_ticker}-gauge-parameters.json",
        mimetype="application/json",
    )


# ===========================================================================
# ACCOUNT CALLBACKS
# ===========================================================================


@callback(
    Output("account-chart", "figure"),
    Output("account-summary", "children"),
    Input("account-tickers", "value"),
    Input("account-window", "value"),
    Input("account-scope", "value"),
)
def update_account(selected_tickers, window, scope):
    """Plot percentage-change performance for the selected account tickers."""
    empty = go.Figure()
    if not selected_tickers:
        return empty, html.Small("Select one or more tickers to plot account performance.", className="text-muted")

    tickers = [ticker.upper().strip() for ticker in selected_tickers if ticker]
    if not tickers:
        return empty, html.Small("No tickers selected.", className="text-muted")

    days = _WINDOW_DAYS.get(window, 365)
    cutoff = datetime.now() - timedelta(days=days)

    fig = go.Figure()
    summary_items = []
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F"]

    for index, ticker in enumerate(tickers):
        try:
            data = fetch_all(ticker)
        except Exception:
            continue

        df_full = _select_scope_df(data.get("monthly"), data.get("daily"), scope)
        if df_full is None or df_full.empty:
            continue

        close = df_full["Close"].dropna()
        close = close[close.index >= cutoff]
        if close.empty or len(close) < 2:
            continue

        pct = ((close / close.iloc[0]) - 1) * 100
        color = colors[index % len(colors)]
        fig.add_trace(go.Scatter(x=pct.index, y=pct.values, name=ticker, line=dict(color=color, width=2)))
        summary_items.append(html.Li(f"{ticker}: {pct.iloc[-1]:+.2f}%", className="small"))

    if not fig.data:
        return empty, html.Small("No chart data available for the selected tickers.", className="text-muted")

    fig.update_layout(
        title=f"Account performance — {window} ({scope.title()})",
        height=500,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="% Change", ticksuffix="%"),
    )

    return fig, html.Ul(summary_items, className="list-unstyled mb-0")


# ===========================================================================
# COMPARE CALLBACKS
# ===========================================================================


@callback(
    Output("compare-chart", "figure"),
    Output("compare-title-a", "children"),
    Output("compare-signals-a", "children"),
    Output("compare-title-b", "children"),
    Output("compare-signals-b", "children"),
    Input("compare-ticker-a", "value"),
    Input("compare-ticker-b", "value"),
    Input("compare-window", "value"),
    Input("compare-normalize", "value"),
    Input("compare-scope", "value"),
)
def update_compare(ticker_a, ticker_b, window, normalize, scope):
    """Compare two stocks side by side."""
    empty = go.Figure()
    if not ticker_a or not ticker_b:
        return empty, "", "", "", ""

    ticker_a = ticker_a.upper().strip()
    ticker_b = ticker_b.upper().strip()
    _portfolio.touch_lru(ticker_a)
    _portfolio.touch_lru(ticker_b)

    days = _WINDOW_DAYS.get(window, 365)
    cutoff = datetime.now() - timedelta(days=days)

    try:
        data_a = fetch_all(ticker_a)
        data_b = fetch_all(ticker_b)
    except Exception:
        return empty, ticker_a, "Error loading data", ticker_b, "Error loading data"

    df_a_full = _select_scope_df(data_a.get("monthly"), data_a.get("daily"), scope)
    df_b_full = _select_scope_df(data_b.get("monthly"), data_b.get("daily"), scope)
    if df_a_full is None or df_b_full is None or df_a_full.empty or df_b_full.empty:
        return empty, ticker_a, "No data", ticker_b, "No data"

    df_a = df_a_full[df_a_full.index >= cutoff]["Close"].dropna()
    df_b = df_b_full[df_b_full.index >= cutoff]["Close"].dropna()

    fig = go.Figure()

    if normalize == "pct":
        # Normalize to percentage change from start
        if len(df_a) > 0 and len(df_b) > 0:
            norm_a = ((df_a / df_a.iloc[0]) - 1) * 100
            norm_b = ((df_b / df_b.iloc[0]) - 1) * 100
            fig.add_trace(go.Scatter(x=norm_a.index, y=norm_a.values, name=ticker_a,
                                     line=dict(color="#1F77B4", width=2)))
            fig.add_trace(go.Scatter(x=norm_b.index, y=norm_b.values, name=ticker_b,
                                     line=dict(color="#D62728", width=2)))
            fig.update_yaxes(title_text="% Change", ticksuffix="%")
    else:
        fig.add_trace(go.Scatter(x=df_a.index, y=df_a.values, name=ticker_a,
                                 line=dict(color="#1F77B4", width=2)))
        fig.add_trace(go.Scatter(x=df_b.index, y=df_b.values, name=ticker_b,
                                 line=dict(color="#D62728", width=2), yaxis="y2"))
        fig.update_layout(yaxis2=dict(title=ticker_b, overlaying="y", side="right"))
        fig.update_yaxes(title_text=ticker_a)

    fig.update_layout(
        title=f"{ticker_a} vs {ticker_b} — {window} ({scope.title()})",
        height=450, margin=dict(t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Signals for each
    def _signal_summary(ticker):
        res = _compute_scores(ticker, scope=scope)
        if res is None:
            return html.P("No data", className="text-muted")
        decisions, patterns, info = res
        return html.Div([
            html.Small(f"{info.get('sector', '')} | {info.get('industry', '')}", className="text-muted d-block mb-2"),
            dbc.Row([
                dbc.Col([html.Small("Buy"), signal_bar(decisions.buy, color="green")], md=3),
                dbc.Col([html.Small("Sell"), signal_bar(decisions.sell, color="red")], md=3),
                dbc.Col([html.Small("Hold"), signal_bar(decisions.hold, color="blue")], md=3),
                dbc.Col([html.Small("Buy+"), signal_bar(decisions.buy_more, color="orange")], md=3),
            ]),
            html.Hr(),
            html.Small("Key patterns:", className="fw-bold"),
            html.Ul([
                html.Li(f"{k.replace('_', ' ').title()}: {v:.3f}", className="small")
                for k, v in sorted(patterns.items(), key=lambda x: abs(x[1] or 0), reverse=True)[:5]
                if v is not None
            ], className="list-unstyled mb-0"),
        ])

    signals_a = _signal_summary(ticker_a)
    signals_b = _signal_summary(ticker_b)

    return fig, ticker_a, signals_a, ticker_b, signals_b


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=_port)