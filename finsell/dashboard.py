"""Finsell visual dashboard — Dash + Plotly."""

from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from finsell.data_cache import fetch_all, fetch_combined


def gauge_0_to_1(value, title, thresholds=None):
    """Gauge from 0 → 1 (e.g. score, probability, strength)."""
    thresholds = thresholds or {"green": 0.7, "yellow": 0.4}
    value = round(float(value), 4) if value is not None else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            number={"valueformat": ".4f"},
            domain={"x": [0.1, 0.9], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.25, 0.5, 0.75, 1]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, thresholds["yellow"]], "color": "#ffcccc"},
                    {"range": [thresholds["yellow"], thresholds["green"]], "color": "#fff3cd"},
                    {"range": [thresholds["green"], 1], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=60, b=20, l=40, r=40))
    return fig


def gauge_neg1_to_pos1(value, title):
    """Gauge from -1 → +1 with 0 centered (e.g. sentiment, momentum direction)."""
    value = round(float(value), 4) if value is not None else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            number={"valueformat": ".4f"},
            domain={"x": [0.1, 0.9], "y": [0, 1]},
            gauge={
                "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1]},
                "bar": {"color": "#444"},
                "steps": [
                    {"range": [-1, -0.5], "color": "#d62728"},
                    {"range": [-0.5, 0], "color": "#ffcccc"},
                    {"range": [0, 0.5], "color": "#d4edda"},
                    {"range": [0.5, 1], "color": "#2ca02c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(height=250, margin=dict(t=60, b=20, l=40, r=40))
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

import argparse

_parser = argparse.ArgumentParser(description="Finsell visual dashboard")
_parser.add_argument("-t", "--ticker", default="ENB.TO", help="Stock ticker symbol (e.g. NVDA)")
_parser.add_argument("-i", "--industry", default=None, help="Industry override (if yfinance lacks it)")
_parser.add_argument("-s", "--sector", default=None, help="Sector override (maps to ETF for comparison)")
_parser.add_argument("-w", "--wac", type=float, default=None, help="Weighted average cost basis")
_parser.add_argument("-p", "--port", type=int, default=8050, help="Dashboard port (default 8050)")
_args = _parser.parse_args()

_default_ticker = _args.ticker.upper()
_default_industry = _args.industry
_default_sector = _args.sector
_default_wac = _args.wac
_port = _args.port

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Finsell Dashboard",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(html.H2("Finsell Dashboard", className="text-center my-3")),
        ),
        # --- Selectors row ---
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Ticker"),
                        dbc.Input(
                            id="ticker-input",
                            type="text",
                            value=_default_ticker,
                            debounce=True,
                            placeholder="e.g. ENB.TO",
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Time window"),
                        dcc.Dropdown(
                            id="window-dropdown",
                            options=[
                                {"label": "3 months", "value": "3m"},
                                {"label": "6 months", "value": "6m"},
                                {"label": "1 year", "value": "1y"},
                                {"label": "2 years", "value": "2y"},
                                {"label": "5 years", "value": "5y"},
                                {"label": "10 years", "value": "10y"},
                            ],
                            value="1y",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("Overlays"),
                        dcc.Checklist(
                            id="overlay-toggles",
                            options=[
                                {"label": " High", "value": "high"},
                                {"label": " Low", "value": "low"},
                                {"label": " Volume", "value": "volume"},
                                {"label": " Dividends", "value": "dividends"},
                                {"label": " DRIP", "value": "drip"},
                                {"label": " Splits", "value": "splits"},
                            ],
                            value=["volume"],
                            inline=True,
                            inputStyle={"marginRight": "4px", "marginLeft": "12px"},
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Label("Market Pulse"),
                        dcc.Dropdown(
                            id="market-pulse-override",
                            options=[
                                {"label": "Auto (US/.TO)", "value": "auto"},
                                {"label": "US only", "value": "us"},
                                {"label": "TSX only", "value": "tsx"},
                                {"label": "Both (conservative)", "value": "dual"},
                            ],
                            value="auto",
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
            ],
            className="mb-3",
        ),
        # --- Info bar ---
        dbc.Row(dbc.Col(html.Div(id="info-bar"), md=12), className="mb-2"),
        # --- Main chart ---
        dbc.Row(dbc.Col(dcc.Graph(id="price-chart"), md=12)),
        # --- Gauges ---
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="gauge-score"), md=4),
                dbc.Col(dcc.Graph(id="gauge-strength"), md=4),
                dbc.Col(dcc.Graph(id="gauge-momentum"), md=4),
            ],
            className="mt-3",
        ),
        # --- Signal mini-gauges ---
        html.Hr(),
        html.H5("Pattern & Signal Scores", className="text-center text-muted mb-2"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="mg-cup-handle"), md=2),
                dbc.Col(dcc.Graph(id="mg-distribution"), md=2),
                dbc.Col(dcc.Graph(id="mg-climax-top"), md=2),
                dbc.Col(dcc.Graph(id="mg-broken-trend"), md=2),
                dbc.Col(dcc.Graph(id="mg-overbought"), md=2),
                dbc.Col(dcc.Graph(id="mg-double-bottom"), md=2),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="mg-three-weeks-tight"), md=2),
                dbc.Col(dcc.Graph(id="mg-ascending-base"), md=2),
                dbc.Col(dcc.Graph(id="mg-flat-base"), md=2),
                dbc.Col(dcc.Graph(id="mg-ma-support"), md=2),
                dbc.Col(dcc.Graph(id="mg-golden-cross"), md=2),
                dbc.Col(dcc.Graph(id="mg-purchase-timing"), md=2),
            ],
            className="mb-3",
        ),
        # --- Decision Signals ---
        html.Hr(),
        html.H5("Decision Signals", className="text-center mb-2"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Market Pulse", className="text-center"),
                            html.Div(id="market-pulse-indicator", className="text-center"),
                        ])
                    ]),
                    md=2,
                ),
                dbc.Col(dcc.Graph(id="dg-buy"), md=2),
                dbc.Col(dcc.Graph(id="dg-sell"), md=2),
                dbc.Col(dcc.Graph(id="dg-hold"), md=2),
                dbc.Col(dcc.Graph(id="dg-buy-more"), md=2),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Details", className="text-center"),
                            html.Div(id="decision-details", style={"fontSize": "11px"}),
                        ])
                    ]),
                    md=2,
                ),
            ],
            className="mb-3",
        ),
    ],
    fluid=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mini_gauge(value, title, range_type="neg1_pos1"):
    """Small gauge for signal grid. range_type: 'neg1_pos1' or '0_1'."""
    if value is None:
        value = 0
    value = round(float(value), 4)
    if range_type == "0_1":
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"font": {"size": 14}, "valueformat": ".4f"},
                title={"text": title, "font": {"size": 11}},
                gauge={
                    "axis": {"range": [0, 1], "tickvals": [0, 0.5, 1], "tickfont": {"size": 8}},
                    "bar": {"color": "#1f77b4", "thickness": 0.6},
                    "steps": [
                        {"range": [0, 0.4], "color": "#ffcccc"},
                        {"range": [0.4, 0.7], "color": "#fff3cd"},
                        {"range": [0.7, 1], "color": "#d4edda"},
                    ],
                },
            )
        )
    else:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                number={"font": {"size": 14}, "valueformat": ".4f"},
                title={"text": title, "font": {"size": 11}},
                gauge={
                    "axis": {"range": [-1, 1], "tickvals": [-1, 0, 1], "tickfont": {"size": 8}},
                    "bar": {"color": "#444", "thickness": 0.6},
                    "steps": [
                        {"range": [-1, -0.3], "color": "#d62728"},
                        {"range": [-0.3, 0.3], "color": "#f0f0f0"},
                        {"range": [0.3, 1], "color": "#2ca02c"},
                    ],
                },
            )
        )
    fig.update_layout(height=160, margin=dict(t=40, b=5, l=15, r=15))
    return fig


def decision_gauge(value, title, color_scheme="green"):
    """Gauge for decision signals (0→1). Color schemes: green, red, blue."""
    if value is None:
        value = 0
    value = round(float(value), 4)
    colors = {
        "green": {"bar": "#2ca02c", "steps": [
            {"range": [0, 0.3], "color": "#f0f0f0"},
            {"range": [0.3, 0.7], "color": "#d4edda"},
            {"range": [0.7, 1], "color": "#2ca02c"},
        ]},
        "red": {"bar": "#d62728", "steps": [
            {"range": [0, 0.3], "color": "#f0f0f0"},
            {"range": [0.3, 0.7], "color": "#fff3cd"},
            {"range": [0.7, 1], "color": "#d62728"},
        ]},
        "blue": {"bar": "#1f77b4", "steps": [
            {"range": [0, 0.3], "color": "#f0f0f0"},
            {"range": [0.3, 0.7], "color": "#cce5ff"},
            {"range": [0.7, 1], "color": "#1f77b4"},
        ]},
    }
    scheme = colors.get(color_scheme, colors["green"])
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 16}, "valueformat": ".4f"},
            title={"text": title, "font": {"size": 12}},
            gauge={
                "axis": {"range": [0, 1], "tickvals": [0, 0.5, 1], "tickfont": {"size": 9}},
                "bar": {"color": scheme["bar"], "thickness": 0.7},
                "steps": scheme["steps"],
            },
        )
    )
    fig.update_layout(height=180, margin=dict(t=45, b=10, l=15, r=15))
    return fig


_WINDOW_DAYS = {
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("price-chart", "figure"),
    Output("info-bar", "children"),
    Output("gauge-score", "figure"),
    Output("gauge-strength", "figure"),
    Output("gauge-momentum", "figure"),
    Output("mg-cup-handle", "figure"),
    Output("mg-distribution", "figure"),
    Output("mg-climax-top", "figure"),
    Output("mg-broken-trend", "figure"),
    Output("mg-overbought", "figure"),
    Output("mg-double-bottom", "figure"),
    Output("mg-three-weeks-tight", "figure"),
    Output("mg-ascending-base", "figure"),
    Output("mg-flat-base", "figure"),
    Output("mg-ma-support", "figure"),
    Output("mg-golden-cross", "figure"),
    Output("mg-purchase-timing", "figure"),
    Output("market-pulse-indicator", "children"),
    Output("dg-buy", "figure"),
    Output("dg-sell", "figure"),
    Output("dg-hold", "figure"),
    Output("dg-buy-more", "figure"),
    Output("decision-details", "children"),
    Input("ticker-input", "value"),
    Input("window-dropdown", "value"),
    Input("overlay-toggles", "value"),
    Input("market-pulse-override", "value"),
)
def update_dashboard(ticker, window, overlays, market_override):
    """Update all visuals when selectors change."""
    empty = go.Figure()
    empty_mini = mini_gauge(0, "—")
    empty_decision = decision_gauge(0, "—")

    if not ticker or len(ticker) < 1:
        return (empty, "") + (empty_mini,) * 15 + ("",) + (empty_decision,) * 4 + ("",)

    ticker = ticker.upper().strip()
    data = fetch_all(ticker, sector_override=_default_sector)
    info = data["info"]
    combined = fetch_combined(ticker)

    if combined.empty:
        return (empty,) + (f"No data found for {ticker}",) + (empty_mini,) * 15 + ("",) + (empty_decision,) * 4 + ("",)

    # Filter to time window
    days = _WINDOW_DAYS.get(window, 365)
    cutoff = datetime.now() - timedelta(days=days)
    df = combined[combined.index >= cutoff].copy()

    if df.empty:
        return (empty,) + (f"No data in selected window for {ticker}",) + (empty_mini,) * 15 + ("",) + (empty_decision,) * 4 + ("",)

    # Build figure with optional volume subplot
    show_volume = "volume" in overlays
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )
    else:
        fig = make_subplots(rows=1, cols=1)

    # Close price line
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["Close"], mode="lines",
            name="Close", line=dict(color="#1f77b4", width=2),
        ),
        row=1, col=1,
    )

    # High overlay
    if "high" in overlays and "High" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["High"], mode="lines",
                name="High", line=dict(color="#2ca02c", width=1, dash="dot"),
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # Low overlay
    if "low" in overlays and "Low" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Low"], mode="lines",
                name="Low", line=dict(color="#d62728", width=1, dash="dot"),
                opacity=0.6,
            ),
            row=1, col=1,
        )

    # DRIP total-return line
    if "drip" in overlays:
        divs = data["dividends"]
        if divs is not None and len(divs) > 0:
            divs = divs.copy()
            divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
            divs_in_window = divs[divs.index >= cutoff]
            if len(divs_in_window) > 0:
                # Calculate cumulative shares with dividend reinvestment
                shares = 1.0
                drip_values = []
                for date, close in df["Close"].items():
                    # Reinvest any dividends on or before this date that haven't been processed
                    last_date = drip_values[-1][0] if drip_values else cutoff
                    divs_on_date = divs_in_window[
                        (divs_in_window.index <= date)
                        & (divs_in_window.index > last_date)
                    ]
                    for _, div_row in divs_on_date.iterrows():
                        div_amount = div_row.iloc[0]
                        if close > 0:
                            shares += (div_amount * shares) / close
                    drip_values.append((date, shares * close))

                if drip_values:
                    drip_dates, drip_prices = zip(*drip_values)
                    fig.add_trace(
                        go.Scatter(
                            x=list(drip_dates), y=list(drip_prices), mode="lines",
                            name="Total Return (DRIP)",
                            line=dict(color="#ff7f0e", width=2, dash="dash"),
                        ),
                        row=1, col=1,
                    )

    # WAC line
    if _default_wac is not None:
        fig.add_hline(
            y=_default_wac, line_dash="dash", line_color="orange",
            annotation_text=f"WAC ${_default_wac:.2f}",
            row=1, col=1,
        )

    # Dividends markers
    if "dividends" in overlays:
        divs = data["dividends"]
        if divs is not None and len(divs) > 0:
            divs = divs.copy()
            divs.index = divs.index.tz_localize(None) if divs.index.tz else divs.index
            divs = divs[divs.index >= cutoff]
            if len(divs) > 0:
                # Map dividend dates to closest price for y-position
                div_prices = []
                for d in divs.index:
                    mask = df.index <= d
                    if mask.any():
                        div_prices.append(df.loc[mask, "Close"].iloc[-1])
                    else:
                        div_prices.append(df["Close"].iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=divs.index, y=div_prices, mode="markers",
                        name="Dividend",
                        marker=dict(symbol="diamond", size=10, color="green"),
                        text=[f"${v:.3f}" for v in divs.iloc[:, 0]],
                        hovertemplate="%{x}<br>Dividend: %{text}<extra></extra>",
                    ),
                    row=1, col=1,
                )

    # Splits markers
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
                    if mask.any():
                        split_prices.append(df.loc[mask, "Close"].iloc[-1])
                    else:
                        split_prices.append(df["Close"].iloc[0])
                fig.add_trace(
                    go.Scatter(
                        x=splits.index, y=split_prices, mode="markers",
                        name="Split",
                        marker=dict(symbol="star", size=14, color="purple"),
                        text=[f"{v:.0f}:1" for v in splits.iloc[:, 0]],
                        hovertemplate="%{x}<br>Split: %{text}<extra></extra>",
                    ),
                    row=1, col=1,
                )

    # Volume bars
    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df.index, y=df["Volume"], name="Volume",
                marker_color="rgba(100,100,100,0.3)",
            ),
            row=2, col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Layout
    name = info.get("name", ticker)
    fig.update_layout(
        title=f"{name} ({ticker}) — {window}",
        xaxis_rangeslider_visible=False,
        height=550 if show_volume else 450,
        margin=dict(t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
    )
    fig.update_yaxes(title_text="Price", tickformat=".4f", hoverformat=".4f", row=1, col=1)

    # Info bar
    sector = _default_sector or info.get("sector", "Unknown")
    industry = _default_industry or info.get("industry", "Unknown")
    pe = info.get("trailingPE")
    pe_str = f"P/E {pe:.1f}" if pe else "P/E n/a"
    beta = info.get("beta")
    beta_str = f"β {beta:.2f}" if beta else "β n/a"
    info_text = f"{name} | {industry} | {sector} | {pe_str} | {beta_str}"
    if _default_wac:
        info_text += f" | WAC ${_default_wac:.2f}"

    # --- Compute signal scores ---
    from finsell.cup_handle_score import score_cup_and_handle
    from finsell.distribution_days import score_distribution_days
    from finsell.climax_top import score_climax_top
    from finsell.broken_trend import score_broken_trend
    from finsell.overbought import score_overbought
    from finsell.double_bottom import score_double_bottom_undercut
    from finsell.purchase_timing import score_three_weeks_tight, score_ascending_base, score_purchase_timing
    from finsell.pyramiding import score_flat_base, score_ma_support
    from finsell.indicators import golden_cross_indicator

    def safe_score(fn, *args, **kwargs):
        """Run a scoring function, return None on failure."""
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    # Scores (all use daily data for best resolution)
    daily = data["daily"]
    if daily is not None and len(daily) > 0:
        score_df = daily.copy()
    else:
        score_df = df.copy()

    cup = safe_score(score_cup_and_handle, score_df)
    cup_val = cup.depth_score if cup else None

    dist = safe_score(score_distribution_days, score_df)
    dist_val = dist.score if dist else None

    climax = safe_score(score_climax_top, score_df)
    climax_val = climax.score if climax else None

    broken = safe_score(score_broken_trend, score_df)
    broken_val = broken.score if broken else None

    ob = safe_score(score_overbought, score_df)
    ob_val = ob.score if ob else None

    dbl = safe_score(score_double_bottom_undercut, score_df)
    dbl_val = dbl.score if dbl else None

    twt = safe_score(score_three_weeks_tight, score_df)
    twt_val = twt.score if twt else None

    asc = safe_score(score_ascending_base, score_df)
    asc_val = asc.score if asc else None

    flat = safe_score(score_flat_base, score_df)
    flat_val = flat.score if flat else None

    ma_sup = safe_score(score_ma_support, score_df)
    ma_sup_val = ma_sup.score if ma_sup else None

    gc_series = safe_score(golden_cross_indicator, score_df)
    gc_val = float(gc_series.iloc[-1]) if gc_series is not None and len(gc_series) > 0 else None

    pt = safe_score(score_purchase_timing, score_df, cost_basis=_default_wac)
    pt_val = pt.score if pt else None

    # Summary gauges
    all_scores = [s for s in [dist_val, climax_val, broken_val, ob_val, dbl_val,
                              twt_val, asc_val, flat_val, ma_sup_val, gc_val, pt_val] if s is not None]
    momentum = sum(all_scores) / len(all_scores) if all_scores else 0
    # Composite: average of 0-1 signals
    pos_scores = [s for s in [cup_val] if s is not None]
    composite = sum(pos_scores) / len(pos_scores) if pos_scores else 0
    strength = (momentum + 1) / 2  # map -1..+1 → 0..1

    # --- Decision Signals ---
    from finsell.decision_signals import compute_decision_signals

    # Use SPY and TSX daily data for dual market pulse
    index_daily = None
    tsx_daily = None
    indices = data.get("indices", {})
    if "SP500" in indices:
        _, index_daily = indices["SP500"]
    if "TSX" in indices:
        _, tsx_daily = indices["TSX"]

    decisions = compute_decision_signals(
        stock_df=score_df,
        index_df=index_daily,
        tsx_index_df=tsx_daily,
        wac=_default_wac,
        ticker=ticker,
        market_override=market_override,
    )

    # Market pulse traffic light (dual)
    pulse = decisions.market_pulse  # DualMarketPulse
    pulse_colors = {"green": "success", "yellow": "warning", "red": "danger"}

    us_badge = dbc.Badge(
        f"US: {pulse.us.stage.upper()}",
        color=pulse_colors.get(pulse.us.stage, "secondary"),
        className="me-1",
    )
    tsx_badge = dbc.Badge(
        f"TSX: {pulse.tsx.stage.upper()}",
        color=pulse_colors.get(pulse.tsx.stage, "secondary"),
        className="me-1",
    )
    eff_badge = dbc.Badge(
        f"M={pulse.effective_multiplier}",
        color=pulse_colors.get(pulse.effective_stage, "secondary"),
        className="fs-6 p-2 mb-1",
    )
    us_detail = html.Small(
        f"US: {pulse.us.distribution_count}d ({pulse.us.stalling_count}s)"
        + (" FTD✓" if pulse.us.ftd_detected else ""),
        className="text-muted d-block",
    )
    tsx_detail = html.Small(
        f"TSX: {pulse.tsx.distribution_count}d ({pulse.tsx.stalling_count}s)"
        + (" FTD✓" if pulse.tsx.ftd_detected else ""),
        className="text-muted d-block",
    )
    pulse_content = html.Div([eff_badge, html.Br(), us_badge, tsx_badge, us_detail, tsx_detail])

    # Decision details summary
    detail_items = []
    for signal_name, detail_dict in decisions.details.items():
        if isinstance(detail_dict, dict):
            for k, v in detail_dict.items():
                if k != "reason":
                    detail_items.append(f"{signal_name}.{k}: {v}")
                else:
                    detail_items.append(f"{signal_name}: {v}")
    details_content = html.Ul(
        [html.Li(item) for item in detail_items[:8]],
        className="list-unstyled mb-0",
    )

    return (
        fig,
        html.Small(info_text, className="text-muted"),
        gauge_0_to_1(composite, "Composite Score"),
        gauge_0_to_1(strength, "Trend Strength"),
        gauge_neg1_to_pos1(momentum, "Momentum"),
        mini_gauge(cup_val, "Cup & Handle", "0_1"),
        mini_gauge(dist_val, "Distribution"),
        mini_gauge(climax_val, "Climax Top"),
        mini_gauge(broken_val, "Broken Trend"),
        mini_gauge(ob_val, "Overbought"),
        mini_gauge(dbl_val, "Double Bottom"),
        mini_gauge(twt_val, "3 Wks Tight"),
        mini_gauge(asc_val, "Ascending Base"),
        mini_gauge(flat_val, "Flat Base"),
        mini_gauge(ma_sup_val, "MA Support"),
        mini_gauge(gc_val, "Golden Cross"),
        mini_gauge(pt_val, "Purchase Timing"),
        pulse_content,
        decision_gauge(decisions.buy, "BUY", "green"),
        decision_gauge(decisions.sell, "SELL", "red"),
        decision_gauge(decisions.hold, "HOLD", "blue"),
        decision_gauge(decisions.buy_more, "BUY MORE", "green"),
        details_content,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=_port)
