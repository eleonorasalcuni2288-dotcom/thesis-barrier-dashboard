"""
pages/03_brownian_bridge.py — Dashboard 3: Brownian Bridge Correction.

Adapted from the original standalone dashboard
(dashboard_brownian_bridge.py, port 8052). Pricing formulas now come
from the shared pricing_bs module instead of the local _rr/_rr_components
duplicate (verified numerically equivalent).

FIX 1: the log-log bias chart's x-axis now uses explicit
tickvals/ticktext (STEP_GRID values, as strings) instead of letting
Plotly auto-generate log-scale tick labels. Plotly's default log-axis
tick formatter abbreviates non-decade ticks (20 -> "2", 500 -> "5")
when space is tight, which looked like duplicated/wrong values in the
deployed app.

FIX 2: the SE Reduction and MSE decomposition bar charts now force
type="category" on their x-axis. Without this, Plotly auto-detects the
numeric-looking bar labels ("10","20",...,"504") and switches to a
linear numeric axis, computing bar width from the *smallest* gap
between consecutive values (10 -> 20 = 10 units). On a linear axis
spanning 10-504 that made every bar except the first one nearly
invisible (they looked "squished" at the left edge).
"""

import dash
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback

from pricing_bs import bs_do_put, bs_di_call

dash.register_page(__name__, path="/brownian-bridge",
                    name="3. Brownian Bridge Correction")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
LFUCHSIA = "#F4C0D1"
GRAY, DGRAY, BG, DARK, LGRAY = "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"
GREEN = "#1D9E75"

PID = "bb3-"
STEP_GRID = [10, 20, 52, 104, 252, 504]

LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif", "letterSpacing": "0.02em"}
VALUE = {"fontSize": "13px", "fontWeight": "600", "color": DARK, "fontFamily": "Georgia,serif"}
CARD = {"backgroundColor": "white", "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}


def mc_naive_sweep(opt, S0, K, H, r, q, sigma, T, steps, M=15000):
    prices, ses = [], []
    for N in steps:
        rng = np.random.default_rng(42)
        dt = T / N; drift = (r - q - 0.5 * sigma ** 2) * dt; vol = sigma * np.sqrt(dt)
        Z = rng.standard_normal((M, N))
        lnS = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
        Sp = np.exp(lnS); ST = Sp[:, -1]
        if opt == "do":
            pay = np.maximum(K - ST, 0) * np.all(Sp > H, axis=1)
        else:
            pay = np.maximum(ST - K, 0) * np.any(Sp <= H, axis=1)
        d = np.exp(-r * T) * pay
        prices.append(d.mean()); ses.append(d.std(ddof=1) / np.sqrt(M))
    return np.array(prices), np.array(ses)


def mc_bb_sweep(opt, S0, K, H, r, q, sigma, T, steps, M=15000):
    prices, ses = [], []
    for N in steps:
        rng = np.random.default_rng(42)
        dt = T / N; lnH = np.log(H)
        drift = (r - q - 0.5 * sigma ** 2) * dt; vol = sigma * np.sqrt(dt); var_dt = sigma ** 2 * dt
        Z = rng.standard_normal((M, N))
        lnS = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
        lnSf = np.hstack([np.full((M, 1), np.log(S0)), lnS])
        a = lnSf[:, :-1]; b = lnSf[:, 1:]
        pn = np.where((a > lnH) & (b > lnH), np.exp(-2 * (a - lnH) * (b - lnH) / var_dt), 1.0)
        pn = np.clip(pn, 0, 1)
        ns = np.all(lnS > lnH, axis=1)
        sp = np.where(ns, np.prod(1 - pn, axis=1), 0.0)
        ST = np.exp(lnS[:, -1])
        if opt == "do":
            pay = np.maximum(K - ST, 0) * sp
        else:
            pay = np.maximum(ST - K, 0) * (1 - sp)
        d = np.exp(-r * T) * pay
        prices.append(d.mean()); ses.append(d.std(ddof=1) / np.sqrt(M))
    return np.array(prices), np.array(ses)


layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                          "fontFamily": "Georgia,serif", "padding": "24px 32px"}, children=[

    html.Div([
        html.H1("Brownian Bridge Correction", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P("Analytical inter-node correction — bias O(\u0394t), variance reduction (Rao-Blackwell)",
               style={"fontSize": "12px", "color": DGRAY, "margin": 0, "fontStyle": "italic"}),
    ], style={"marginBottom": "22px", "borderBottom": f"1px solid {LGRAY}", "paddingBottom": "14px"}),

    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[

        html.Div(style={**CARD, "width": "240px", "flexShrink": "0"}, children=[
            html.P("PARAMETERS", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                         "color": GRAY, "marginBottom": "16px"}),
            *[html.Div([
                html.Div([html.Span(lbl, style=LABEL), html.Span(id=f"{PID}{id_}-val", style=VALUE)],
                         style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                dcc.Slider(id=f"{PID}{id_}", min=mn, max=mx, step=st, value=vl,
                           marks=None, tooltip={"always_visible": False}),
            ], style={"marginBottom": "16px"})
              for lbl, id_, mn, mx, st, vl in [
                  ("Barrier H", "h2", 50, 99, 1, 80),
                  ("Volatility \u03c3", "sigma2", 5, 60, 1, 20),
                  ("Maturity T (yr)", "t2", 25, 200, 5, 100),
                  ("Risk-free r", "r2", 0, 15, 1, 5),
              ]],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 16px"}),
            html.P("OPTION TYPE", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                          "color": GRAY, "marginBottom": "10px"}),
            dcc.RadioItems(
                id=f"{PID}opt-type2",
                options=[{"label": "  Down-and-Out Put", "value": "do"},
                         {"label": "  Down-and-In Call", "value": "di"}],
                value="do",
                style={"fontSize": "12px", "fontFamily": "Georgia,serif", "color": DARK},
                labelStyle={"display": "block", "marginBottom": "8px"}),
            html.Hr(style={"borderColor": LGRAY, "margin": "16px 0"}),
            html.Button("\u25b6  Run Simulation", id=f"{PID}run-btn2", n_clicks=0,
                        style={"width": "100%", "padding": "10px", "backgroundColor": BLUE,
                               "color": "white", "border": "none", "borderRadius": "8px",
                               "fontSize": "13px", "fontFamily": "Georgia,serif",
                               "cursor": "pointer", "fontWeight": "600"}),
            html.Hr(style={"borderColor": LGRAY, "margin": "16px 0"}),
            html.Div(id=f"{PID}result-panel2"),
        ]),

        html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column",
                         "gap": "16px", "minWidth": "480px"}, children=[
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}conv2", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}loglog2", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
            ]),
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}se2", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}mse2", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
            ]),
        ]),
    ]),

    html.P("S\u2080=100, K=100, M=15,000 paths. Press 'Run Simulation' to update.",
           style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}sim-store2"),
])


@callback(
    Output(f"{PID}h2-val", "children"), Output(f"{PID}sigma2-val", "children"),
    Output(f"{PID}t2-val", "children"), Output(f"{PID}r2-val", "children"),
    Input(f"{PID}h2", "value"), Input(f"{PID}sigma2", "value"),
    Input(f"{PID}t2", "value"), Input(f"{PID}r2", "value"),
)
def update_labels(h, sigma, t, r):
    return str(h), f"{sigma}%", f"{t/100:.2f}y", f"{r}%"


@callback(
    Output(f"{PID}sim-store2", "data"),
    Output(f"{PID}result-panel2", "children"),
    Input(f"{PID}run-btn2", "n_clicks"),
    State(f"{PID}h2", "value"), State(f"{PID}sigma2", "value"),
    State(f"{PID}t2", "value"), State(f"{PID}r2", "value"),
    State(f"{PID}opt-type2", "value"),
    prevent_initial_call=False,
)
def run_sim(n_clicks, h, sigma_pct, t_sc, r_pct, opt):
    S0 = 100.0; K = 100.0
    H = float(h); sigma = sigma_pct / 100; r = r_pct / 100; T = t_sc / 100

    bs_val = bs_do_put(S0, K, H, r, 0, sigma, T) if opt == "do" \
        else bs_di_call(S0, K, H, r, 0, sigma, T)

    naive_p, naive_se = mc_naive_sweep(opt, S0, K, H, r, 0, sigma, T, STEP_GRID)
    bb_p, bb_se = mc_bb_sweep(opt, S0, K, H, r, 0, sigma, T, STEP_GRID)

    i52 = STEP_GRID.index(52)
    se_ratio = naive_se[i52] / bb_se[i52]
    bias_n52 = naive_p[i52] - bs_val
    bias_b52 = bb_p[i52] - bs_val

    panel = html.Div([
        html.P("RESULTS  (N=52)", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                          "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lbl, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lbl, val, col in [
              ("BS benchmark", f"{bs_val:.4f}", DARK),
              ("Naive price", f"{naive_p[i52]:.4f}", FUCHSIA),
              ("BB price", f"{bb_p[i52]:.4f}", BLUE),
              ("Naive bias", f"{bias_n52:+.4f}", "#E24B4A" if abs(bias_n52) > 0.05 else GREEN),
              ("BB bias", f"{bias_b52:+.4f}", "#E24B4A" if abs(bias_b52) > 0.05 else GREEN),
              ("SE ratio", f"{se_ratio:.2f}\u00d7", BLUE),
          ]],
    ])

    store = {
        "naive_p": naive_p.tolist(), "naive_se": naive_se.tolist(),
        "bb_p": bb_p.tolist(), "bb_se": bb_se.tolist(),
        "bs_val": bs_val,
    }
    return store, panel


@callback(
    Output(f"{PID}conv2", "figure"), Output(f"{PID}loglog2", "figure"),
    Output(f"{PID}se2", "figure"), Output(f"{PID}mse2", "figure"),
    Input(f"{PID}sim-store2", "data"),
)
def update_graphs(data):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if data is None:
        return empty, empty, empty, empty

    naive_p = np.array(data["naive_p"]); naive_se = np.array(data["naive_se"])
    bb_p = np.array(data["bb_p"]); bb_se = np.array(data["bb_se"])
    bs_val = data["bs_val"]

    base = dict(paper_bgcolor=BG, plot_bgcolor=BG,
                font=dict(family="Georgia,serif", color=DARK, size=11),
                margin=dict(l=50, r=20, t=45, b=50),
                xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
                legend=dict(font=dict(size=10)), hovermode="x unified")

    # ── Convergence ──────────────────────────────────────────────────────
    f1 = go.Figure()
    f1.add_hline(y=bs_val, line_color=DARK, line_dash="dash", line_width=2,
                 annotation_text="BS analytical", annotation_font_size=10)
    f1.add_trace(go.Scatter(x=STEP_GRID, y=naive_p, mode="lines+markers",
        name="MC Naive", line=dict(color=FUCHSIA, width=2.5), marker=dict(size=7, symbol="square"),
        error_y=dict(type="data", array=2 * naive_se, visible=True, color=LFUCHSIA, thickness=1.5)))
    f1.add_trace(go.Scatter(x=STEP_GRID, y=bb_p, mode="lines+markers",
        name="MC BB", line=dict(color=BLUE, width=2.5), marker=dict(size=7),
        error_y=dict(type="data", array=2 * bb_se, visible=True, color=LBLUE, thickness=1.5)))
    f1.update_layout(**base, title=dict(text="Price convergence vs N", font=dict(size=12)))
    f1.update_xaxes(title_text="Time steps N"); f1.update_yaxes(title_text="Option price")

    # ── Log-log ───────────────────────────────────────────────────────────
    nb = np.abs(naive_p - bs_val) + 1e-8; bb_b = np.abs(bb_p - bs_val) + 1e-8
    log_N = np.log(STEP_GRID)
    sn, icn = np.polyfit(log_N, np.log(nb), 1); sb, icb = np.polyfit(log_N, np.log(bb_b), 1)
    ref_half = np.exp(icn) * np.array(STEP_GRID) ** (-0.5)
    ref_one = np.exp(icn) * np.array(STEP_GRID) ** (-1.0)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=STEP_GRID, y=nb, mode="lines+markers",
        name=f"Naive (slope={sn:.2f})", line=dict(color=FUCHSIA, width=2.5), marker=dict(size=7, symbol="square")))
    f2.add_trace(go.Scatter(x=STEP_GRID, y=bb_b, mode="lines+markers",
        name=f"BB (slope={sb:.2f})", line=dict(color=BLUE, width=2.5), marker=dict(size=7)))
    f2.add_trace(go.Scatter(x=STEP_GRID, y=ref_half, mode="lines", name="O(N\u207b\u2070\u00b7\u2075)",
        line=dict(color=FUCHSIA, width=1, dash="dot"), opacity=0.6))
    f2.add_trace(go.Scatter(x=STEP_GRID, y=ref_one, mode="lines", name="O(N\u207b\u00b9\u00b7\u2070)",
        line=dict(color=BLUE, width=1, dash="dot"), opacity=0.6))
    f2.update_layout(**base, title=dict(text="Log-log bias comparison", font=dict(size=12)))
    # FIX 1: explicit tickvals/ticktext instead of Plotly's auto log-axis
    # ticks, which abbreviate non-decade values (20 -> "2", 500 -> "5").
    f2.update_xaxes(title_text="N", type="log",
                     tickvals=STEP_GRID, ticktext=[str(n) for n in STEP_GRID])
    f2.update_yaxes(title_text="|bias|", type="log")

    # ── SE ratio ─────────────────────────────────────────────────────────
    se_ratio = naive_se / bb_se
    f3 = go.Figure()
    f3.add_hline(y=1, line_color=GRAY, line_dash="dash", line_width=1)
    f3.add_trace(go.Bar(x=[str(n) for n in STEP_GRID], y=naive_se * 1000,
        name="Naive SE", marker_color=LFUCHSIA, marker_line_color=FUCHSIA, marker_line_width=1))
    f3.add_trace(go.Bar(x=[str(n) for n in STEP_GRID], y=bb_se * 1000,
        name="BB SE", marker_color=LBLUE, marker_line_color=BLUE, marker_line_width=1, opacity=0.85))
    for i, r_ in enumerate(se_ratio):
        f3.add_annotation(x=str(STEP_GRID[i]), y=max(naive_se[i], bb_se[i]) * 1000 + 0.03,
            text=f"{r_:.1f}\u00d7", showarrow=False, font=dict(size=9, color=DARK))
    f3.update_layout(**base, barmode="overlay", title=dict(text="SE Reduction  (Rao-Blackwell)", font=dict(size=12)))
    # FIX 2: force categorical x-axis (see module docstring for why).
    f3.update_xaxes(title_text="Time steps N", type="category")
    f3.update_yaxes(title_text="SE (\u00d710\u207b\u00b3)")

    # ── MSE decomposition ────────────────────────────────────────────────
    bias2_n = (naive_p - bs_val) ** 2; var_n = naive_se ** 2
    bias2_b = (bb_p - bs_val) ** 2; var_b = bb_se ** 2

    f4 = make_subplots(rows=1, cols=2, subplot_titles=["MC Naive", "MC Brownian Bridge"])
    for col, (b2, va, cb, cv) in enumerate([(bias2_n, var_n, FUCHSIA, LFUCHSIA),
                                             (bias2_b, var_b, BLUE, LBLUE)], 1):
        f4.add_trace(go.Bar(x=STEP_GRID, y=b2 * 1e4, name="Bias\u00b2",
            marker_color=cb, showlegend=(col == 1)), row=1, col=col)
        f4.add_trace(go.Bar(x=STEP_GRID, y=va * 1e4, name="Variance",
            marker_color=cv, showlegend=(col == 1)), row=1, col=col)
    f4.update_layout(**base, barmode="stack", title=dict(text="MSE = Bias\u00b2 + Variance  (\u00d710\u207b\u2074)", font=dict(size=12)))
    # FIX 2 (same as f3, applied to both subplot columns).
    f4.update_xaxes(title_text="N", type="category")
    f4.update_yaxes(title_text="MSE (\u00d710\u207b\u2074)")

    return f1, f2, f3, f4
