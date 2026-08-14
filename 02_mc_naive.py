"""
pages/02_mc_naive.py — Dashboard 2: Monte Carlo Naive (fixed barrier).

Adapted from the original standalone dashboard (dashboard_mc_naive.py,
port 8051). Pricing formulas now come from the shared pricing_bs module.
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_bs import bs_do_put, bs_di_call

dash.register_page(__name__, path="/mc-naive", name="2. MC Naive")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
LFUCHSIA = "#F4C0D1"
GRAY, DGRAY, BG, DARK, LGRAY = "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"
GREEN = "#1D9E75"

LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif"}
VALUE = {"fontSize": "13px", "fontWeight": "600", "color": DARK, "fontFamily": "Georgia,serif"}
CARD = {"backgroundColor": "white", "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}

PID = "mc2-"
STEP_GRID = [10, 20, 52, 104, 252, 504]


def mc_naive_sweep(opt, S0, K, H, r, q, sigma, T, M=20000):
    prices, ses = [], []
    for N in STEP_GRID:
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
        prices.append(float(d.mean())); ses.append(float(d.std(ddof=1) / np.sqrt(M)))
    return np.array(prices), np.array(ses)


def generate_paths(S0, H, r, q, sigma, T, N=252, M_show=150):
    rng = np.random.default_rng(0)
    dt = T / N
    Z = rng.standard_normal((M_show, N))
    lnS = np.log(S0) + np.cumsum((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S_sim = np.hstack([np.full((M_show, 1), S0), np.exp(lnS)])
    t_grid = np.linspace(0, T, N + 1)
    survived = np.all(S_sim[:, 1:] > H, axis=1)
    return S_sim, t_grid, survived


layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                          "fontFamily": "Georgia,serif", "padding": "24px 32px"}, children=[

    html.Div([
        html.H1("Monte Carlo Naive", style={"fontSize": "20px", "fontWeight": "600",
                                             "margin": "0 0 3px", "color": DARK}),
        html.P("Euler-Maruyama discretization — systematic bias O(\u221a\u0394t)",
               style={"fontSize": "12px", "color": DGRAY, "margin": 0, "fontStyle": "italic"}),
    ], style={"marginBottom": "22px", "borderBottom": f"1px solid {LGRAY}", "paddingBottom": "14px"}),

    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[

        html.Div(style={**CARD, "width": "240px", "flexShrink": "0"}, children=[
            html.P("PARAMETERS", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                         "color": GRAY, "marginBottom": "16px"}),
            *[html.Div([
                html.Div([html.Span(lbl, style=LABEL),
                          html.Span(id=f"{PID}{id_}-val", style=VALUE)],
                         style={"display": "flex", "justifyContent": "space-between",
                                "marginBottom": "4px"}),
                dcc.Slider(id=f"{PID}{id_}", min=mn, max=mx, step=st, value=vl,
                           marks=None, tooltip={"always_visible": False}),
            ], style={"marginBottom": "16px"})
              for lbl, id_, mn, mx, st, vl in [
                  ("Barrier H", "h", 50, 99, 1, 80),
                  ("Volatility \u03c3", "sigma", 5, 60, 1, 20),
                  ("Maturity T (yr)", "t", 25, 200, 5, 100),
                  ("Risk-free r", "r", 0, 15, 1, 5),
              ]],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 16px"}),
            html.P("OPTION TYPE", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                          "color": GRAY, "marginBottom": "10px"}),
            dcc.RadioItems(
                id=f"{PID}opt-type",
                options=[{"label": "  Down-and-Out Put", "value": "do"},
                         {"label": "  Down-and-In Call", "value": "di"}],
                value="do",
                style={"fontSize": "12px", "fontFamily": "Georgia,serif", "color": DARK},
                labelStyle={"display": "block", "marginBottom": "8px"}),
            html.Hr(style={"borderColor": LGRAY, "margin": "16px 0"}),
            html.Button("\u25b6  Run Simulation", id=f"{PID}run-btn", n_clicks=0, style={
                "width": "100%", "padding": "10px", "backgroundColor": BLUE,
                "color": "white", "border": "none", "borderRadius": "8px",
                "fontSize": "13px", "fontFamily": "Georgia,serif",
                "cursor": "pointer", "fontWeight": "600"}),
            html.Hr(style={"borderColor": LGRAY, "margin": "16px 0"}),
            html.Div(id=f"{PID}result-panel"),
        ]),

        html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column",
                         "gap": "16px", "minWidth": "480px"}, children=[
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}conv-graph", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}loglog-graph", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
            ]),
            html.Div(style=CARD,
                      children=[dcc.Graph(id=f"{PID}paths-graph", style={"height": "320px"},
                                           config={"displayModeBar": False})]),
        ]),
    ]),

    html.P("S\u2080=100, K=100, M=20,000 paths. Press 'Run Simulation' to update.",
           style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}sim-store"),
])


@callback(
    Output(f"{PID}h-val", "children"), Output(f"{PID}sigma-val", "children"),
    Output(f"{PID}t-val", "children"), Output(f"{PID}r-val", "children"),
    Input(f"{PID}h", "value"), Input(f"{PID}sigma", "value"),
    Input(f"{PID}t", "value"), Input(f"{PID}r", "value"),
)
def update_labels(h, sigma, t, r):
    return str(h), f"{sigma}%", f"{t/100:.2f}y", f"{r}%"


@callback(
    Output(f"{PID}sim-store", "data"),
    Output(f"{PID}result-panel", "children"),
    Input(f"{PID}run-btn", "n_clicks"),
    State(f"{PID}h", "value"), State(f"{PID}sigma", "value"),
    State(f"{PID}t", "value"), State(f"{PID}r", "value"),
    State(f"{PID}opt-type", "value"),
    prevent_initial_call=False,
)
def run_simulation(n_clicks, h, sigma_pct, t_sc, r_pct, opt):
    S0 = 100.0; K = 100.0
    H = float(h); sigma = sigma_pct / 100; r = r_pct / 100; T = t_sc / 100
    bs_val = bs_do_put(S0, K, H, r, 0, sigma, T) if opt == "do" \
        else bs_di_call(S0, K, H, r, 0, sigma, T)
    prices, ses = mc_naive_sweep(opt, S0, K, H, r, 0, sigma, T)
    biases = prices - bs_val
    i52 = STEP_GRID.index(52)

    panel = html.Div([
        html.P("RESULTS", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                  "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lbl, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lbl, val, col in [
              ("BS benchmark", f"{bs_val:.4f}", DARK),
              ("Naive (N=52)", f"{prices[i52]:.4f}", FUCHSIA),
              ("Bias (N=52)", f"{biases[i52]:+.4f}",
               "#E24B4A" if abs(biases[i52]) > 0.05 else GREEN),
              ("Bias (N=504)", f"{biases[-1]:+.4f}",
               "#E24B4A" if abs(biases[-1]) > 0.05 else GREEN),
          ]],
    ])
    store = {"prices": prices.tolist(), "ses": ses.tolist(),
             "bs_val": bs_val, "biases": biases.tolist(),
             "H": H, "sigma": sigma, "r": r, "T": T, "opt": opt}
    return store, panel


@callback(
    Output(f"{PID}conv-graph", "figure"),
    Output(f"{PID}loglog-graph", "figure"),
    Output(f"{PID}paths-graph", "figure"),
    Input(f"{PID}sim-store", "data"),
)
def update_graphs(data):
    def empty_fig():
        return go.Figure().update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            xaxis=dict(visible=False), yaxis=dict(visible=False))

    if data is None:
        return empty_fig(), empty_fig(), empty_fig()

    prices = np.array(data["prices"])
    ses = np.array(data["ses"])
    bs_val = data["bs_val"]
    biases = np.abs(np.array(data["biases"]))
    H = data["H"]; sigma = data["sigma"]; r = data["r"]; T = data["T"]

    def base_layout(title):
        return dict(
            paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Georgia,serif", color=DARK, size=11),
            margin=dict(l=50, r=20, t=45, b=50),
            xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
            legend=dict(font=dict(size=10)), hovermode="x unified",
            title=dict(text=title, font=dict(size=12)))

    # ── Convergence ──────────────────────────────────────────────────────
    f1 = go.Figure()
    f1.add_hline(y=bs_val, line_color=DARK, line_dash="dash", line_width=2,
                 annotation_text="BS analytical", annotation_font_size=10)
    f1.add_trace(go.Scatter(
        x=STEP_GRID, y=prices, mode="lines+markers", name="MC Naive",
        line=dict(color=FUCHSIA, width=2.5), marker=dict(size=7, symbol="square"),
        error_y=dict(type="data", array=2 * ses, visible=True,
                     color=LFUCHSIA, thickness=1.5)))
    f1.update_layout(**base_layout("Price convergence vs N"))
    f1.update_xaxes(title_text="Time steps N")
    f1.update_yaxes(title_text="Option price")

    # ── Log-log ───────────────────────────────────────────────────────────
    log_N = np.log(STEP_GRID)
    slope, ic = np.polyfit(log_N, np.log(biases + 1e-8), 1)
    ref = np.exp(ic) * np.array(STEP_GRID) ** (-0.5)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(
        x=STEP_GRID, y=biases, mode="lines+markers",
        name=f"Naive |bias| (slope={slope:.2f})",
        line=dict(color=FUCHSIA, width=2.5), marker=dict(size=7, symbol="square")))
    f2.add_trace(go.Scatter(
        x=STEP_GRID, y=ref, mode="lines", name="O(N\u207b\u2070\u00b7\u2075) theory",
        line=dict(color=GRAY, width=1.5, dash="dot")))
    f2.update_layout(**base_layout(f"Log-log bias  [slope \u2248 {slope:.2f}, theory \u22120.50]"))
    f2.update_xaxes(title_text="Time steps N", type="log")
    f2.update_yaxes(title_text="|MC price \u2212 BS price|", type="log")

    # ── Sample paths ─────────────────────────────────────────────────────
    S_sim, t_grid, survived = generate_paths(100.0, H, r, 0, sigma, T)
    knocked = ~survived

    f3 = go.Figure()
    for i in range(len(S_sim)):
        f3.add_trace(go.Scatter(
            x=t_grid, y=S_sim[i], mode="lines", showlegend=False,
            line=dict(color=LBLUE if survived[i] else LFUCHSIA, width=0.5),
            opacity=0.3 if survived[i] else 0.2))
    surv_idx = np.where(survived)[0][:3]
    knock_idx = np.where(knocked)[0][:3]
    for i, idx in enumerate(surv_idx):
        f3.add_trace(go.Scatter(
            x=t_grid, y=S_sim[idx], mode="lines",
            name="Survived", showlegend=bool(i == 0),
            line=dict(color=BLUE, width=1.8)))
    for i, idx in enumerate(knock_idx):
        f3.add_trace(go.Scatter(
            x=t_grid, y=S_sim[idx], mode="lines",
            name="Knocked out", showlegend=bool(i == 0),
            line=dict(color=FUCHSIA, width=1.8)))
    f3.add_hline(y=H, line_color="#E24B4A", line_dash="dash", line_width=2,
                 annotation_text=f"Barrier H={int(H)}", annotation_font_size=10)
    f3.add_hline(y=100, line_color=DGRAY, line_dash="dot", line_width=1,
                 annotation_text="Strike K=100", annotation_font_size=10)
    f3.update_layout(**base_layout(
        f"Euler-Maruyama sample paths  |  Survived: {survived.sum()}  \u00b7  Knocked: {knocked.sum()}"))
    f3.update_xaxes(title_text="Time t (years)")
    f3.update_yaxes(title_text="Asset price S(t)")

    return f1, f2, f3
