"""
pages/05_mc_bb_floating.py — Dashboard 5.5: MC BB, floating barrier.

Adapted from the original standalone dashboard
(Dashboard_Floating_MC_BB.py, port 8054). Pricing now comes from the
shared pricing_floating module instead of the local mc_naive/mc_bb
duplicates. BB here is a local approximation, not exact for the
floating barrier (already noted in the original dashboard's caption).
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_floating import mc_naive, mc_bb

dash.register_page(__name__, path="/mc-bb-floating",
                    name="5.5 MC BB (Floating)")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
GRAY, DGRAY, BG, DARK, LGRAY = "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"
GREEN = "#1D9E75"

PID = "mcbb5-"

LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif", "letterSpacing": "0.02em"}
VALUE = {"fontSize": "13px", "fontWeight": "600", "color": DARK, "fontFamily": "Georgia,serif"}
CARD = {"backgroundColor": "white", "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}
BASE = dict(paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Georgia,serif", color=DARK, size=11),
            margin=dict(l=50, r=20, t=45, b=50),
            xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
            legend=dict(font=dict(size=10)), hovermode="x unified")

layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                          "fontFamily": "Georgia,serif", "padding": "24px 32px"}, children=[

    html.Div([
        html.H1("MC BB — Floating Barrier", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P(
            "Local Brownian Bridge approximation for continuous-monitoring effects",
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
                  ("Barrier fraction \u03b1", "alpha", 50, 98, 1, 80),
                  ("Strike K", "k", 80, 120, 1, 100),
                  ("Volatility \u03c3", "sig", 5, 60, 1, 20),
                  ("Risk-free r", "r", 0, 15, 1, 5),
                  ("Maturity T (yr)", "t", 25, 200, 5, 100),
                  ("Steps N", "n", 10, 252, 10, 52),
                  ("Paths M (\u00d71k)", "m", 5, 100, 5, 30),
              ]],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 14px"}),
            html.Button("\u25b6  Run", id=f"{PID}run", n_clicks=0, style={
                "width": "100%", "padding": "10px", "backgroundColor": BLUE,
                "color": "white", "border": "none", "borderRadius": "8px",
                "fontSize": "13px", "fontFamily": "Georgia,serif",
                "cursor": "pointer", "fontWeight": "600"}),
            html.Hr(style={"borderColor": LGRAY, "margin": "14px 0"}),
            html.Div(id=f"{PID}res"),
        ]),

        html.Div(style={"flex": "1", "display": "flex", "flexDirection": "column",
                         "gap": "16px", "minWidth": "480px"}, children=[
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-cmp", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-surv", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
            ]),
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-conv", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-bias", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
            ]),
        ]),
    ]),

    html.P("S\u2080=100, q=0. BB is a local approximation \u2014 not exact for floating barrier.",
           style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}store"),
])


@callback(
    Output(f"{PID}alpha-val", "children"), Output(f"{PID}k-val", "children"),
    Output(f"{PID}sig-val", "children"), Output(f"{PID}r-val", "children"),
    Output(f"{PID}t-val", "children"), Output(f"{PID}n-val", "children"),
    Output(f"{PID}m-val", "children"),
    Input(f"{PID}alpha", "value"), Input(f"{PID}k", "value"), Input(f"{PID}sig", "value"),
    Input(f"{PID}r", "value"), Input(f"{PID}t", "value"), Input(f"{PID}n", "value"),
    Input(f"{PID}m", "value"),
)
def lbl(a, k, s, r, t, n, m):
    return f"{a/100:.2f}", str(k), f"{s}%", f"{r}%", f"{t/100:.2f}y", str(n), f"{m}k"


@callback(
    Output(f"{PID}store", "data"), Output(f"{PID}res", "children"),
    Input(f"{PID}run", "n_clicks"),
    State(f"{PID}alpha", "value"), State(f"{PID}k", "value"), State(f"{PID}sig", "value"),
    State(f"{PID}r", "value"), State(f"{PID}t", "value"), State(f"{PID}n", "value"),
    State(f"{PID}m", "value"),
    prevent_initial_call=False,
)
def run(nc, alpha_pct, K, sig_pct, r_pct, t_sc, N, M_k):
    S0 = 100.0; alpha = alpha_pct / 100; K = float(K)
    sigma = sig_pct / 100; r = r_pct / 100; T = t_sc / 100; M = M_k * 1000

    bb = mc_bb(S0, K, alpha, r, sigma, T, N=N, M=M)
    nv = mc_naive(S0, K, alpha, r, sigma, T, N=N, M=M)

    # Alpha sweep
    alphas = np.linspace(0.50, 0.97, 20)
    bb_p, nv_p, bb_s, nv_s = [], [], [], []
    for a in alphas:
        r1 = mc_bb(S0, K, a, r, sigma, T, N=N, M=M)
        r2 = mc_naive(S0, K, a, r, sigma, T, N=N, M=M)
        bb_p.append(r1["do_put"]); nv_p.append(r2["do_put"])
        bb_s.append(r1["surv"]); nv_s.append(r2["surv"])

    # N sweep
    N_vals = [10, 25, 52, 100, 252]
    bb_conv, nv_conv = [], []
    for nv2 in N_vals:
        r1 = mc_bb(S0, K, alpha, r, sigma, T, N=nv2, M=M)
        r2 = mc_naive(S0, K, alpha, r, sigma, T, N=nv2, M=M)
        bb_conv.append(r1["do_put"]); nv_conv.append(r2["do_put"])

    store = {
        "alphas": alphas.tolist(), "bb_p": bb_p, "nv_p": nv_p,
        "bb_s": bb_s, "nv_s": nv_s,
        "N_vals": N_vals, "bb_conv": bb_conv, "nv_conv": nv_conv,
        "bb": bb, "nv": nv, "alpha": alpha, "N": N,
    }

    diff = bb["do_put"] - nv["do_put"]
    panel = html.Div([
        html.P("CURRENT PRICES", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                         "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lb, style=LABEL),
                    html.Span(f"{vl:.4f}", style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, vl, col in [
              ("BB Put", bb["do_put"], BLUE),
              ("Naive Put", nv["do_put"], FUCHSIA),
              ("Diff", diff, "#E24B4A" if diff > 0 else GREEN),
              ("BB Surv", bb["surv"], BLUE),
              ("Naive Surv", nv["surv"], FUCHSIA),
              ("BS Call", bb["c_bs"], DGRAY),
          ]],
    ])
    return store, panel


@callback(
    Output(f"{PID}g-cmp", "figure"), Output(f"{PID}g-surv", "figure"),
    Output(f"{PID}g-conv", "figure"), Output(f"{PID}g-bias", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]; bb_p = d["bb_p"]; nv_p = d["nv_p"]
    bb_s = d["bb_s"]; nv_s = d["nv_s"]
    N_vals = d["N_vals"]; bb_conv = d["bb_conv"]; nv_conv = d["nv_conv"]

    # ── BB vs Naive price vs alpha ───────────────────────────────────────
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=bb_p, name="MC BB",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f1.add_trace(go.Scatter(x=alphas, y=nv_p, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1)
    f1.update_layout(**BASE, title=dict(text="DO Put: BB vs Naive vs \u03b1", font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1"); f1.update_yaxes(title_text="DO Put price")

    # ── Survival vs alpha ────────────────────────────────────────────────
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=alphas, y=bb_s, name="BB surv",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f2.add_trace(go.Scatter(x=alphas, y=nv_s, name="Naive surv",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f2.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1)
    f2.update_layout(**BASE, title=dict(text="Effective Survival vs \u03b1", font=dict(size=12)))
    f2.update_xaxes(title_text="\u03b1"); f2.update_yaxes(title_text="Survival", range=[0, 1])

    # ── Convergence in N ────────────────────────────────────────────────
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=N_vals, y=bb_conv, name="MC BB",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=7)))
    f3.add_trace(go.Scatter(x=N_vals, y=nv_conv, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines+markers",
        marker=dict(size=7, symbol="square")))
    f3.add_vline(x=d["N"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"N={d['N']}", annotation_font_size=10)
    f3.update_layout(**BASE, title=dict(text="DO Put Convergence in N", font=dict(size=12)))
    f3.update_xaxes(title_text="N steps"); f3.update_yaxes(title_text="DO Put price")

    # ── BB vs Naive difference vs N ──────────────────────────────────────
    diff_N = [b - n for b, n in zip(bb_conv, nv_conv)]
    f4 = go.Figure()
    f4.add_trace(go.Bar(x=[str(n) for n in N_vals], y=diff_N,
        marker_color=[BLUE if v < 0 else FUCHSIA for v in diff_N], name="BB - Naive"))
    f4.add_hline(y=0, line_color=DGRAY, line_width=1)
    f4.update_layout(**BASE, title=dict(text="DO Put: BB \u2212 Naive (per N)", font=dict(size=12)))
    f4.update_xaxes(title_text="N steps"); f4.update_yaxes(title_text="Price difference")

    return f1, f2, f3, f4
