"""
pages/04_mc_naive_floating.py — Dashboard 4.4: MC Naive, floating barrier.

Adapted from the original standalone dashboard
(Dashboard_Floating_MC_Naive.py, port 8053). Pricing now comes from the
shared pricing_floating module instead of the local mc_naive/bs_vanilla
duplicates.
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_floating import mc_naive

dash.register_page(__name__, path="/mc-naive-floating",
                    name="4.4 MC Naive (Floating)")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
GRAY, DGRAY, BG, DARK, LGRAY = "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"
GREEN = "#1D9E75"

PID = "mcf4-"

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
        html.H1("MC Naive — Floating Barrier", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P(
            "B\u209c = \u03b1 \u00b7 max\u209a\u2264\u209c S\u209a   \u2014   "
            "Discrete monitoring, no intra-step correction",
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
                          children=[dcc.Graph(id=f"{PID}g-price", style={"height": "300px"},
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
                          children=[dcc.Graph(id=f"{PID}g-sig", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
            ]),
        ]),
    ]),

    html.P("S\u2080=100, q=0. Press Run to update.",
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
    S0 = 100.0
    alpha = alpha_pct / 100; K = float(K)
    sigma = sig_pct / 100; r = r_pct / 100; T = t_sc / 100; M = M_k * 1000

    res = mc_naive(S0, K, alpha, r, sigma, T, N=N, M=M)

    # Alpha sweep
    alphas = np.linspace(0.50, 0.97, 20)
    ap, ac, ad, sv_a = [], [], [], []
    for a in alphas:
        rr = mc_naive(S0, K, a, r, sigma, T, N=N, M=M)
        ap.append(rr["do_put"]); ac.append(rr["do_call"])
        ad.append(rr["di_call"]); sv_a.append(rr["surv"])

    # N sweep
    N_vals = [10, 25, 52, 100, 252]
    conv_p, conv_s = [], []
    for nv in N_vals:
        rr = mc_naive(S0, K, alpha, r, sigma, T, N=nv, M=M)
        conv_p.append(rr["do_put"]); conv_s.append(rr["surv"])

    # Sigma sweep
    sigs = np.linspace(0.05, 0.55, 20)
    sp_s, sc_s = [], []
    for sg in sigs:
        rr = mc_naive(S0, K, alpha, r, sg, T, N=N, M=M)
        sp_s.append(rr["do_put"]); sc_s.append(rr["do_call"])

    store = {
        "alphas": alphas.tolist(), "ap": ap, "ac": ac, "ad": ad, "sv_a": sv_a,
        "N_vals": N_vals, "conv_p": conv_p, "conv_s": conv_s,
        "sigs": (sigs * 100).tolist(), "sp_s": sp_s, "sc_s": sc_s,
        "res": res, "alpha": alpha, "N": N, "sigma_pct": sig_pct,
        "c_bs": res["c_bs"],
    }

    panel = html.Div([
        html.P("CURRENT PRICES", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                         "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lb, style=LABEL),
                    html.Span(f"{vl:.4f}", style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, vl, col in [
              ("DO Put", res["do_put"], BLUE),
              ("\u00b1 SE", res["se_put"], GRAY),
              ("DO Call", res["do_call"], FUCHSIA),
              ("DI Call", res["di_call"], LBLUE),
              ("BS Vanilla", res["c_bs"], DGRAY),
              ("Survival", res["surv"], GREEN),
          ]],
    ])
    return store, panel


@callback(
    Output(f"{PID}g-price", "figure"), Output(f"{PID}g-surv", "figure"),
    Output(f"{PID}g-conv", "figure"), Output(f"{PID}g-sig", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]; ap = d["ap"]; ac = d["ac"]; ad = d["ad"]
    sv_a = d["sv_a"]
    N_vals = d["N_vals"]; conv_p = d["conv_p"]; conv_s = d["conv_s"]
    sigs = d["sigs"]; sp_s = d["sp_s"]; sc_s = d["sc_s"]
    c_bs = d["c_bs"]

    # ── Price vs alpha ───────────────────────────────────────────────────
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=ap, name="DO Put",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f1.add_trace(go.Scatter(x=alphas, y=ac, name="DO Call",
        line=dict(color=FUCHSIA, width=2.0), mode="lines"))
    f1.add_trace(go.Scatter(x=alphas, y=ad, name="DI Call",
        line=dict(color=LBLUE, width=2.0, dash="dash"), mode="lines"))
    f1.add_hline(y=c_bs, line_color=GRAY, line_dash="dot", line_width=1.5,
                 annotation_text=f"BS={c_bs:.3f}", annotation_font_size=10)
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b1={d['alpha']:.2f}", annotation_font_size=10)
    f1.update_layout(**BASE, title=dict(text="Price vs \u03b1  (MC Naive)", font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1"); f1.update_yaxes(title_text="Option price")

    # ── Survival vs alpha ────────────────────────────────────────────────
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=alphas, y=sv_a, name="Survival",
        line=dict(color=DGRAY, width=2.5), mode="lines",
        fill="tozeroy", fillcolor="rgba(24,95,165,0.08)"))
    f2.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1)
    f2.update_layout(**BASE, title=dict(text="Survival Probability vs \u03b1", font=dict(size=12)))
    f2.update_xaxes(title_text="\u03b1"); f2.update_yaxes(title_text="P(survived)", range=[0, 1])

    # ── Convergence in N ────────────────────────────────────────────────
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=N_vals, y=conv_p, name="DO Put",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=7)))
    f3.add_vline(x=d["N"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"N={d['N']}", annotation_font_size=10)
    f3.update_layout(**BASE, title=dict(text="DO Put Convergence in N", font=dict(size=12)))
    f3.update_xaxes(title_text="N steps"); f3.update_yaxes(title_text="DO Put price")

    # ── Price vs sigma ───────────────────────────────────────────────────
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=sigs, y=sp_s, name="DO Put",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f4.add_trace(go.Scatter(x=sigs, y=sc_s, name="DO Call",
        line=dict(color=FUCHSIA, width=2.0), mode="lines"))
    f4.add_vline(x=d["sigma_pct"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03c3={d['sigma_pct']}%", annotation_font_size=10)
    f4.update_layout(**BASE, title=dict(text="Price vs \u03c3  (MC Naive)", font=dict(size=12)))
    f4.update_xaxes(title_text="\u03c3 (%)"); f4.update_yaxes(title_text="Option price")

    return f1, f2, f3, f4
