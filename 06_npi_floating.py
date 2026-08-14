"""
pages/06_npi_floating.py — Dashboard 6.6: NPI, floating barrier (fixed grid).

Adapted from the original standalone dashboard
(Dashboard_Floating_NPI.py, port 8055).

IMPORTANT CHANGE vs the original: the original `npi()` used a pure
Python double loop (`for _ in range(N_steps): for mp in range(Nx1): ...`)
which does not scale well, especially with 15+4 sweep calls per Run on
shared/free hosting CPUs. It has been replaced here by
pricing_floating.npi_uniform(), the vectorized version already verified
numerically equivalent (see thesis chat log for the cross-check:
identical prices, 2.4x-4.8x faster depending on grid size).

Slider max ranges for Mx/Mv/N are kept slightly lower here than the
original dashboard to keep worst-case Run time reasonable in production.
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_floating import npi_uniform as npi, mc_naive

dash.register_page(__name__, path="/npi-floating",
                    name="6.6 NPI (Floating, fixed grid)")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
GRAY, DGRAY, BG, DARK, LGRAY = "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"
GREEN = "#1D9E75"

PID = "npif6-"

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
        html.H1("Numerical Path Integral — Floating Barrier", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P("Deterministic density propagation on augmented state space (x, v) "
               "\u2014 vectorized implementation",
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
                  ("Steps N", "n", 10, 100, 5, 50),
                  ("Grid Mx", "mx", 50, 150, 10, 100),
                  ("Grid Mv", "mv", 30, 100, 10, 70),
              ]],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 14px"}),
            html.Button("\u25b6  Run NPI", id=f"{PID}run", n_clicks=0, style={
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
                          children=[dcc.Graph(id=f"{PID}g-heat", style={"height": "300px"},
                                               config={"displayModeBar": False})]),
            ]),
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-surv", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
                html.Div(style={**CARD, "flex": "1", "minWidth": "300px"},
                          children=[dcc.Graph(id=f"{PID}g-conv", style={"height": "280px"},
                                               config={"displayModeBar": False})]),
            ]),
        ]),
    ]),

    html.P("S\u2080=100, q=0. NPI and MC use same N for fair comparison.",
           style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}store"),
])


@callback(
    Output(f"{PID}alpha-val", "children"), Output(f"{PID}k-val", "children"),
    Output(f"{PID}sig-val", "children"), Output(f"{PID}r-val", "children"),
    Output(f"{PID}t-val", "children"), Output(f"{PID}n-val", "children"),
    Output(f"{PID}mx-val", "children"), Output(f"{PID}mv-val", "children"),
    Input(f"{PID}alpha", "value"), Input(f"{PID}k", "value"), Input(f"{PID}sig", "value"),
    Input(f"{PID}r", "value"), Input(f"{PID}t", "value"), Input(f"{PID}n", "value"),
    Input(f"{PID}mx", "value"), Input(f"{PID}mv", "value"),
)
def lbl(a, k, s, r, t, n, mx, mv):
    return f"{a/100:.2f}", str(k), f"{s}%", f"{r}%", f"{t/100:.2f}y", str(n), str(mx), str(mv)


@callback(
    Output(f"{PID}store", "data"), Output(f"{PID}res", "children"),
    Input(f"{PID}run", "n_clicks"),
    State(f"{PID}alpha", "value"), State(f"{PID}k", "value"), State(f"{PID}sig", "value"),
    State(f"{PID}r", "value"), State(f"{PID}t", "value"), State(f"{PID}n", "value"),
    State(f"{PID}mx", "value"), State(f"{PID}mv", "value"),
    prevent_initial_call=False,
)
def run(nc, alpha_pct, K, sig_pct, r_pct, t_sc, N, Mx, Mv):
    S0 = 100.0; alpha = alpha_pct / 100; K = float(K)
    sigma = sig_pct / 100; r = r_pct / 100; T = t_sc / 100

    res = npi(S0, K, alpha, r, sigma, T, N_steps=N, Mx=Mx, Mv=Mv)

    # Alpha sweep
    alphas = np.linspace(0.50, 0.97, 15)
    npi_p, npi_s, mc_p, mc_s = [], [], [], []
    for a in alphas:
        r1 = npi(S0, K, a, r, sigma, T, N_steps=N, Mx=Mx, Mv=Mv)
        r2 = mc_naive(S0, K, a, r, sigma, T, N=N)
        npi_p.append(r1["do_put"]); npi_s.append(r1["surv"])
        mc_p.append(r2["do_put"]); mc_s.append(r2["surv"])

    # N sweep
    N_vals = [10, 25, 50, 100]
    npi_conv, mc_conv = [], []
    for nv in N_vals:
        r1 = npi(S0, K, alpha, r, sigma, T, N_steps=nv, Mx=Mx, Mv=Mv)
        r2 = mc_naive(S0, K, alpha, r, sigma, T, N=nv)
        npi_conv.append(r1["do_put"]); mc_conv.append(r2["do_put"])

    G_N = res["G_N"]; x_grid = res["x_grid"]; v_grid = res["v_grid"]; b = res["b"]

    store = {
        "alphas": alphas.tolist(), "npi_p": npi_p, "npi_s": npi_s,
        "mc_p": mc_p, "mc_s": mc_s,
        "N_vals": N_vals, "npi_conv": npi_conv, "mc_conv": mc_conv,
        "G_N": G_N.tolist(), "x_grid": x_grid.tolist(),
        "v_grid": v_grid.tolist(), "b": float(b),
        "res": {"do_put": res["do_put"], "do_call": res["do_call"],
                "di_call": res["di_call"], "c_bs": res["c_bs"],
                "surv": res["surv"], "mass": res["surv"]},
        "alpha": alpha, "N": N, "Mx": Mx, "Mv": Mv,
    }

    panel = html.Div([
        html.P("NPI PRICES", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                     "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lb, style=LABEL),
                    html.Span(f"{vl:.4f}", style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, vl, col in [
              ("DO Put", res["do_put"], BLUE),
              ("DO Call", res["do_call"], FUCHSIA),
              ("DI Call", res["di_call"], LBLUE),
              ("BS Call", res["c_bs"], DGRAY),
              ("Survival", res["surv"], GREEN),
              ("Mass \u2211G\u00b7dx\u00b7dv", res["surv"], GREEN),
          ]],
    ])
    return store, panel


@callback(
    Output(f"{PID}g-price", "figure"), Output(f"{PID}g-heat", "figure"),
    Output(f"{PID}g-surv", "figure"), Output(f"{PID}g-conv", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]; npi_p = d["npi_p"]; mc_p = d["mc_p"]
    npi_s = d["npi_s"]; mc_s = d["mc_s"]
    N_vals = d["N_vals"]; npi_conv = d["npi_conv"]; mc_conv = d["mc_conv"]
    G_N = np.array(d["G_N"]); x_grid = np.array(d["x_grid"])
    v_grid = np.array(d["v_grid"]); b = d["b"]

    # ── NPI vs MC price vs alpha ─────────────────────────────────────────
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=npi_p, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f1.add_trace(go.Scatter(x=alphas, y=mc_p, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b1={d['alpha']:.2f}", annotation_font_size=10)
    f1.update_layout(**BASE, title=dict(text="DO Put: NPI vs MC Naive vs \u03b1", font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1"); f1.update_yaxes(title_text="DO Put price")

    # ── Terminal density heatmap ─────────────────────────────────────────
    f2 = go.Figure(data=go.Heatmap(
        z=G_N.T, x=x_grid.tolist(), y=v_grid.tolist(),
        colorscale="Blues", showscale=True,
        colorbar=dict(title=dict(text="G_N(x,v)", font=dict(size=11)))))
    f2.add_hline(y=b, line_color=FUCHSIA, line_width=2, line_dash="dash",
                 annotation_text=f"b={b:.3f}", annotation_font_color=FUCHSIA, annotation_font_size=10)
    f2.add_vline(x=float(np.log(100)), line_color=DGRAY, line_width=1.5, line_dash="dot",
                 annotation_text="ln S\u2080", annotation_font_size=10)
    f2.update_layout(**{k: v for k, v in BASE.items() if k != "hovermode"},
        title=dict(text="Terminal Density G_N(x,v)", font=dict(size=12)), hovermode="closest")
    f2.update_xaxes(title_text="Log-price x = ln S")
    f2.update_yaxes(title_text="Drawdown v = a \u2212 x")

    # ── Survival vs alpha ────────────────────────────────────────────────
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=alphas, y=npi_s, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines",
        fill="tozeroy", fillcolor="rgba(24,95,165,0.07)"))
    f3.add_trace(go.Scatter(x=alphas, y=mc_s, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f3.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1)
    f3.update_layout(**BASE, title=dict(text="Survival Probability vs \u03b1", font=dict(size=12)))
    f3.update_xaxes(title_text="\u03b1"); f3.update_yaxes(title_text="Survival prob", range=[0, 1])

    # ── NPI vs MC convergence in N ────────────────────────────────────────
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=N_vals, y=npi_conv, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=7)))
    f4.add_trace(go.Scatter(x=N_vals, y=mc_conv, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines+markers",
        marker=dict(size=7, symbol="square")))
    f4.add_vline(x=d["N"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"N={d['N']}", annotation_font_size=10)
    f4.update_layout(**BASE, title=dict(text="NPI vs MC Convergence in N", font=dict(size=12)))
    f4.update_xaxes(title_text="N steps"); f4.update_yaxes(title_text="DO Put price")

    return f1, f2, f3, f4
