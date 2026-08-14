"""
pages/07_comparison.py — Dashboard 7: Method Comparison.

Adapted from the original standalone dashboard
(Dashboard_Comparison.py, port 8056). Compares NPI, MC Naive and MC BB
on the same interactive panel.

IMPORTANT CHANGE vs the original: this is the heaviest of all the
dashboards — the original made ~20 calls to the slow pure-Python NPI
loop per single "Run Comparison" click (15 alpha-sweep + 4 N-sweep + 1
current point). It now uses pricing_floating.npi_uniform(), the
vectorized version verified numerically equivalent to the original
(see thesis chat log). Slider max ranges for Mx/Mv/N/M are also kept
lower than the original to keep worst-case Run time reasonable on
shared/free hosting CPUs.
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_floating import npi_uniform as npi, mc_naive, mc_bb

dash.register_page(__name__, path="/comparison", name="7. Method Comparison")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
GREEN, GRAY, DGRAY, BG, DARK, LGRAY = \
    "#1D9E75", "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF"

PID = "cmp7-"

LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif"}
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
        html.H1("Method Comparison — Floating Barrier", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P(
            "B\u209c = \u03b1 \u00b7 max S\u1d64   \u2014   NPI vs MC Naive vs MC BB "
            "\u2014 vectorized NPI implementation",
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
                  ("Barrier \u03b1", "alpha", 50, 98, 1, 80),
                  ("Strike K", "k", 80, 120, 1, 100),
                  ("Volatility \u03c3", "sig", 5, 60, 1, 20),
                  ("Risk-free r", "r", 0, 15, 1, 5),
                  ("Maturity T (yr)", "t", 25, 200, 5, 100),
                  ("Steps N", "n", 10, 100, 10, 50),
                  ("Paths M (\u00d71k)", "m", 5, 50, 5, 20),
                  ("Grid Mx", "mx", 50, 150, 10, 100),
                  ("Grid Mv", "mv", 30, 100, 10, 70),
              ]],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 14px"}),
            html.Button("\u25b6  Run Comparison", id=f"{PID}run", n_clicks=0, style={
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
                          children=[dcc.Graph(id=f"{PID}g-alpha", style={"height": "300px"},
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
                          children=[dcc.Graph(id=f"{PID}g-heat", style={"height": "280px"},
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
    Output(f"{PID}m-val", "children"), Output(f"{PID}mx-val", "children"),
    Output(f"{PID}mv-val", "children"),
    Input(f"{PID}alpha", "value"), Input(f"{PID}k", "value"), Input(f"{PID}sig", "value"),
    Input(f"{PID}r", "value"), Input(f"{PID}t", "value"), Input(f"{PID}n", "value"),
    Input(f"{PID}m", "value"), Input(f"{PID}mx", "value"), Input(f"{PID}mv", "value"),
)
def lbl(a, k, s, r, t, n, m, mx, mv):
    return f"{a/100:.2f}", str(k), f"{s}%", f"{r}%", f"{t/100:.2f}y", str(n), f"{m}k", str(mx), str(mv)


@callback(
    Output(f"{PID}store", "data"), Output(f"{PID}res", "children"),
    Input(f"{PID}run", "n_clicks"),
    State(f"{PID}alpha", "value"), State(f"{PID}k", "value"), State(f"{PID}sig", "value"),
    State(f"{PID}r", "value"), State(f"{PID}t", "value"), State(f"{PID}n", "value"),
    State(f"{PID}m", "value"), State(f"{PID}mx", "value"), State(f"{PID}mv", "value"),
    prevent_initial_call=False,
)
def run(nc, alpha_pct, K, sig_pct, r_pct, t_sc, N, M_k, Mx, Mv):
    S0 = 100.0; alpha = alpha_pct / 100; K = float(K)
    sigma = sig_pct / 100; r = r_pct / 100; T = t_sc / 100; M = M_k * 1000

    r_npi = npi(S0, K, alpha, r, sigma, T, N_steps=N, Mx=Mx, Mv=Mv)
    r_mn = mc_naive(S0, K, alpha, r, sigma, T, N=N, M=M)
    r_mb = mc_bb(S0, K, alpha, r, sigma, T, N=N, M=M)

    # Alpha sweep
    alphas = np.linspace(0.50, 0.97, 15)
    npi_p, mn_p, mb_p, mn_se = [], [], [], []
    npi_s, mn_s, mb_s = [], [], []
    for a in alphas:
        r1 = npi(S0, K, a, r, sigma, T, N_steps=N, Mx=Mx, Mv=Mv)
        r2 = mc_naive(S0, K, a, r, sigma, T, N=N, M=M)
        r3 = mc_bb(S0, K, a, r, sigma, T, N=N, M=M)
        npi_p.append(r1["do_put"]); mn_p.append(r2["do_put"])
        mb_p.append(r3["do_put"]); mn_se.append(r2["se_put"])
        npi_s.append(r1["surv"]); mn_s.append(r2["surv"])
        mb_s.append(r3["surv"])

    # N sweep
    N_vals = [10, 25, 50, 100]
    npi_n, mn_n, mb_n, mn_se_n = [], [], [], []
    for nv in N_vals:
        r1 = npi(S0, K, alpha, r, sigma, T, N_steps=nv, Mx=Mx, Mv=Mv)
        r2 = mc_naive(S0, K, alpha, r, sigma, T, N=nv, M=M)
        r3 = mc_bb(S0, K, alpha, r, sigma, T, N=nv, M=M)
        npi_n.append(r1["do_put"]); mn_n.append(r2["do_put"])
        mb_n.append(r3["do_put"]); mn_se_n.append(r2["se_put"])

    G_N = r_npi["G_N"]; x_grid = r_npi["x_grid"]
    v_grid = r_npi["v_grid"]; b = r_npi["b"]

    store = {
        "alphas": alphas.tolist(),
        "npi_p": npi_p, "mn_p": mn_p, "mb_p": mb_p, "mn_se": mn_se,
        "npi_s": npi_s, "mn_s": mn_s, "mb_s": mb_s,
        "N_vals": N_vals, "npi_n": npi_n, "mn_n": mn_n, "mb_n": mb_n, "mn_se_n": mn_se_n,
        "G_N": G_N.tolist(), "x_grid": x_grid.tolist(),
        "v_grid": v_grid.tolist(), "b": float(b),
        "npi": r_npi, "mn": r_mn, "mb": r_mb,
        "alpha": alpha, "N": N, "sigma_pct": sig_pct,
    }

    c_bs = r_npi["c_bs"]
    parity_ok = abs(r_npi["do_call"] + r_npi["di_call"] - c_bs) < 0.01
    panel = html.Div([
        html.P("CURRENT PRICES", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                         "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lb, style=LABEL),
                    html.Span(f"{vl:.4f}", style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, vl, col in [
              ("NPI Put", r_npi["do_put"], BLUE),
              ("MC Naive", r_mn["do_put"], FUCHSIA),
              ("\u00b1 SE", r_mn["se_put"], GRAY),
              ("MC BB", r_mb["do_put"], GREEN),
              ("BS Call", c_bs, DGRAY),
              ("NPI Surv", r_npi["surv"], BLUE),
              ("MC Surv", r_mn["surv"], FUCHSIA),
          ]],
        html.Hr(style={"borderColor": LGRAY, "margin": "8px 0"}),
        html.Div([
            html.Span("Parity", style=LABEL),
            html.Span("\u2713" if parity_ok else "\u2717",
                      style={**VALUE, "color": "#1D9E75" if parity_ok else "#E24B4A"})
        ], style={"display": "flex", "justifyContent": "space-between"}),
    ])
    return store, panel


@callback(
    Output(f"{PID}g-alpha", "figure"), Output(f"{PID}g-surv", "figure"),
    Output(f"{PID}g-conv", "figure"), Output(f"{PID}g-heat", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]
    npi_p = d["npi_p"]; mn_p = d["mn_p"]; mb_p = d["mb_p"]; mn_se = d["mn_se"]
    npi_s = d["npi_s"]; mn_s = d["mn_s"]; mb_s = d["mb_s"]
    N_vals = d["N_vals"]; npi_n = d["npi_n"]; mn_n = d["mn_n"]
    mb_n = d["mb_n"]; mn_se_n = d["mn_se_n"]
    G_N = np.array(d["G_N"]); x_grid = np.array(d["x_grid"])
    v_grid = np.array(d["v_grid"]); b = d["b"]

    # ── Price vs alpha ───────────────────────────────────────────────────
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=npi_p, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines"))
    f1.add_trace(go.Scatter(x=alphas, y=mn_p, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f1.add_trace(go.Scatter(
        x=alphas + alphas[::-1],
        y=[p + 1.96 * s for p, s in zip(mn_p, mn_se)] +
          [p - 1.96 * s for p, s in zip(mn_p, mn_se)][::-1],
        fill="toself", fillcolor="rgba(212,83,126,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="MC 95% CI"))
    f1.add_trace(go.Scatter(x=alphas, y=mb_p, name="MC BB",
        line=dict(color=GREEN, width=2.0, dash="dot"), mode="lines"))
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b1={d['alpha']:.2f}", annotation_font_size=10)
    f1.update_layout(**BASE, title=dict(text="DO Put vs \u03b1", font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1"); f1.update_yaxes(title_text="DO Put price")

    # ── Survival vs alpha ────────────────────────────────────────────────
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=alphas, y=npi_s, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines",
        fill="tozeroy", fillcolor="rgba(24,95,165,0.07)"))
    f2.add_trace(go.Scatter(x=alphas, y=mn_s, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines"))
    f2.add_trace(go.Scatter(x=alphas, y=mb_s, name="MC BB",
        line=dict(color=GREEN, width=2.0, dash="dot"), mode="lines"))
    f2.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1)
    f2.update_layout(**BASE, title=dict(text="Survival Probability vs \u03b1", font=dict(size=12)))
    f2.update_xaxes(title_text="\u03b1"); f2.update_yaxes(title_text="Survival", range=[0, 1])

    # ── Convergence in N ────────────────────────────────────────────────
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=N_vals, y=npi_n, name="NPI",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=7)))
    f3.add_trace(go.Scatter(x=N_vals, y=mn_n, name="MC Naive",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines+markers",
        marker=dict(size=6, symbol="square")))
    f3.add_trace(go.Scatter(
        x=N_vals + N_vals[::-1],
        y=[p + 1.96 * s for p, s in zip(mn_n, mn_se_n)] +
          [p - 1.96 * s for p, s in zip(mn_n, mn_se_n)][::-1],
        fill="toself", fillcolor="rgba(212,83,126,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    f3.add_trace(go.Scatter(x=N_vals, y=mb_n, name="MC BB",
        line=dict(color=GREEN, width=2.0, dash="dot"), mode="lines+markers",
        marker=dict(size=6, symbol="triangle-up")))
    f3.add_vline(x=d["N"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"N={d['N']}", annotation_font_size=10)
    f3.update_layout(**BASE, title=dict(text="DO Put Convergence in N", font=dict(size=12)))
    f3.update_xaxes(title_text="N steps"); f3.update_yaxes(title_text="DO Put price")

    # ── NPI terminal density heatmap ────────────────────────────────────
    f4 = go.Figure(data=go.Heatmap(
        z=G_N.T, x=x_grid.tolist(), y=v_grid.tolist(),
        colorscale="Blues", showscale=True,
        colorbar=dict(title=dict(text="G_N(x,v)", font=dict(size=11)))))
    f4.add_hline(y=b, line_color=FUCHSIA, line_width=2, line_dash="dash",
                 annotation_text=f"b={b:.3f}", annotation_font_color=FUCHSIA, annotation_font_size=10)
    f4.add_vline(x=float(np.log(100)), line_color=DGRAY, line_width=1.5, line_dash="dot",
                 annotation_text="ln S\u2080", annotation_font_size=10)
    f4.update_layout(**{k: v for k, v in BASE.items() if k != "hovermode"},
        title=dict(text="Terminal Density G_N(x,v)", font=dict(size=12)), hovermode="closest")
    f4.update_xaxes(title_text="Log-price x = ln S")
    f4.update_yaxes(title_text="Drawdown v = a \u2212 x")

    return f1, f2, f3, f4
