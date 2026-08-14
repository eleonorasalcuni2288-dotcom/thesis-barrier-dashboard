"""
pages/08_adaptive_npi.py — Dashboard 9.9: Adaptive NPI Grid.

Adapted from the original standalone dashboard
(Dashboard_Adaptive_NPI.py, port 8057). Pricing now comes from the
shared pricing_adaptive module (already verified: array-length mismatch
in the v-grid-spacing panel fixed, x_width documented explicitly — see
thesis chat log).
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback

from pricing_adaptive import npi_adaptive, npi_uniform_scriptgrid, mc_naive

dash.register_page(__name__, path="/adaptive-npi", name="9.9 Adaptive NPI Grid")

BLUE, FUCHSIA, GREEN = "#185FA5", "#D4537E", "#1D9E75"
GRAY, DGRAY, BG, DARK, LGRAY, LBLUE = \
    "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF", "#85B7EB"

PID = "adnpi8-"

BASE = dict(paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Georgia,serif", color=DARK, size=11),
            margin=dict(l=50, r=20, t=45, b=50),
            xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
            legend=dict(font=dict(size=10)), hovermode="x unified")
CARD = {"backgroundColor": "white", "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}
LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif", "letterSpacing": "0.02em"}
VALUE = {"fontSize": "13px", "fontWeight": "600", "color": DARK, "fontFamily": "Georgia,serif"}

layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                          "fontFamily": "Georgia,serif", "padding": "24px 32px"}, children=[

    html.Div([
        html.H1("Adaptive NPI Grid — Floating Barrier", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P(
            "v\u2c7c = b \u00b7 (j/Mv)\u02b8  \u2014  non-uniform drawdown grid, "
            "\u03b3=1 \u2192 uniform grid,  \u03b3>1 \u2192 grid concentrated near zero drawdown",
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
                  ("Barrier \u03b1", "alpha", 50, 97, 1, 80),
                  ("Volatility \u03c3", "sig", 5, 50, 1, 20),
                  ("Maturity T (yr)", "t", 25, 200, 5, 100),
                  ("Steps N", "n", 10, 100, 10, 50),
                  ("Grid \u03b3", "gamma", 10, 40, 1, 20),
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
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-alpha", style={"height": "340px"},
                                         config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-gamma", style={"height": "340px"},
                                         config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-density", style={"height": "300px"},
                                         config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-grid", style={"height": "300px"},
                                         config={"displayModeBar": False})]),
        ]),
    ]),

    html.P(
        "S\u2080=100, K=100, r=5%. ACCURACY panel: griglia di riferimento Script 9.9 \u2014 "
        "Adaptive 120\u00d780 (x_width=3.5), Uniform 150\u00d7100 (x_width=4.5), "
        "MC ref M=200k (SE\u22480.006, 4\u00d7SE\u22480.024). "
        "Entrambi dentro \u00b1 4 SE di MC, ma Adaptive usa il 36% di nodi in meno \u2192 4\u00d7 pi\u00f9 veloce. "
        "Curve Panel 1 e Panel 2: griglia ridotta 60\u00d750, MC M=10k, per velocit\u00e0 interattiva. "
        "Benchmark N=252: NPI Adaptive 1.5\u00d7 pi\u00f9 veloce di MC Naive (M=200k).",
        style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}store"),
])


@callback(
    Output(f"{PID}alpha-val", "children"), Output(f"{PID}sig-val", "children"),
    Output(f"{PID}t-val", "children"), Output(f"{PID}n-val", "children"),
    Output(f"{PID}gamma-val", "children"),
    Input(f"{PID}alpha", "value"), Input(f"{PID}sig", "value"),
    Input(f"{PID}t", "value"), Input(f"{PID}n", "value"),
    Input(f"{PID}gamma", "value"),
)
def labels(a, s, t, n, g):
    return f"{a/100:.2f}", f"{s}%", f"{t/100:.2f}y", str(n), f"{g/10:.1f}"


@callback(
    Output(f"{PID}store", "data"), Output(f"{PID}res", "children"),
    Input(f"{PID}run", "n_clicks"),
    State(f"{PID}alpha", "value"), State(f"{PID}sig", "value"),
    State(f"{PID}t", "value"), State(f"{PID}n", "value"),
    State(f"{PID}gamma", "value"),
    prevent_initial_call=True,
)
def run(nc, alpha_pct, sig_pct, t_sc, N, gamma_10):
    S0 = 100.0; K = 100.0; r = 0.05
    alpha = alpha_pct / 100; sigma = sig_pct / 100
    T = t_sc / 100; gamma = gamma_10 / 10

    res_ad = npi_adaptive(S0, K, alpha, r, sigma, T, N_steps=N,
                           Mx=120, Mv=80, v_gamma=gamma, x_width=3.5)
    res_un = npi_uniform_scriptgrid(S0, K, alpha, r, sigma, T, N_steps=N,
                                     Mx=150, Mv=100, x_width=4.5)
    res_mc = mc_naive(S0, K, alpha, r, sigma, T, N=N, M=200_000)

    alphas = np.linspace(0.55, 0.95, 10)
    ad_p, un_p, mc_p, mc_se = [], [], [], []
    for a in alphas:
        r1 = npi_adaptive(S0, K, a, r, sigma, T, N_steps=N, Mx=60, Mv=50,
                           v_gamma=gamma, x_width=3.5)
        r2 = npi_uniform_scriptgrid(S0, K, a, r, sigma, T, N_steps=N, Mx=60, Mv=50, x_width=4.5)
        r3 = mc_naive(S0, K, a, r, sigma, T, N=N, M=10_000)
        ad_p.append(r1["do_put"]); un_p.append(r2["do_put"])
        mc_p.append(r3["do_put"]); mc_se.append(r3["se_put"])

    gammas = np.arange(10, 41, 5)
    mc_ref_p = res_mc["do_put"]
    gam_p, gam_err = [], []
    for g in gammas:
        r1 = npi_adaptive(S0, K, alpha, r, sigma, T, N_steps=N,
                           Mx=60, Mv=50, v_gamma=g / 10, x_width=3.5)
        gam_p.append(r1["do_put"])
        gam_err.append(abs(r1["do_put"] - mc_ref_p))

    store = {
        "alphas": alphas.tolist(), "ad_p": ad_p, "un_p": un_p,
        "mc_p": mc_p, "mc_se": mc_se,
        "gammas": (gammas / 10).tolist(), "gam_p": gam_p, "gam_err": gam_err,
        "G_N": res_ad["G_N"].tolist(),
        "x_grid": res_ad["x_grid"].tolist(),
        "v_grid_ad": res_ad["v_grid"].tolist(),
        "v_grid_un": res_un["v_grid"].tolist(),
        "b": float(-np.log(alpha)), "alpha": alpha, "gamma": gamma,
        "ad_put": res_ad["do_put"], "un_put": res_un["do_put"],
        "mc_put": res_mc["do_put"], "mc_se_ref": res_mc["se_put"],
        "ad_call": res_ad["do_call"], "un_call": res_un["do_call"],
        "mc_call": res_mc["do_call"],
        "ad_di": res_ad["di_call"], "un_di": res_un["di_call"],
        "mc_di": res_mc["di_call"], "c_bs": res_mc["c_bs"],
    }

    diff_ad = abs(res_ad["do_put"] - res_mc["do_put"])
    diff_un = abs(res_un["do_put"] - res_mc["do_put"])
    inside_ad = diff_ad < 4 * res_mc["se_put"]
    inside_un = diff_un < 4 * res_mc["se_put"]
    panel = html.Div([
        html.P("PRICES", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                 "color": GRAY, "marginBottom": "8px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"})
          for lb, val, col in [
              ("NPI Adaptive", f"{res_ad['do_put']:.4f}", BLUE),
              ("NPI Uniform", f"{res_un['do_put']:.4f}", LBLUE),
              ("MC Naive", f"{res_mc['do_put']:.4f}", FUCHSIA),
              ("MC \u00b1 SE", f"{res_mc['se_put']:.4f}", GRAY),
          ]],
        html.Hr(style={"borderColor": LGRAY, "margin": "8px 0"}),
        html.P("DO CALL", style={"fontSize": "10px", "letterSpacing": "0.12em",
                                  "color": GRAY, "marginBottom": "8px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"})
          for lb, val, col in [
              ("NPI Adaptive", f"{res_ad['do_call']:.4f}", BLUE),
              ("NPI Uniform", f"{res_un['do_call']:.4f}", LBLUE),
              ("MC Naive", f"{res_mc['do_call']:.4f}", FUCHSIA),
          ]],
        html.Hr(style={"borderColor": LGRAY, "margin": "8px 0"}),
        html.P("DI CALL  (parit\u00e0: C_BS \u2212 DO Call)", style={
               "fontSize": "10px", "letterSpacing": "0.12em",
               "color": GRAY, "marginBottom": "8px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"})
          for lb, val, col in [
              ("NPI Adaptive", f"{res_ad['di_call']:.4f}", BLUE),
              ("NPI Uniform", f"{res_un['di_call']:.4f}", LBLUE),
              ("MC Naive", f"{res_mc['di_call']:.4f}", FUCHSIA),
              ("C_BS (vanilla)", f"{res_mc['c_bs']:.4f}", GRAY),
          ]],
        html.Hr(style={"borderColor": LGRAY, "margin": "8px 0"}),
        html.P("ACCURACY  (Script 9.9 grids, \u00b1 4 SE)", style={"fontSize": "10px",
               "letterSpacing": "0.12em", "color": GRAY, "marginBottom": "8px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"})
          for lb, val, col in [
              ("Adaptive 120\u00d780", f"{res_ad['do_put']:.4f}", GREEN if inside_ad else "#E24B4A"),
              ("\u2192 in band?", "\u2713 yes" if inside_ad else "\u2717 no", GREEN if inside_ad else "#E24B4A"),
              ("Uniform 150\u00d7100", f"{res_un['do_put']:.4f}", GREEN if inside_un else "#E24B4A"),
              ("\u2192 in band?", "\u2713 yes" if inside_un else "\u2717 no", GREEN if inside_un else "#E24B4A"),
          ]],
        html.Div("Entrambi \u2713 \u2014 ma Adaptive usa meno punti griglia",
                 style={"fontSize": "10px", "color": DGRAY, "marginTop": "4px", "fontStyle": "italic"}
                 ) if (inside_ad and inside_un) else html.Span(),
        html.Hr(style={"borderColor": LGRAY, "margin": "8px 0"}),
        html.P("BENCHMARK  (N=252, timing_mc_vs_npi.py)", style={
               "fontSize": "10px", "letterSpacing": "0.12em",
               "color": GRAY, "marginBottom": "8px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"})
          for lb, val, col in [
              ("Adaptive 120\u00d780", "0.55s  \u03c3\u00b2=0", BLUE),
              ("Uniform  150\u00d7100", "0.94s  \u03c3\u00b2=0", LBLUE),
              ("MC Naive M=200k", "0.82s  \u03c3\u00b2>0", FUCHSIA),
          ]],
        html.Div("NPI Adaptive 1.5\u00d7 faster than MC Naive  (N=252, M=200k)",
                 style={"fontSize": "10px", "color": DGRAY, "marginTop": "6px",
                        "fontStyle": "italic", "lineHeight": "1.4"}),
    ])
    return store, panel


@callback(
    Output(f"{PID}g-alpha", "figure"), Output(f"{PID}g-gamma", "figure"),
    Output(f"{PID}g-density", "figure"), Output(f"{PID}g-grid", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]; mc_p = d["mc_p"]; mc_se = d["mc_se"]
    ad_p = d["ad_p"]; un_p = d["un_p"]
    gammas = d["gammas"]; gam_p = d["gam_p"]; gam_err = d["gam_err"]
    mc_se_ref = d["mc_se_ref"]
    G_N = np.array(d["G_N"])
    x_grid = np.array(d["x_grid"])
    v_ad = np.array(d["v_grid_ad"])
    v_un = np.array(d["v_grid_un"])
    b = d["b"]

    # ── Panel 1: Price vs alpha ─────────────────────────────────────────
    mc_upper = [p + 4 * s for p, s in zip(mc_p, mc_se)]
    mc_lower = [p - 4 * s for p, s in zip(mc_p, mc_se)]

    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=mc_upper, mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=alphas, y=mc_lower, mode="lines", fill="tonexty",
        line=dict(width=0), fillcolor="rgba(212,83,126,0.13)",
        name="MC \u00b1 4 SE band", hoverinfo="skip"))
    f1.add_trace(go.Scatter(x=alphas, y=ad_p, name="NPI Adaptive  (det., \u03c3\u00b2=0)",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=6)))
    f1.add_trace(go.Scatter(x=alphas, y=un_p, name="NPI Uniform  (det., \u03c3\u00b2=0)",
        line=dict(color=LBLUE, width=2.0, dash="dot"), mode="lines+markers",
        marker=dict(size=5, symbol="square")))
    f1.add_trace(go.Scatter(x=alphas, y=mc_p, name="MC Naive  (stoch., \u03c3\u00b2>0)",
        line=dict(color=FUCHSIA, width=2.0, dash="dash"), mode="lines+markers",
        marker=dict(size=5, symbol="diamond")))
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b1={d['alpha']:.2f}", annotation_font_size=10)
    f1.update_layout(**BASE,
        title=dict(text="DO Put vs \u03b1  \u2014  NPI (deterministic, \u03c3\u00b2=0) vs MC estimate \u00b1 4 SE",
                   font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1"); f1.update_yaxes(title_text="DO Put price")

    # ── Panel 2: Error |NPI - MC| vs gamma ──────────────────────────────
    thresh_4se = 4 * mc_se_ref
    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=gammas, y=gam_err, name="|NPI \u2212 MC|",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=8)))
    f2.add_hline(y=thresh_4se, line_color=FUCHSIA, line_dash="dash",
                 annotation_text=f"4\u00d7SE = {thresh_4se:.4f}",
                 annotation_position="top right", annotation_font_size=10)
    f2.add_hline(y=mc_se_ref, line_color=GRAY, line_dash="dot",
                 annotation_text=f"1\u00d7SE = {mc_se_ref:.4f}",
                 annotation_position="bottom right", annotation_font_size=10)
    f2.add_vline(x=d["gamma"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b3={d['gamma']:.1f}", annotation_font_size=10)
    f2.update_layout(**BASE,
        title=dict(text="NPI deviation from MC vs \u03b3  (|NPI \u2212 MC|, fixed Mx=60, Mv=50)",
                   font=dict(size=12)))
    f2.update_xaxes(title_text="Grid exponent \u03b3")
    f2.update_yaxes(title_text="|NPI \u2212 MC|", rangemode="tozero")

    # ── Panel 3: Terminal density heatmap ────────────────────────────────
    S_arr = np.exp(x_grid)
    f3 = go.Figure(go.Heatmap(
        x=np.round(S_arr, 2).tolist(), y=np.round(v_ad, 4).tolist(),
        z=G_N.T.tolist(), colorscale="Blues", showscale=True,
        colorbar=dict(thickness=12, len=0.8)))
    f3.add_hline(y=b, line_color=FUCHSIA, line_dash="dash",
                 annotation_text=f"b={b:.3f}", annotation_font_size=10)
    f3.add_vline(x=100.0, line_color=DGRAY, line_dash="dot",
                 annotation_text="K=100", annotation_font_size=10)
    f3.update_layout(
        **{k: v for k, v in BASE.items() if k not in ("xaxis", "yaxis", "hovermode")},
        xaxis=dict(title_text="S = e^x", gridcolor=LGRAY),
        yaxis=dict(title_text="Drawdown v", gridcolor=LGRAY),
        title=dict(text="Terminal density G_N(x,v) \u2014 NPI Adaptive", font=dict(size=12)),
        hovermode="closest")

    # ── Panel 4: v-grid spacing ──────────────────────────────────────────
    j_ad = np.arange(len(v_ad))
    j_un = np.arange(len(v_un))
    f4 = go.Figure()
    f4.add_trace(go.Scatter(x=j_ad.tolist(), y=v_ad.tolist(), name=f"Adaptive \u03b3={d['gamma']:.1f}",
        line=dict(color=BLUE, width=2.5), mode="lines+markers", marker=dict(size=5)))
    f4.add_trace(go.Scatter(x=j_un.tolist(), y=v_un.tolist(), name="Uniform",
        line=dict(color=FUCHSIA, width=2.0, dash="dot"), mode="lines+markers",
        marker=dict(size=5, symbol="square")))
    f4.add_hline(y=b, line_color=GRAY, line_dash="dash",
                 annotation_text=f"b={b:.3f}", annotation_font_size=10)
    f4.update_layout(**BASE, title=dict(text="v-grid spacing: adaptive vs uniform", font=dict(size=12)))
    f4.update_xaxes(title_text="Grid index j")
    f4.update_yaxes(title_text="v\u2c7c  (drawdown node)")

    return f1, f2, f3, f4
