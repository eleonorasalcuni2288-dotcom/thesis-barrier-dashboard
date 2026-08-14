"""
pages/09_heston.py — Dashboard 10: Heston NPI 3D, floating barrier.

Adapted from the original standalone Heston dashboard (port 8058), which
already went through an extensive round of fixes verified in the thesis
chat log:
  - sigma_v/dv >= 3 stability check on the live coarse NPI, with a
    visible warning instead of a silently unreliable number.
  - Fixed columnwidth on the NPI table (previously clipped text).
  - More visible NPI marker on the alpha-sweep chart.
  - DI Call added to both MC and NPI (was being computed and discarded).
  - Fixed a 100x scaling bug in the running-max trace on the Sample
    Heston Paths panel (S(t) and its running max were on different
    scales).
  - Added a theta (long-run variance) sensitivity panel, previously
    missing despite theta having its own slider.
  - Reduced-resolution NPI curve (5 points) added to the alpha-sweep
    chart, in addition to the single current-alpha marker.
Pricing functions now come from the shared pricing_heston module.
"""

import time
import dash
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Input, Output, State, callback

from pricing_heston import (
    mc_heston, mc_naive_bs, heston_paths_sample, heston_npi_live,
    warmup_numba, MIN_RESOLUTION_RATIO,
)

dash.register_page(__name__, path="/heston", name="10. Heston NPI 3D")

warmup_numba()  # compile numba kernels once, at page-import time

BLUE, FUCHSIA, GREEN = "#185FA5", "#D4537E", "#1D9E75"
GRAY, DGRAY, BG, DARK, LGRAY, LBLUE, LGREEN = \
    "#B4B2A9", "#5F5E5A", "#F8F9FC", "#2C2C2A", "#E8E6DF", "#85B7EB", "#6CC5A0"
AMBER = "#C98A2A"

PID = "hest9-"

BASE = dict(paper_bgcolor=BG, plot_bgcolor=BG,
            font=dict(family="Georgia,serif", color=DARK, size=11),
            margin=dict(l=50, r=20, t=45, b=50),
            xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
            legend=dict(font=dict(size=10)), hovermode="x unified")
CARD = {"backgroundColor": "white", "borderRadius": "10px", "padding": "18px",
        "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}
LABEL = {"fontSize": "12px", "color": DGRAY, "fontFamily": "Georgia,serif", "letterSpacing": "0.02em"}
VALUE = {"fontSize": "13px", "fontWeight": "600", "color": DARK, "fontFamily": "Georgia,serif"}

MC_HESTON_REF = dict(do_put=0.5686, se=0.0048, M="200k", time="0.29s")

SLIDERS = [
    ("Mean reversion \u03ba", "kappa", 50, 500, 5, 200),
    ("Long-run var \u03b8", "theta", 10, 120, 2, 40),
    ("Vol-of-vol \u03be", "xi", 10, 100, 5, 30),
    ("Correlation \u03c1", "rho", -90, -5, 5, -70),
    ("Init var V\u2080", "v0", 10, 120, 2, 40),
    ("Barrier \u03b1", "alpha", 50, 97, 1, 80),
]

layout = html.Div(style={"backgroundColor": BG, "minHeight": "100vh",
                          "fontFamily": "Georgia,serif", "padding": "24px 32px"}, children=[

    html.Div([
        html.H1("Heston Model \u2014 Floating Barrier DO Put", style={
            "fontSize": "20px", "fontWeight": "600", "margin": "0 0 3px", "color": DARK}),
        html.P(
            "dS/S = r dt + \u221aV dW_S,   dV = \u03ba(\u03b8-V)dt + \u03be\u221aV dW_V,   "
            "\u03c1 = Corr(W_S, W_V)",
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
              for lbl, id_, mn, mx, st, vl in SLIDERS],
            html.Hr(style={"borderColor": LGRAY, "margin": "4px 0 14px"}),
            html.Button("\u25b6  Run", id=f"{PID}run", n_clicks=0, style={
                "width": "100%", "padding": "10px", "backgroundColor": GREEN,
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
                     children=[dcc.Graph(id=f"{PID}g-sens", style={"height": "340px"},
                                         config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-paths", style={"height": "380px"},
                                         config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1"},
                     children=[dcc.Graph(id=f"{PID}g-npi", style={"height": "340px"},
                                         config={"displayModeBar": False})]),
        ]),
    ]),

    html.P(
        "S\u2080=100, K=100, r=5%, T=1y, N=50, M=15k paths. "
        "NPI 3D medium/fine rows are precomputed from Script 10 (fixed at "
        "default params). NPI coarse curve is recomputed live at 5 \u03b1 points "
        "each Run (~15s total, coarse grid \u22483s/point). BS MC uses \u03c3=\u221a\u03b8. "
        "DO Call / DI Call under Heston are shown for reference, not as "
        "fully validated prices \u2014 see note under the NPI table.",
        style={"fontSize": "11px", "color": GRAY, "marginTop": "16px", "fontStyle": "italic"}),
    dcc.Store(id=f"{PID}store"),
])


@callback(
    Output(f"{PID}kappa-val", "children"), Output(f"{PID}theta-val", "children"),
    Output(f"{PID}xi-val", "children"), Output(f"{PID}rho-val", "children"),
    Output(f"{PID}v0-val", "children"), Output(f"{PID}alpha-val", "children"),
    Input(f"{PID}kappa", "value"), Input(f"{PID}theta", "value"),
    Input(f"{PID}xi", "value"), Input(f"{PID}rho", "value"),
    Input(f"{PID}v0", "value"), Input(f"{PID}alpha", "value"),
)
def labels(k, th, xi, rho, v0, a):
    return (f"{k/100:.2f}", f"{th/1000:.3f}", f"{xi/100:.2f}",
            f"{rho/100:.2f}", f"{v0/1000:.3f}", f"{a/100:.2f}")


@callback(
    Output(f"{PID}store", "data"), Output(f"{PID}res", "children"),
    Input(f"{PID}run", "n_clicks"),
    State(f"{PID}kappa", "value"), State(f"{PID}theta", "value"),
    State(f"{PID}xi", "value"), State(f"{PID}rho", "value"),
    State(f"{PID}v0", "value"), State(f"{PID}alpha", "value"),
    prevent_initial_call=False,
)
def run(nc, kappa_sc, theta_sc, xi_sc, rho_sc, v0_sc, alpha_pct):
    S0 = 100.0; K = 100.0; r = 0.05; T = 1.0
    kappa = kappa_sc / 100; theta = theta_sc / 1000
    xi = xi_sc / 100; rho = rho_sc / 100
    V0 = v0_sc / 1000; alpha = alpha_pct / 100
    sigma_bs = float(np.sqrt(theta))

    res = mc_heston(S0, K, alpha, r, kappa, theta, xi, rho, V0, T)
    res_bs = mc_naive_bs(S0, K, alpha, r, sigma_bs, T)

    t0_npi = time.perf_counter()
    npi = heston_npi_live(S0, K, alpha, r, kappa, theta, xi, rho, V0, T)
    npi_elapsed = time.perf_counter() - t0_npi

    # Alpha sweep (MC, 12 points)
    alphas = np.linspace(0.55, 0.95, 12)
    h_p, h_se, bs_p, bs_se = [], [], [], []
    for a in alphas:
        r1 = mc_heston(S0, K, a, r, kappa, theta, xi, rho, V0, T, M=10_000)
        r2 = mc_naive_bs(S0, K, a, r, sigma_bs, T, M=10_000)
        h_p.append(r1["do_put"]); h_se.append(r1["se_put"])
        bs_p.append(r2["do_put"]); bs_se.append(r2["se_put"])

    # NPI alpha sweep (reduced, 5 points)
    npi_sweep_alphas = sorted(set(
        [round(a, 4) for a in [0.60, 0.70, 0.80, 0.90]] + [round(alpha, 4)]))
    npi_sweep_a, npi_sweep_p = [], []
    for a in npi_sweep_alphas:
        npi_pt = npi if abs(a - alpha) < 1e-9 else \
            heston_npi_live(S0, K, a, r, kappa, theta, xi, rho, V0, T)
        npi_sweep_a.append(a); npi_sweep_p.append(npi_pt["do_put"])

    # Heston param sensitivity (fixed alpha=0.80)
    a80 = 0.80
    kappas = np.arange(50, 505, 50) / 100
    xis = np.arange(10, 105, 10) / 100
    rhos = np.arange(-90, -5, 10) / 100
    thetas = np.arange(10, 105, 10) / 1000
    kap_p, xi_p, rho_p, theta_p = [], [], [], []
    for k2 in kappas:
        kap_p.append(mc_heston(S0, K, a80, r, k2, theta, xi, rho, V0, T, M=8_000)["do_put"])
    for x2 in xis:
        xi_p.append(mc_heston(S0, K, a80, r, kappa, theta, x2, rho, V0, T, M=8_000)["do_put"])
    for r2 in rhos:
        rho_p.append(mc_heston(S0, K, a80, r, kappa, theta, xi, r2, V0, T, M=8_000)["do_put"])
    for th2 in thetas:
        theta_p.append(mc_heston(S0, K, a80, r, kappa, th2, xi, rho, V0, T, M=8_000)["do_put"])

    # Sample paths
    t_arr, S_paths, V_paths = heston_paths_sample(
        S0, kappa, theta, xi, rho, V0, r, T, n_paths=12, seed=7)

    store = {
        "alphas": alphas.tolist(), "h_p": h_p, "h_se": h_se,
        "bs_p": bs_p, "bs_se": bs_se,
        "kappas": kappas.tolist(), "kap_p": kap_p,
        "xis": xis.tolist(), "xi_p": xi_p,
        "rhos": rhos.tolist(), "rho_p": rho_p,
        "thetas": thetas.tolist(), "theta_p": theta_p,
        "t_arr": t_arr.tolist(), "S_paths": S_paths.tolist(), "V_paths": V_paths.tolist(),
        "alpha": alpha, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho, "V0": V0,
        "do_put": res["do_put"], "se_put": res["se_put"],
        "do_call": res["do_call"], "se_call": res["se_call"],
        "di_call": res["di_call"], "se_di_call": res["se_di_call"],
        "surv": res["surv"],
        "bs_put": res_bs["do_put"], "bs_se_ref": res_bs["se_put"],
        "npi_do_put": npi["do_put"], "npi_do_call": npi["do_call"],
        "npi_di_call": npi["di_call"],
        "npi_surv": npi["surv"], "npi_time": npi_elapsed,
        "npi_resolution_ratio": npi["resolution_ratio"], "npi_stable": npi["stable"],
        "npi_sweep_a": npi_sweep_a, "npi_sweep_p": npi_sweep_p,
    }

    npi_diff = npi["do_put"] - res["do_put"]
    npi_di_diff = npi["di_call"] - res["di_call"]
    c_vanilla_mc = res["do_call"] + res["di_call"]
    c_vanilla_npi = npi["do_call"] + npi["di_call"]

    warning_block = []
    if not npi["stable"]:
        warning_block = [
            html.Div([
                html.Span("\u26a0 ", style={"color": AMBER, "fontWeight": "700"}),
                html.Span(
                    f"Variance grid too coarse for these params "
                    f"(\u03c3\u1d65/dv = {npi['resolution_ratio']:.2f} < {MIN_RESOLUTION_RATIO:.0f}). "
                    f"NPI result below may be unreliable \u2014 increase \u03be, "
                    f"increase V\u2080, or treat this number with caution.",
                    style={"fontSize": "11px", "color": AMBER}),
            ], style={"backgroundColor": "#FBF1E0", "border": f"1px solid {AMBER}",
                       "borderRadius": "6px", "padding": "8px 10px", "marginBottom": "10px"})
        ]

    npi_put_color = BLUE if npi["stable"] else AMBER

    panel = html.Div([
        html.P("RESULTS \u2014 MC Heston (M=15k paths)", style={"fontSize": "10px",
               "letterSpacing": "0.12em", "color": GRAY, "marginBottom": "10px"}),
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, val, col in [
              ("DO Put", f"{res['do_put']:.4f}", GREEN),
              ("\u00b1SE", f"{res['se_put']:.4f}", GRAY),
              ("DO Call", f"{res['do_call']:.4f}", LGREEN),
              ("\u00b1SE (DO Call)", f"{res['se_call']:.4f}", GRAY),
              ("DI Call", f"{res['di_call']:.4f}", FUCHSIA),
              ("\u00b1SE (DI Call)", f"{res['se_di_call']:.4f}", GRAY),
              ("C_vanilla = DO+DI", f"{c_vanilla_mc:.4f}", DGRAY),
              ("Survival prob", f"{res['surv']:.4f}", DGRAY),
              ("\u03c3_BS = \u221a\u03b8", f"{sigma_bs:.4f}", GRAY),
              ("BS DO Put", f"{res_bs['do_put']:.4f}", BLUE),
          ]],
        html.Hr(style={"borderColor": LGRAY, "margin": "10px 0"}),
        html.P("NPI 3D  (coarse 121\u00d7151\u00d760)", style={"fontSize": "10px",
               "letterSpacing": "0.12em", "color": GRAY, "marginBottom": "8px"}),
        *warning_block,
        *[html.Div([html.Span(lb, style=LABEL), html.Span(val, style={**VALUE, "color": col})],
                   style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"})
          for lb, val, col in [
              ("NPI DO Put", f"{npi['do_put']:.4f}", npi_put_color),
              ("NPI DO Call", f"{npi['do_call']:.4f}", LBLUE),
              ("NPI DI Call", f"{npi['di_call']:.4f}", FUCHSIA),
              ("NPI C_vanilla", f"{c_vanilla_npi:.4f}", DGRAY),
              ("NPI Surv", f"{npi['surv']:.4f}", DGRAY),
              ("NPI \u2212 MC (Put)", f"{npi_diff:+.4f}",
                  GREEN if abs(npi_diff) < 4 * res["se_put"] else FUCHSIA),
              ("NPI \u2212 MC (DI Call)", f"{npi_di_diff:+.4f}",
                  GREEN if abs(npi_di_diff) < 4 * res["se_di_call"] else FUCHSIA),
              ("\u03c3\u1d65/dv ratio", f"{npi['resolution_ratio']:.2f}",
                  GREEN if npi["stable"] else AMBER),
              ("NPI time", f"{npi_elapsed:.1f}s", GRAY),
          ]],
    ])
    return store, panel


@callback(
    Output(f"{PID}g-alpha", "figure"), Output(f"{PID}g-sens", "figure"),
    Output(f"{PID}g-paths", "figure"), Output(f"{PID}g-npi", "figure"),
    Input(f"{PID}store", "data"),
)
def graphs(d):
    empty = go.Figure().update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    if d is None:
        return empty, empty, empty, empty

    alphas = d["alphas"]; h_p = d["h_p"]; h_se = d["h_se"]
    bs_p = d["bs_p"]; bs_se = d["bs_se"]
    t_arr = np.array(d["t_arr"])
    S_pths = np.array(d["S_paths"]); V_pths = np.array(d["V_paths"])
    npi_put = d.get("npi_do_put"); npi_surv = d.get("npi_surv")
    npi_do_call = d.get("npi_do_call"); npi_di_call = d.get("npi_di_call")
    npi_time = d.get("npi_time")
    npi_stable = d.get("npi_stable", True)
    npi_ratio = d.get("npi_resolution_ratio")
    mc_do_call = d.get("do_call"); mc_di_call = d.get("di_call")
    npi_sweep_a = d.get("npi_sweep_a"); npi_sweep_p = d.get("npi_sweep_p")

    # ── Panel 1: DO Put vs alpha ─────────────────────────────────────────
    f1 = go.Figure()
    f1.add_trace(go.Scatter(x=alphas, y=h_p, name="MC Heston",
        line=dict(color=GREEN, width=2.5), mode="lines+markers", marker=dict(size=6),
        error_y=dict(type="data", array=[2 * s for s in h_se], visible=True, color=GREEN, thickness=1.2)))
    f1.add_trace(go.Scatter(x=alphas, y=bs_p, name="MC BS (\u03c3=\u221a\u03b8)",
        line=dict(color=BLUE, width=2.0, dash="dot"), mode="lines+markers",
        marker=dict(size=5, symbol="square"),
        error_y=dict(type="data", array=[2 * s for s in bs_se], visible=True, color=BLUE, thickness=1.0)))
    f1.add_vline(x=d["alpha"], line_color=DGRAY, line_dash="dash", line_width=1,
                 annotation_text=f"\u03b1={d['alpha']:.2f}", annotation_font_size=10)
    if npi_sweep_a and npi_sweep_p:
        f1.add_trace(go.Scatter(x=npi_sweep_a, y=npi_sweep_p, name="NPI 3D (coarse, 5 pts)",
            mode="lines+markers", line=dict(color=BLUE, width=1.5, dash="dash"),
            marker=dict(size=7, symbol="diamond", color=BLUE, line=dict(color="white", width=1))))
    if npi_put is not None:
        marker_color = BLUE if npi_stable else AMBER
        npi_name = "NPI 3D (current \u03b1)" if npi_stable else "NPI 3D (current \u03b1, unstable grid)"
        f1.add_trace(go.Scatter(x=[d["alpha"]], y=[npi_put], name=npi_name, mode="markers",
            marker=dict(color=marker_color, size=15, symbol="diamond",
                        line=dict(color="white", width=2)), showlegend=True))
    f1.update_layout(**BASE, title=dict(text="DO Put vs \u03b1  (Heston vs BS)", font=dict(size=12)))
    f1.update_xaxes(title_text="\u03b1 (barrier level)")
    f1.update_yaxes(title_text="DO Put price")
    f1.data = tuple(sorted(
        f1.data, key=lambda tr: 1 if (tr.mode == "markers" and tr.marker.size == 15) else 0))

    # ── Panel 2: Parameter sensitivity ──────────────────────────────────
    f2 = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.06,
                       subplot_titles=["vs \u03ba", "vs \u03be", "vs \u03c1", "vs \u03b8"])
    f2.add_trace(go.Scatter(x=d["kappas"], y=d["kap_p"], line=dict(color=GREEN, width=2.0),
        mode="lines+markers", name="\u03ba sweep", showlegend=False, marker=dict(size=5)), row=1, col=1)
    f2.add_vline(x=d["kappa"], line_color=DGRAY, line_dash="dash", line_width=1, row=1, col=1)
    f2.add_trace(go.Scatter(x=d["xis"], y=d["xi_p"], line=dict(color=FUCHSIA, width=2.0),
        mode="lines+markers", name="\u03be sweep", showlegend=False, marker=dict(size=5)), row=1, col=2)
    f2.add_vline(x=d["xi"], line_color=DGRAY, line_dash="dash", line_width=1, row=1, col=2)
    f2.add_trace(go.Scatter(x=d["rhos"], y=d["rho_p"], line=dict(color=LBLUE, width=2.0),
        mode="lines+markers", name="\u03c1 sweep", showlegend=False, marker=dict(size=5)), row=1, col=3)
    f2.add_vline(x=d["rho"], line_color=DGRAY, line_dash="dash", line_width=1, row=1, col=3)
    f2.add_trace(go.Scatter(x=d["thetas"], y=d["theta_p"], line=dict(color="#8E6BB0", width=2.0),
        mode="lines+markers", name="\u03b8 sweep", showlegend=False, marker=dict(size=5)), row=1, col=4)
    f2.add_vline(x=d["theta"], line_color=DGRAY, line_dash="dash", line_width=1, row=1, col=4)
    f2.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Georgia,serif", color=DARK, size=11),
        margin=dict(l=50, r=20, t=50, b=50),
        title=dict(text="Parameter Sensitivity  (\u03b1=0.80)", font=dict(size=12)),
        hovermode="x unified")
    for axis in ["xaxis", "xaxis2", "xaxis3", "xaxis4", "yaxis"]:
        f2.update_layout(**{axis: dict(gridcolor=LGRAY)})
    f2.update_yaxes(title_text="DO Put", row=1, col=1)
    f2.update_xaxes(tickangle=-45, row=1, col=3)

    # ── Panel 3: Sample Heston paths ─────────────────────────────────────
    f3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.6, 0.4], vertical_spacing=0.04)
    n_paths = S_pths.shape[0]
    for i in range(n_paths):
        col = GREEN if i % 2 == 0 else LGREEN
        f3.add_trace(go.Scatter(x=t_arr.tolist(), y=S_pths[i].tolist(),
            line=dict(color=col, width=1.0), mode="lines", opacity=0.5, showlegend=(i == 0),
            name="S(t)" if i == 0 else ""), row=1, col=1)
        f3.add_trace(go.Scatter(x=t_arr.tolist(),
            y=(np.exp(np.maximum.accumulate(np.log(S_pths[i])))).tolist(),
            line=dict(color=GRAY, width=0.8, dash="dot"),
            mode="lines", opacity=0.4, showlegend=(i == 0),
            name="Running max" if i == 0 else ""), row=1, col=1)
        f3.add_trace(go.Scatter(x=t_arr.tolist(), y=(V_pths[i] * 100).tolist(),
            line=dict(color=BLUE if i % 2 == 0 else LBLUE, width=1.0), mode="lines",
            opacity=0.5, showlegend=(i == 0), name="V(t)\u00d7100" if i == 0 else ""), row=2, col=1)
    f3.add_hline(y=d["theta"] * 100, line_color=FUCHSIA, line_dash="dash", row=2, col=1,
                 annotation_text=f"\u03b8={d['theta']:.3f}", annotation_font_size=9)
    f3.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Georgia,serif", color=DARK, size=10),
        margin=dict(l=50, r=20, t=58, b=70),
        title=dict(
            text="Sample Heston Paths \u2014 underlying dynamics only<br>"
                 "<span style='font-size:10px;color:%s'>same S(t), V(t) scenarios "
                 "feed every payoff below (Put, Call, DO, DI)</span>" % DGRAY,
            font=dict(size=12)),
        hovermode="x unified", showlegend=False)
    f3.update_yaxes(title_text="S(t)", gridcolor=LGRAY, row=1, col=1)
    f3.update_yaxes(title_text="V\u00d7100", gridcolor=LGRAY, row=2, col=1)
    f3.update_xaxes(title_text="t (years)", gridcolor=LGRAY, row=2, col=1)
    f3.update_xaxes(gridcolor=LGRAY, row=1, col=1)
    f3.add_annotation(
        text=("<span style='color:%s'>&#9644;</span> S(t)   "
              "<span style='color:%s'>&#8231;&#8231;&#8231;</span> running max "
              "(barrier trigger)   "
              "<span style='color:%s'>&#9644;</span> V(t)\u00d7100") % (GREEN, GRAY, BLUE),
        xref="paper", yref="paper", x=0.0, y=-0.24, showarrow=False, align="left",
        font=dict(size=10, family="Georgia,serif", color=DGRAY))

    # ── Panel 4: NPI 3D table ─────────────────────────────────────────────
    headers = ["Grid", "Sizes", "DO Put", "DO Call", "DI Call", "Surv", "Time", "Params"]
    npi_time_str = f"{npi_time:.1f}s" if npi_time is not None else "\u2014"
    npi_put_str = f"{npi_put:.4f}" if npi_put is not None else "\u2014"
    npi_surv_str = f"{npi_surv:.4f}" if npi_surv is not None else "\u2014"
    npi_docall_str = f"{npi_do_call:.4f}" if npi_do_call is not None else "\u2014"
    npi_dicall_str = f"{npi_di_call:.4f}" if npi_di_call is not None else "\u2014"
    if not npi_stable and npi_put is not None:
        npi_put_str += " \u26a0"

    cur_label = (f"\u03b1={d['alpha']:.2f} \u03ba={d['kappa']:.1f}\n"
                 f"\u03b8={d['theta']:.3f} \u03be={d['xi']:.2f} \u03c1={d['rho']:.2f}")
    def_label = "\u03b1=0.80 \u03ba=2.0\n\u03b8=0.04 \u03be=0.30 \u03c1=-0.70"
    live_grid_label = "NPI Coarse\n(live)" if npi_stable else "NPI Coarse\n(live, unstable)"

    mc_docall_str = f"{mc_do_call:.4f}" if mc_do_call is not None else "\u2014"
    mc_dicall_str = f"{mc_di_call:.4f}" if mc_di_call is not None else "\u2014"

    col_grid = [live_grid_label, "NPI Medium\n(precomp.)", "NPI Fine\n(precomp.)", "MC ref"]
    col_sizes = ["121\u00d7151\u00d760", "131\u00d7171\u00d770", "151\u00d7201\u00d780", f"M={MC_HESTON_REF['M']}"]
    col_put = [npi_put_str, "0.5642", "0.5481", f"{MC_HESTON_REF['do_put']:.4f}"]
    col_docall = [npi_docall_str, "10.1854", "10.1224", mc_docall_str]
    col_dicall = [npi_dicall_str, "\u2014", "\u2014", mc_dicall_str]
    col_surv = [npi_surv_str, "0.6505", "0.6497", "\u2014"]
    col_time = [npi_time_str, "7.05s", "~60s", MC_HESTON_REF["time"]]
    col_params = [cur_label, def_label, def_label, def_label]

    LIVE_COL = "#D6EAF8" if npi_stable else "#FBEAD2"
    cell_colors = [
        [LIVE_COL, "white", BG, LGREEN], [LIVE_COL, "white", BG, LGREEN],
        [LIVE_COL, BG, BG, LGREEN], [LIVE_COL, BG, BG, LGREEN],
        [LIVE_COL, BG, BG, LGREEN], [LIVE_COL, BG, BG, "white"],
        [LIVE_COL, "white", "white", "white"], [LIVE_COL, "white", "white", "white"],
    ]
    f4 = go.Figure(go.Table(
        columnwidth=[85, 85, 60, 60, 60, 55, 50, 140],
        header=dict(values=headers, fill_color=DARK, font_color="white",
                    align="center", font_size=11, line_color=DARK, height=28),
        cells=dict(values=[col_grid, col_sizes, col_put, col_docall, col_dicall,
                            col_surv, col_time, col_params],
                   fill_color=cell_colors, align="center", font_size=10,
                   line_color=LGRAY, height=32)))
    subtitle = "NPI 3D  \u2014  live coarse (current params) + precomputed reference"
    if not npi_stable and npi_ratio is not None:
        subtitle += f"  \u26a0 \u03c3\u1d65/dv={npi_ratio:.2f} < {MIN_RESOLUTION_RATIO:.0f}"
    f4.update_layout(
        paper_bgcolor=BG,
        font=dict(family="Georgia,serif", color=DARK, size=11),
        margin=dict(l=10, r=10, t=45, b=28),
        title=dict(text=subtitle, font=dict(size=11)),
        annotations=[dict(
            text="\u26a0 DO Call / DI Call under Heston are indicative, not "
                 "validated prices. Medium/Fine DI Call left blank rather "
                 "than estimated.",
            xref="paper", yref="paper", x=0, y=-0.10, showarrow=False, align="left",
            font=dict(size=9, color=AMBER, family="Georgia,serif"))])

    return f1, f2, f3, f4
