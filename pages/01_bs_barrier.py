"""
pages/01_bs_barrier.py — Dashboard 1: Black-Scholes analytical barrier.

Adapted from the original standalone dashboard (dashboard.py, port 8050)
to run as a page of the multi-page app. Pricing formulas now come from
the shared, verified pricing_bs module instead of a local duplicate.
"""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, callback

from pricing_bs import bs_down_barrier

dash.register_page(__name__, path="/bs-barrier",
                    name="1. BS Barrier (analytical)")

BLUE, LBLUE, FUCHSIA = "#185FA5", "#85B7EB", "#D4537E"
GRAY, LGRAY, BG, DARK = "#B4B2A9", "#E8E6DF", "#F8F9FC", "#2C2C2A"

LABEL_STYLE = {"fontSize": "12px", "color": "#5F5E5A",
               "fontFamily": "Georgia, serif", "letterSpacing": "0.03em"}
VALUE_STYLE = {"fontSize": "13px", "fontWeight": "600", "color": DARK,
               "fontFamily": "Georgia, serif"}

PID = "bs1-"  # id prefix to avoid any cross-page collisions


def _slider_block(label, id_):
    return html.Div([
        html.Div([
            html.Span(label, style=LABEL_STYLE),
            html.Span(id=f"{PID}{id_}-val", style=VALUE_STYLE),
        ], style={"display": "flex", "justifyContent": "space-between",
                   "marginBottom": "4px"}),
    ])


layout = html.Div(style={
    "backgroundColor": BG, "minHeight": "100vh",
    "fontFamily": "Georgia, serif", "padding": "28px 36px"
}, children=[

    html.Div([
        html.H1("Barrier Option Pricing", style={
            "fontSize": "22px", "fontWeight": "600", "margin": "0 0 4px",
            "color": DARK, "letterSpacing": "-0.02em"}),
        html.P("Black-Scholes analytical — Reiner & Rubinstein (1991)", style={
            "fontSize": "13px", "color": "#73726C", "margin": 0,
            "fontStyle": "italic"}),
    ], style={"marginBottom": "28px", "borderBottom": f"1px solid {LGRAY}",
              "paddingBottom": "16px"}),

    html.Div(style={"display": "flex", "gap": "32px", "flexWrap": "wrap"},
             children=[

        html.Div(style={
            "width": "260px", "flexShrink": "0",
            "backgroundColor": "white", "borderRadius": "10px",
            "padding": "20px", "border": f"1px solid {LGRAY}",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"
        }, children=[

            html.P("PARAMETRI", style={
                "fontSize": "10px", "letterSpacing": "0.12em", "color": GRAY,
                "marginBottom": "18px", "fontFamily": "Georgia, serif"}),

            *[html.Div([
                _slider_block(lbl, id_),
                dcc.Slider(id=f"{PID}{id_}", min=mn, max=mx, step=st,
                           value=vl, marks=None,
                           tooltip={"always_visible": False}),
            ], style={"marginBottom": "18px"})
              for lbl, id_, mn, mx, st, vl in [
                  ("Spot price S\u2080", "s0", 80, 120, 1, 100),
                  ("Strike K", "k", 80, 120, 1, 100),
                  ("Barrier H", "h", 50, 99, 1, 80),
                  ("Volatility \u03c3", "sigma", 5, 60, 1, 20),
                  ("Risk-free rate r", "r", 0, 15, 1, 5),
                  ("Maturity T (years)", "t", 25, 300, 5, 100),
              ]],

            html.Hr(style={"borderColor": LGRAY, "margin": "0 0 16px"}),

            html.P("PREZZI CORRENTI", style={
                "fontSize": "10px", "letterSpacing": "0.12em", "color": GRAY,
                "marginBottom": "12px"}),
            html.Div(id=f"{PID}price-table"),
        ]),

        html.Div(style={"flex": "1", "minWidth": "480px"}, children=[
            dcc.Graph(id=f"{PID}main-graph", style={"height": "460px"},
                      config={"displayModeBar": False}),
            dcc.Graph(id=f"{PID}sigma-graph",
                      style={"height": "380px", "marginTop": "16px"},
                      config={"displayModeBar": False}),
        ]),
    ]),

    html.P(
        "S\u2080=100, r=5%, q=0%, rebate=0. Formula verificata contro "
        "QuantLib AnalyticBarrierEngine.",
        style={"fontSize": "11px", "color": GRAY, "marginTop": "20px",
               "fontStyle": "italic"}),
])


def make_price_row(label, value, color):
    return html.Div([
        html.Span(label, style={"fontSize": "12px", "color": "#5F5E5A",
                                 "fontFamily": "Georgia,serif"}),
        html.Span(f"{value:.4f}", style={
            "fontSize": "13px", "fontWeight": "600", "color": color,
            "fontFamily": "Georgia,serif"}),
    ], style={"display": "flex", "justifyContent": "space-between",
              "marginBottom": "8px"})


@callback(
    Output(f"{PID}s0-val", "children"),
    Output(f"{PID}k-val", "children"),
    Output(f"{PID}h-val", "children"),
    Output(f"{PID}sigma-val", "children"),
    Output(f"{PID}r-val", "children"),
    Output(f"{PID}t-val", "children"),
    Output(f"{PID}price-table", "children"),
    Output(f"{PID}main-graph", "figure"),
    Output(f"{PID}sigma-graph", "figure"),
    Input(f"{PID}s0", "value"),
    Input(f"{PID}k", "value"),
    Input(f"{PID}h", "value"),
    Input(f"{PID}sigma", "value"),
    Input(f"{PID}r", "value"),
    Input(f"{PID}t", "value"),
)
def update(s0, K, H, sigma_pct, r_pct, t_scaled):
    S0 = float(s0)
    K = float(K)
    H = min(float(H), S0 - 1)
    sigma = sigma_pct / 100
    r = r_pct / 100
    T = t_scaled / 100

    res = bs_down_barrier(S0, K, H, r, 0, sigma, T)
    if res is None:
        empty = go.Figure()
        return "100", "100", "80", "20%", "5%", "1.0y", \
            html.P("H deve essere < S\u2080"), empty, empty

    do_p, di_p = res["do_put"], res["di_put"]
    do_c, di_c = res["do_call"], res["di_call"]
    vp, vc = res["vanilla_put"], res["vanilla_call"]

    s0_lbl = str(int(s0))
    k_lbl = str(int(K))
    h_lbl = str(int(H))
    sig_lbl = f"{sigma_pct}%"
    r_lbl = f"{r_pct}%"
    t_lbl = f"{T:.2f}y"

    price_table = html.Div([
        make_price_row("DO Put", do_p, BLUE),
        make_price_row("DI Put", di_p, FUCHSIA),
        make_price_row("DO Call", do_c, LBLUE),
        make_price_row("DI Call", di_c, "#B06090"),
        html.Hr(style={"borderColor": LGRAY, "margin": "6px 0"}),
        make_price_row("Vanilla Put", vp, GRAY),
        make_price_row("Vanilla Call", vc, GRAY),
        html.Div([
            html.Span("Parity \u2713" if abs(do_p + di_p - vp) < 1e-6 else "Parity \u2717",
                      style={"fontSize": "11px",
                             "color": "#1D9E75" if abs(do_p + di_p - vp) < 1e-6 else "#E24B4A"})
        ]),
    ])

    # ── Grafico 1: price vs H ────────────────────────────────────────────
    hs = np.linspace(max(50, S0 * 0.50), S0 - 0.5, 200)
    do_ph, di_ph, do_ch, di_ch_arr, vp_arr = [], [], [], [], []
    for h in hs:
        r2 = bs_down_barrier(S0, K, h, r, 0, sigma, T)
        if r2:
            do_ph.append(r2["do_put"]); di_ph.append(r2["di_put"])
            do_ch.append(r2["do_call"]); di_ch_arr.append(r2["di_call"])
            vp_arr.append(r2["vanilla_put"])
        else:
            do_ph.append(None); di_ph.append(None)
            do_ch.append(None); di_ch_arr.append(None); vp_arr.append(None)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=hs, y=do_ph, name="DO Put", line=dict(color=BLUE, width=2.5)))
    fig1.add_trace(go.Scatter(x=hs, y=di_ph, name="DI Put", line=dict(color=FUCHSIA, width=2.5)))
    fig1.add_trace(go.Scatter(x=hs, y=do_ch, name="DO Call", line=dict(color=LBLUE, width=2, dash="dot")))
    fig1.add_trace(go.Scatter(x=hs, y=di_ch_arr, name="DI Call", line=dict(color="#B06090", width=2, dash="dot")))
    fig1.add_trace(go.Scatter(x=hs, y=vp_arr, name="Vanilla Put",
                               line=dict(color=GRAY, width=1.5, dash="dash")))
    fig1.add_vline(x=H, line_color="#E24B4A", line_dash="dash", line_width=1.5,
                   annotation_text=f"H={int(H)}", annotation_font_color="#E24B4A",
                   annotation_font_size=11)
    fig1.add_vline(x=K, line_color=DARK, line_dash="dot", line_width=1,
                   annotation_text=f"K={int(K)}", annotation_font_color=DARK,
                   annotation_font_size=11)
    fig1.update_layout(
        title=dict(text=f"Price vs Barrier H  (\u03c3={sigma_pct}%, T={T:.2f}y, K={int(K)})",
                   font=dict(size=13, color=DARK, family="Georgia,serif")),
        xaxis_title="Barrier level H", yaxis_title="Option price",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.18, font=dict(size=11, family="Georgia,serif")),
        margin=dict(l=50, r=20, t=50, b=60),
        font=dict(family="Georgia,serif", color=DARK),
        xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
        hovermode="x unified")

    # ── Grafico 2: price vs sigma ────────────────────────────────────────
    sigs = np.linspace(0.05, 0.65, 200)
    do_ps2, di_ps2, di_cs2, vp_s2 = [], [], [], []
    for s in sigs:
        r2 = bs_down_barrier(S0, K, H, r, 0, s, T)
        if r2:
            do_ps2.append(r2["do_put"]); di_ps2.append(r2["di_put"])
            di_cs2.append(r2["di_call"]); vp_s2.append(r2["vanilla_put"])
        else:
            do_ps2.append(None); di_ps2.append(None)
            di_cs2.append(None); vp_s2.append(None)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sigs * 100, y=do_ps2, name="DO Put",
                               line=dict(color=BLUE, width=2.5)))
    fig2.add_trace(go.Scatter(x=sigs * 100, y=di_ps2, name="DI Put",
                               line=dict(color=FUCHSIA, width=2.5)))
    fig2.add_trace(go.Scatter(x=sigs * 100, y=di_cs2, name="DI Call",
                               line=dict(color="#B06090", width=2, dash="dot")))
    fig2.add_trace(go.Scatter(x=sigs * 100, y=vp_s2, name="Vanilla Put",
                               line=dict(color=GRAY, width=1.5, dash="dash")))
    fig2.add_vline(x=sigma_pct, line_color="#E24B4A", line_dash="dash", line_width=1.5,
                   annotation_text=f"\u03c3={sigma_pct}%", annotation_font_color="#E24B4A",
                   annotation_font_size=11)
    fig2.update_layout(
        title=dict(text=f"Price vs Volatility \u03c3  (H={int(H)}, T={T:.2f}y, K={int(K)})",
                   font=dict(size=13, color=DARK, family="Georgia,serif")),
        xaxis_title="Volatility \u03c3 (%)", yaxis_title="Option price",
        plot_bgcolor=BG, paper_bgcolor=BG,
        legend=dict(orientation="h", y=-0.22, font=dict(size=11, family="Georgia,serif")),
        margin=dict(l=50, r=20, t=50, b=70),
        font=dict(family="Georgia,serif", color=DARK),
        xaxis=dict(gridcolor=LGRAY), yaxis=dict(gridcolor=LGRAY),
        hovermode="x unified")

    return s0_lbl, k_lbl, h_lbl, sig_lbl, r_lbl, t_lbl, price_table, fig1, fig2
