"""
pages/00_home.py — Landing page / index for the thesis dashboard.
"""

import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

BG = "#F8F9FC"
DARK = "#2C2C2A"
DGRAY = "#5F5E5A"
GRAY = "#B4B2A9"
LGRAY = "#E8E6DF"
BLUE = "#185FA5"
CARD = {
    "backgroundColor": "white", "borderRadius": "10px", "padding": "18px 20px",
    "border": f"1px solid {LGRAY}", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
    "display": "block", "textDecoration": "none",
}

SECTIONS = [
    ("Fixed barrier  (B = H, constant)", [
        ("01", "Black-Scholes Barrier (analytical)",
         "Reiner-Rubinstein closed-form formula — DO/DI Put & Call.", "/bs-barrier"),
        ("02", "Monte Carlo Naive",
         "Euler-Maruyama discretization, bias O(√Δt) vs BS benchmark.", "/mc-naive"),
        ("03", "Brownian Bridge Correction",
         "Analytical inter-node correction — bias O(Δt), variance reduction.", "/brownian-bridge"),
    ]),
    ("Floating barrier  (B_t = α · running max)", [
        ("4.4", "MC Naive — Floating Barrier",
         "Discrete monitoring, no intra-step correction.", "/mc-naive-floating"),
        ("5.5", "MC BB — Floating Barrier",
         "Local Brownian Bridge approximation (not exact for floating barrier).", "/mc-bb-floating"),
        ("6.6", "NPI — Floating Barrier (fixed grid)",
         "Deterministic density propagation on augmented state (x, v).", "/npi-floating"),
        ("7", "Method Comparison",
         "NPI vs MC Naive vs MC BB side by side.", "/comparison"),
        ("9.9", "Adaptive NPI Grid",
         "Non-uniform v-grid, concentrated near zero drawdown.", "/adaptive-npi"),
        ("10", "Heston NPI 3D",
         "Stochastic volatility extension — state (x, v, drawdown).", "/heston"),
    ]),
]


def make_card(num, title, desc, href):
    return dcc.Link(href=href, style=CARD, children=[
        html.Div([
            html.Span(num, style={
                "fontSize": "11px", "color": BLUE, "fontWeight": "700",
                "border": f"1px solid {BLUE}", "borderRadius": "4px",
                "padding": "1px 6px", "marginRight": "8px"}),
            html.Span(title, style={
                "fontSize": "14px", "fontWeight": "600", "color": DARK}),
        ], style={"marginBottom": "6px"}),
        html.P(desc, style={
            "fontSize": "12px", "color": DGRAY, "margin": 0,
            "lineHeight": "1.4"}),
    ])


layout = html.Div(style={
    "padding": "36px 32px", "maxWidth": "1100px", "margin": "0 auto",
}, children=[
    html.H1("Barrier Option Pricing via Path Integral Methods", style={
        "fontSize": "24px", "fontWeight": "700", "color": DARK,
        "marginBottom": "6px"}),
    html.P(
        "Interactive companion dashboards — Master's thesis, Quantitative Finance.",
        style={"fontSize": "13px", "color": DGRAY, "fontStyle": "italic",
               "marginBottom": "32px"}),

    *[html.Div([
        html.P(section_title, style={
            "fontSize": "11px", "letterSpacing": "0.1em", "color": GRAY,
            "textTransform": "uppercase", "marginBottom": "12px",
            "borderBottom": f"1px solid {LGRAY}", "paddingBottom": "8px"}),
        html.Div(style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fill, minmax(300px, 1fr))",
            "gap": "14px", "marginBottom": "32px",
        }, children=[make_card(n, t, d, h) for n, t, d, h in cards]),
    ]) for section_title, cards in SECTIONS],
])
