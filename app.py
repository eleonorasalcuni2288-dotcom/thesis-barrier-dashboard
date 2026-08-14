"""
app.py — Entry point for the multi-page thesis dashboard.

Barrier Option Pricing via Path Integral Methods
Uses dash.register_page() so every dashboard becomes a page of a single
app, served from a single URL (needed for the QR code in the thesis).

Run locally with:
    python app.py
Then open: http://127.0.0.1:8000

Deploy: see README.md for GitHub + Render instructions.
"""

import os
import dash
from dash import Dash, html, dcc, page_container

BG = "#F8F9FC"
DARK = "#2C2C2A"
LGRAY = "#E8E6DF"
BLUE = "#185FA5"

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
app.title = "Barrier Option Pricing — Path Integral Methods"
server = app.server  # exposed for gunicorn / Render

app.layout = html.Div(style={
    "backgroundColor": BG, "minHeight": "100vh",
    "fontFamily": "Georgia, serif",
}, children=[

    # ── Top nav bar, present on every page ────────────────────────────────
    html.Div(style={
        "backgroundColor": "white", "borderBottom": f"1px solid {LGRAY}",
        "padding": "14px 32px", "display": "flex", "alignItems": "center",
        "justifyContent": "space-between", "flexWrap": "wrap", "gap": "10px",
    }, children=[
        dcc.Link(
            "Barrier Option Pricing — Path Integral Methods",
            href="/",
            style={"fontSize": "16px", "fontWeight": "700", "color": DARK,
                   "textDecoration": "none"},
        ),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                  children=[
            dcc.Link(page["name"], href=page["path"], style={
                "fontSize": "12px", "color": BLUE, "textDecoration": "none",
                "fontFamily": "Georgia, serif",
            })
            for page in dash.page_registry.values()
        ]),
    ]),

    page_container,
])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  Thesis dashboard → http://127.0.0.1:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
