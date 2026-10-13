from __future__ import annotations

"""Shared UI helpers: Times New Roman white theme, KPI cards, styled tables, chart styling."""
import html
import inspect
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

NAVY, BLUE, LIGHT_BLUE, GREY = "#1F3A5F", "#2F6690", "#9DB7D5", "#5B6773"
RISK_COLORS = {"Low": "#2E7D32", "Medium": "#E0A100", "High": "#B3261E"}
RISK_ORDER = ["Low", "Medium", "High"]
FONT = "Times New Roman, Times, serif"

CSS = """
<style>
html, body, .stApp,
.stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]) {
    font-family: 'Times New Roman', Times, 'Liberation Serif', serif !important;
}
.stApp pre:not(#a), .stApp pre *:not(#a), .stApp code:not(#a) { font-family: 'Courier New', Courier, monospace !important; font-size: .95rem; }
.stApp { background: #FFFFFF; color: #1B1B1B; }
header[data-testid="stHeader"] { background: #FFFFFF; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
.stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.55; }
section[data-testid="stSidebar"] { background: #F4F6F9; border-right: 1px solid #DDE3EA; }
.brand { padding: .4rem .2rem 1rem .2rem; border-bottom: 2px solid #1F3A5F; margin-bottom: 1rem; }
.brand .t { font-size: 1.35rem; font-weight: 700; color: #1F3A5F; line-height: 1.25; }
.brand .s { font-size: .95rem; color: #5B6773; margin-top: .25rem; }
.page-header { border-bottom: 2px solid #1F3A5F; padding-bottom: .55rem; margin-bottom: 1.3rem; }
.page-header h1 { font-size: 2.1rem; color: #1F3A5F; margin: 0; padding: 0; font-weight: 700; }
.page-header p { color: #5B6773; margin: .25rem 0 0 0; font-size: 1.1rem; }
.section-title { font-size: 1.3rem; font-weight: 700; color: #1F3A5F; border-left: 4px solid #2F6690;
    padding-left: .6rem; margin: 1.5rem 0 .7rem 0; }
.kpi { background: #FFFFFF; border: 1px solid #DDE3EA; border-top: 4px solid #1F3A5F; border-radius: 6px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(16,24,40,.07); height: 118px; box-sizing: border-box; }
.kpi .label { font-size: .88rem; color: #5B6773; text-transform: uppercase; letter-spacing: .05em; font-weight: 700; }
.kpi .value { font-size: 1.85rem; font-weight: 700; color: #1F3A5F; line-height: 1.25; margin-top: 2px; }
.kpi .sub { font-size: .92rem; color: #5B6773; }
.kpi.red { border-top-color: #B3261E; } .kpi.red .value { color: #B3261E; }
.kpi.green { border-top-color: #2E7D32; }
.tbl-wrap { overflow: auto; border: 1px solid #DDE3EA; border-radius: 6px; margin-bottom: .8rem; }
table.dw { border-collapse: collapse; width: 100%; font-size: 1rem; }
table.dw th { background: #1F3A5F; color: #FFFFFF; position: sticky; top: 0; padding: 8px 10px;
    text-align: left; white-space: nowrap; font-weight: 700; }
table.dw td { padding: 7px 10px; border-bottom: 1px solid #EEF1F5; vertical-align: top; }
table.dw tr:nth-child(even) td { background: #F7F9FB; }
table.kv td:first-child { font-weight: 700; color: #1F3A5F; width: 42%; background: #F4F6F9; }
.badge { padding: 2px 10px; border-radius: 10px; font-weight: 700; font-size: .88rem; color: #FFFFFF; white-space: nowrap; }
.badge.High { background: #B3261E; } .badge.Medium { background: #C98A00; } .badge.Low { background: #2E7D32; }
.note { background: #F4F6F9; border-left: 4px solid #2F6690; padding: .7rem 1rem; border-radius: 4px; margin: .5rem 0 1rem 0; }
div[data-testid="stPlotlyChart"] { border: 1px solid #EEF1F5; border-radius: 6px; padding: 4px; }
.stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 700; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h1>{html.escape(title)}</h1>{sub}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def kpi_row(items) -> None:
    """items: list of (label, value, sub[, css_class])."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, sub = item[:3]
        cls = item[3] if len(item) > 3 else ""
        col.markdown(
            f'<div class="kpi {cls}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="sub">{html.escape(str(sub))}</div></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- formatting
def fmt_inr(x) -> str:
    """Compact rupee format (Lakh / Crore)."""
    if x is None or pd.isna(x):
        return "-"
    x = float(x)
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def inr_full(x) -> str:
    """Full rupee amount with Indian digit grouping, e.g. ₹12,34,500."""
    if x is None or pd.isna(x):
        return "-"
    n = int(round(float(x)))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"{'-' if n < 0 else ''}₹{s}"


def _cell(v, badge: bool) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if badge:
        return f'<span class="badge {html.escape(str(v))}">{html.escape(str(v))}</span>'
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    return html.escape(str(v))


def html_table(df: pd.DataFrame, badge_cols=("risk_level", "Risk Level"), max_height: int = 460, kv: bool = False) -> None:
    """Render a DataFrame as a styled Times New Roman HTML table."""
    d = df.copy()
    for col in d.columns:
        d[col] = [_cell(v, col in badge_cols) for v in d[col]]
    table = d.to_html(index=False, escape=False, border=0, classes="dw kv" if kv else "dw",
                      header=not kv)
    table = re.sub(r">\s+<", "><", table).replace("\n", "")
    st.markdown(f'<div class="tbl-wrap" style="max-height:{max_height}px">{table}</div>', unsafe_allow_html=True)


def kv_table(pairs: dict) -> None:
    html_table(pd.DataFrame({"k": list(pairs.keys()), "v": [str(v) for v in pairs.values()]}), kv=True, max_height=520)


# ----------------------------------------------------------------------------- charts
def style_fig(fig: go.Figure, height: int = 380, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family=FONT, size=14, color="#1B1B1B"),
        title=dict(text=title, x=0.0, font=dict(size=17, color=NAVY)) if title else None,
        margin=dict(l=40, r=20, t=60 if title else 25, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title_text=""),
        colorway=[NAVY, "#4C8BC2", "#9DB7D5", "#C98A00", "#B3261E", "#2E7D32"],
    )
    fig.update_xaxes(showgrid=False, linecolor="#C9D1DB")
    fig.update_yaxes(gridcolor="#EEF1F5", linecolor="#C9D1DB")
    return fig


_NEW_API = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig: go.Figure, key: str | None = None) -> None:
    if _NEW_API:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def gauge(value: float, title: str = "Risk Score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), title={"text": title, "font": {"size": 18}},
        number={"font": {"size": 40, "color": NAVY}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
               "steps": [{"range": [0, 35], "color": "#DCEFE0"}, {"range": [35, 60], "color": "#FBEFC9"},
                         {"range": [60, 100], "color": "#F6D5D2"}],
               "threshold": {"line": {"color": "#B3261E", "width": 3}, "thickness": 0.8, "value": 60}}))
    fig.update_layout(height=270, margin=dict(l=25, r=25, t=60, b=10), paper_bgcolor="#FFFFFF",
                      font=dict(family=FONT))
    return fig
