"""AI Energy Lab - Project 74: Energy, Cost and Sustainability Analysis of AI Systems.

Streamlit interactive dashboard. Empirical physical telemetry from local AI inference,
cloud alternatives, and fleet sizing with a sleek white & orange design.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aienergy.model_arena import render_model_arena
from aienergy import sustainability as S

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AI Energy Lab · Energy, Cost & Carbon Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from aienergy import calculator as C
    from aienergy.profiles import build_profiles, load_results, placeholder_profiles
    from scenarios.campus_chatbot import G as CG, W as CW
except ImportError as exc:
    st.error(f"Could not import project modules: {exc}. Run Streamlit from the project folder.")
    st.stop()

# ---------------------------------------------------------------- Sleek White & Orange Design System
ORANGE_PRIMARY = "#F97316"
ORANGE_DARK = "#EA580C"
ORANGE_LIGHT = "#FFF7ED"
ORANGE_BORDER = "#FDBA74"
AMBER = "#D97706"
BLUE = "#0284C7"
EMERALD = "#10B981"
ROSE = "#E11D48"

_VER = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
STRETCH = {"width": "stretch"} if _VER >= (1, 50) else {"use_container_width": True}
PALETTE = [ORANGE_PRIMARY, BLUE, EMERALD, "#8B5CF6", ROSE, AMBER, "#64748B"]
KIND_LABELS = {"local": "Local server", "cloud_api": "Cloud API", "cloud_gpu": "Cloud GPU rental"}
QUALITY_MANUAL = Path("data/results/phase1_quality_manual_audit.csv")
QUALITY_AUTO = Path("data/quality.csv")
CLOUD_PATH = Path("data/cloud_options.json")
CLOUD_BACKUP = Path("data/cloud_options.backup.json")

# ---------------------------------------------------------------- Dark / Light Mode State
st.session_state.setdefault("dark_mode", False)

# ---------------------------------------------------------------- CSS — injected dynamically per mode
def _inject_css(dark: bool) -> None:
    """Inject the stylesheet for the active theme.

    Both themes style Streamlit's own widgets (tabs, labels, buttons, inputs,
    alerts, expanders) as well as this app's custom HTML cards, so switching
    modes never leaves text stranded at the wrong contrast.
    """
    if dark:
        # ── Dark Mode ── deep slate surfaces / orange accent
        css = """
<style>
/* ═══════════ DARK MODE ═══════════ */
:root { color-scheme: dark; }
.stApp {
  --background-color: #0F172A;
  --secondary-background-color: #1E293B;
  --text-color: #E6EDF6;
  --primary-color: #F97316;
}
html, body, .stApp {
  font-size: 15px !important;
  background-color: #0F172A !important;
  color: #E6EDF6 !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stAppViewContainer"], section[data-testid="stMain"],
section[data-testid="stMain"] > div, [data-testid="stHeader"], [data-testid="stToolbar"] {
  background-color: #0F172A !important;
}
.block-container { max-width: 1480px; padding-top: 3.6rem; padding-bottom: 3.5rem; }

/* ── Slim app bar ──
   Streamlit's stock header is a blank 56px band pinned over the page, and it
   was sitting on top of the tab row. Replace it with a compact translucent
   bar that actually carries the app's identity, and clear its height so no
   content hides underneath. */
[data-testid="stHeader"], header[data-testid="stHeader"] {
  height: 44px !important; min-height: 44px !important;
  background: rgba(15,23,42,0.90) !important;
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border-bottom: 1px solid #1E293B !important; box-shadow: none !important;
}
[data-testid="stHeader"] [data-testid="stToolbar"] {
  height: 44px !important; min-height: 44px !important; padding: 0 10px !important;
}
[data-testid="stToolbar"]::before {
  content: "⚡ AI Energy Lab";
  position: absolute; left: 58px; top: 50%; transform: translateY(-50%);
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
  color: #FDBA74 !important; white-space: nowrap; pointer-events: none;
}
[data-testid="stHeader"] button, [data-testid="stHeader"] [data-testid="stMainMenu"] {
  border-radius: 6px !important;
}
[data-testid="stHeader"] svg { fill: #CBD5E1 !important; }
[data-testid="stHeader"] button:hover { background: #243044 !important; }

/* ── Base text: nothing may stay dark-on-dark ──
   Kept at one-class specificity so the component and card rules below can
   re-assert their own accent colours (badges, tabs, warnings). */
.stApp * { color: #E6EDF6 !important; }
.stApp p, .stApp li { line-height: 1.6 !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp h5, .stApp h6, .stApp strong { color: #FFFFFF !important; }
/* Section headers get an orange rule so they never disappear */
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h1,
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h2,
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h3 {
  border-left: 4px solid #F97316; padding-left: 12px;
  margin: 1.2rem 0 0.5rem 0 !important; line-height: 1.25;
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span { color: #A9B7CB !important; }
.stApp a, .stApp a * { color: #FDBA74 !important; text-decoration-color: #FDBA74; }
.stApp hr { border-color: #334155 !important; }
.stApp code { background: #243044 !important; color: #FDBA74 !important; border-radius: 4px; }
/* Markdown tables */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; }
[data-testid="stMarkdownContainer"] th {
  background: #243044 !important; color: #FFFFFF !important;
  border: 1px solid #334155 !important; padding: 8px 10px !important; text-align: left;
}
[data-testid="stMarkdownContainer"] td {
  border: 1px solid #334155 !important; padding: 8px 10px !important; color: #E6EDF6 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: #1B2637 !important;
  border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * { color: #D8E1EC !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #A9B7CB !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px; border-bottom: 1.5px solid #334155; padding-bottom: 4px;
  flex-wrap: wrap; background: transparent !important;
}
.stTabs [data-baseweb="tab"], [data-testid="stTab"] {
  background: #1E293B !important; border: 1px solid #334155 !important;
  border-radius: 8px 8px 0 0; padding: 8px 16px; transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"] p, [data-testid="stTab"] p {
  color: #C7D3E3 !important; font-size: 0.92rem !important; font-weight: 600 !important;
}
.stTabs [data-baseweb="tab"]:hover p, [data-testid="stTab"]:hover p { color: #FDBA74 !important; }
.stTabs [data-baseweb="tab"]:hover, [data-testid="stTab"]:hover { border-color: #F97316 !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] {
  background: #243044 !important; border-color: #F97316 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTab"][aria-selected="true"] p {
  color: #FDBA74 !important; font-weight: 700 !important;
}
[data-testid="stTabHighlight"], .stTabs [data-baseweb="tab-highlight"] { background-color: #F97316 !important; }

/* ── Forms / widget labels ── */
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
  color: #D8E1EC !important; font-weight: 600 !important;
}
[data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stNumberInput"] [data-baseweb="input"] {
  background: #0F172A !important; border-color: #3B4A63 !important;
}
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: #0F172A !important; color: #E6EDF6 !important;
  -webkit-text-fill-color: #E6EDF6 !important;
}
[data-testid="stNumberInput"] button {
  background: #243044 !important; border-color: #3B4A63 !important;
}
[data-testid="stNumberInput"] button svg { fill: #D8E1EC !important; }
[data-baseweb="select"] > div, [data-baseweb="select"] [role="button"] {
  background: #0F172A !important; border-color: #3B4A63 !important; color: #E6EDF6 !important;
}
[data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] {
  background: #1E293B !important; color: #E6EDF6 !important;
}
[data-baseweb="popover"] [role="option"]:hover { background: #243044 !important; }
[data-testid="stCheckbox"] label p, [data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label span, [data-testid="stRadio"] label span { color: #D8E1EC !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color: #0F172A !important; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stSliderTickBarMin"] p, [data-testid="stSliderTickBarMax"] p { color: #A9B7CB !important; }
[data-testid="stTooltipContent"] { background: #243044 !important; }

/* ── Buttons ── */
.stButton button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {
  background: #243044 !important; border: 1px solid #3B4A63 !important;
  color: #EEF4FB !important; border-radius: 8px !important;
}
.stButton button p, [data-testid="stDownloadButton"] button p { color: #EEF4FB !important; font-weight: 600 !important; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover { border-color: #F97316 !important; }
.stButton button:hover p, [data-testid="stDownloadButton"] button:hover p { color: #FDBA74 !important; }
.stButton button:hover svg, [data-testid="stDownloadButton"] button:hover svg { fill: #FDBA74 !important; }
[data-testid="stBaseButton-primary"] { background: #F97316 !important; border-color: #F97316 !important; }
[data-testid="stBaseButton-primary"] p { color: #0F172A !important; }

/* ── Expanders / alerts ── */
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {
  background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 10px !important;
}
[data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span {
  color: #EEF4FB !important; font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg { fill: #C7D3E3 !important; }
[data-testid="stAlertContainer"], [data-testid="stAlert"] {
  background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 10px !important;
}
[data-testid="stAlertContainer"] p, [data-testid="stAlert"] p { color: #E6EDF6 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: #1E293B !important; border: 1px solid #334155 !important;
  border-left: 4px solid #F97316 !important; border-radius: 12px !important;
  padding: 14px 18px !important; box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  transition: transform 0.18s ease, border-color 0.18s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #F97316 !important; }
[data-testid="stMetricLabel"] p { font-size: 0.88rem !important; color: #A9B7CB !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 800 !important; color: #FFFFFF !important; }
[data-testid="stMetricDelta"] { color: #34D399 !important; }

/* ── Dataframes (canvas keeps the light theme's ink, so the panel stays light) ── */
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] {
  background: #FFFFFF !important; border: 1px solid #334155 !important;
  border-radius: 12px !important; overflow: hidden !important;
}
[data-testid="stDataFrame"] * { color: #0F172A !important; }
[data-testid="stDataFrame"] svg { fill: #334155 !important; }

/* ═══════════ App cards ═══════════ */
.white-card {
  background: #1E293B !important; border: 1px solid #334155 !important;
  border-radius: 12px; padding: 18px 20px; margin-bottom: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.white-card h4 { color: #FFFFFF !important; }
.white-card p  { color: #B7C4D6 !important; }

.launch-card {
  background: #1E293B !important; border: 1px solid #334155 !important;
  border-top: 4px solid #F97316 !important; border-radius: 12px;
  padding: 18px 20px; height: 100%;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.launch-card:hover { box-shadow: 0 8px 24px rgba(249,115,22,0.18); transform: translateY(-3px); }
.launch-card h3 { color: #FFFFFF !important; }
.launch-card p  { color: #B7C4D6 !important; }

.launch-header {
  background: linear-gradient(135deg, #1E293B 0%, #1a2540 55%, #1e1a2e 100%) !important;
  border: 1px solid #334155 !important; border-left: 6px solid #F97316 !important;
  border-radius: 14px; padding: 22px 26px; margin-bottom: 1.2rem;
}
.launch-header h1 { color: #FFFFFF !important; }
.launch-header p  { color: #B7C4D6 !important; }
.launch-header .kicker { color: #FDBA74 !important; }

.explainer-box {
  background: #1a1f2e !important; border: 1px solid #3d2b00 !important;
  border-left: 4px solid #F97316 !important; color: #FDBA74 !important;
}

/* Info tiles used by the "how to read" and glossary blocks */
.info-tile {
  background: #1E293B !important; border: 1px solid #334155 !important;
  border-left: 3px solid #F97316 !important; border-radius: 10px;
  padding: 14px 16px; margin-bottom: 10px;
}
.info-tile h5 { margin: 0 0 4px 0; color: #FFFFFF !important; font-size: 0.95rem; font-weight: 700; }
.info-tile p  { margin: 0; color: #B7C4D6 !important; font-size: 0.88rem; line-height: 1.5; }
.info-tile .formula {
  display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 5px;
  background: #243044 !important; color: #FDBA74 !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
}

/* Status pill + comparison cards + recommendation + savings callout */
[data-testid="stSidebar"] .status-pill {
  background: #14532D !important; border-color: #166534 !important; color: #86EFAC !important;
}
.cmp-card { box-shadow: none !important; }
.cmp-good { background: #0F1F1B !important; border-color: #10B981 !important; }
.cmp-bad  { background: #1F1115 !important; border-color: #E11D48 !important; }
.cmp-good h4 { color: #34D399 !important; }
.cmp-bad  h4 { color: #FB7185 !important; }
.cmp-good div, .cmp-bad div, .cmp-good p, .cmp-bad p { color: #C7D3E3 !important; }
.cmp-good strong, .cmp-bad strong { color: #FFFFFF !important; }
.cmp-foot-good { color: #34D399 !important; border-top-color: #334155 !important; }
.cmp-foot-bad  { color: #FB7185 !important; border-top-color: #334155 !important; }
.rec-card { background: #1E293B !important; border-color: #F97316 !important; box-shadow: none !important; }
.rec-card h3 { color: #FFFFFF !important; }
.rec-card p  { color: #C7D3E3 !important; }
.rec-note { background: #3d1f00 !important; color: #FDBA74 !important; }
.rec-kicker { color: #FDBA74 !important; }
.rec-meta { color: #A9B7CB !important; }
.savings-box { background: #1E293B !important; border-color: #F97316 !important; }
.savings-box h4 { color: #FDBA74 !important; }
.savings-box p  { color: #C7D3E3 !important; }
.savings-box strong { color: #FFFFFF !important; }

/* ── Badges ── */
.badge { border-radius: 6px; }
.badge-orange { background: #3d1f00 !important; color: #FDBA74 !important; border: 1px solid #92400e !important; }
.badge-blue   { background: #0c2340 !important; color: #7DD3FC !important; border: 1px solid #0369A1 !important; }
.badge-emerald{ background: #052e16 !important; color: #6EE7B7 !important; border: 1px solid #065F46 !important; }
.badge-rose   { background: #3d0014 !important; color: #FDA4AF !important; border: 1px solid #9f1239 !important; }

/* Plotly container */
.js-plotly-plot .plotly { background: transparent !important; }
/* Smooth mode transition */
html, body, .stApp, [data-testid="stSidebar"],
[data-testid="stMetric"], .white-card, .launch-card, .launch-header {
  transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}
</style>"""
    else:
        # ── Light Mode ── crisp white / orange
        css = """
<style>
/* ═══════════ LIGHT MODE ═══════════ */
:root { color-scheme: light; }
.stApp {
  --background-color: #F8FAFC;
  --secondary-background-color: #FFFFFF;
  --text-color: #0F172A;
  --primary-color: #F97316;
}
html, body, .stApp {
  font-size: 15px;
  background-color: #F8FAFC !important;
  color: #0F172A !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.block-container { max-width: 1480px; padding-top: 3.6rem; padding-bottom: 3.5rem; }

/* ── Slim app bar (see dark-mode comment) ── */
[data-testid="stHeader"], header[data-testid="stHeader"] {
  height: 44px !important; min-height: 44px !important;
  background: rgba(255,255,255,0.92) !important;
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border-bottom: 1px solid #E2E8F0 !important; box-shadow: none !important;
}
[data-testid="stHeader"] [data-testid="stToolbar"] {
  height: 44px !important; min-height: 44px !important; padding: 0 10px !important;
}
[data-testid="stToolbar"]::before {
  content: "⚡ AI Energy Lab";
  position: absolute; left: 58px; top: 50%; transform: translateY(-50%);
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em;
  color: #C2410C !important; white-space: nowrap; pointer-events: none;
}
[data-testid="stHeader"] button, [data-testid="stHeader"] [data-testid="stMainMenu"] {
  border-radius: 6px !important;
}
[data-testid="stHeader"] button:hover { background: #FFF7ED !important; }

/* Section headers get an orange rule so they stand out from body copy */
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h1,
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h2,
[data-testid="stMarkdownContainer"] > [data-testid="stHeadingWithActionElements"] h3 {
  border-left: 4px solid #F97316; padding-left: 12px;
  margin: 1.2rem 0 0.5rem 0 !important; line-height: 1.25; color: #0F172A !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 { color: #0F172A !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span { color: #475569 !important; }
[data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
  color: #0F172A !important; font-weight: 600 !important;
}
/* Slider value bubble sits on the orange thumb, so it needs dark ink in both themes */
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] p { color: #0F172A !important; }
/* Primary buttons: deeper orange keeps white labels legible */
[data-testid="stBaseButton-primary"] { background: #C2410C !important; border-color: #C2410C !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] div,
[data-testid="stBaseButton-primary"] span { color: #FFFFFF !important; }
.rec-kicker { color: #C2410C !important; }
/* Tabs: explicit ink so labels never fade out */
[data-testid="stTab"] p, .stTabs [data-baseweb="tab"] p { color: #334155 !important; font-weight: 600 !important; }
[data-testid="stTab"]:hover p, .stTabs [data-baseweb="tab"]:hover p { color: #C2410C !important; }
[data-testid="stTab"][aria-selected="true"] p, .stTabs [data-baseweb="tab"][aria-selected="true"] p {
  color: #C2410C !important; font-weight: 700 !important;
}
/* Markdown tables */
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; }
[data-testid="stMarkdownContainer"] th {
  background: #F1F5F9 !important; color: #0F172A !important;
  border: 1px solid #E2E8F0 !important; padding: 8px 10px !important; text-align: left;
}
[data-testid="stMarkdownContainer"] td {
  border: 1px solid #E2E8F0 !important; padding: 8px 10px !important; color: #334155 !important;
}
.launch-header {
  background: linear-gradient(135deg, #FFFFFF 0%, #FFF7ED 55%, #FFEDD5 100%);
  border: 1px solid #FDBA74; border-left: 6px solid #F97316;
  border-radius: 14px; padding: 22px 26px; margin-bottom: 1.2rem;
  box-shadow: 0 4px 20px -2px rgba(249,115,22,0.12);
}
.launch-header .kicker { color: #C2410C; font-weight: 700; font-size: 0.82rem; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 4px; }
.launch-header h1 { margin: 0 0 6px 0; font-size: 2.2rem; color: #0F172A !important; font-weight: 800; letter-spacing: -0.025em; }
.launch-header p  { margin: 0; font-size: 0.98rem; color: #334155; max-width: 1050px; line-height: 1.5; }
[data-testid="stMetric"] {
  background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
  border-left: 4px solid #F97316 !important; border-radius: 12px !important;
  padding: 14px 18px !important; box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  transition: transform 0.18s ease, border-color 0.18s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #F97316 !important; box-shadow: 0 6px 18px rgba(249,115,22,0.12); }
[data-testid="stMetricLabel"] p { font-size: 0.88rem !important; color: #475569 !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 800 !important; color: #0F172A !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1.5px solid #E2E8F0; padding-bottom: 4px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px 8px 0 0; padding: 8px 16px; color: #334155; font-size: 0.92rem; font-weight: 600; transition: all 0.2s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #C2410C; background: #FFF7ED; border-color: #FDBA74; }
.stTabs [aria-selected="true"] { background: #FFF7ED !important; color: #C2410C !important; border-bottom: 2.5px solid #F97316 !important; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #F97316 !important; }
.white-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 18px 20px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); }
.white-card h4 { margin: 0 0 6px 0; color: #0F172A !important; font-size: 1.05rem; font-weight: 700; }
.white-card p  { margin: 0; color: #334155; font-size: 0.92rem; line-height: 1.5; }
.launch-card { background: #FFFFFF; border: 1px solid #E2E8F0; border-top: 4px solid #F97316; border-radius: 12px; padding: 18px 20px; height: 100%; box-shadow: 0 4px 12px rgba(0,0,0,0.04); transition: transform 0.18s ease, box-shadow 0.18s ease; }
.launch-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(249,115,22,0.14); }
.launch-card h3 { margin: 0 0 6px 0; color: #0F172A !important; font-size: 1.15rem; font-weight: 750; }
.launch-card p  { margin: 0 0 12px 0; color: #334155; font-size: 0.9rem; line-height: 1.45; }
.explainer-box { background: #FFF7ED; border: 1px solid #FDBA74; border-left: 4px solid #F97316; border-radius: 10px; padding: 12px 16px; margin: 10px 0 14px 0; font-size: 0.92rem; color: #7C2D12; display: flex; align-items: flex-start; gap: 10px; line-height: 1.5; }
.explainer-icon { font-size: 1.2rem; line-height: 1; }
/* Info tiles used by the "how to read" and glossary blocks */
.info-tile {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid #F97316;
  border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.info-tile h5 { margin: 0 0 4px 0; color: #0F172A !important; font-size: 0.95rem; font-weight: 700; }
.info-tile p  { margin: 0; color: #334155; font-size: 0.88rem; line-height: 1.5; }
.info-tile .formula {
  display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 5px;
  background: #FFF7ED; color: #C2410C;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem;
}
.badge { display: inline-block; padding: 3px 9px; font-size: 0.74rem; font-weight: 700; border-radius: 6px; letter-spacing: 0.04em; text-transform: uppercase; }
.badge-orange { background: #FFF7ED; color: #C2410C; border: 1px solid #FDBA74; }
.badge-blue   { background: #F0F9FF; color: #0369A1; border: 1px solid #BAE6FD; }
.badge-emerald{ background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.badge-rose   { background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }
div[data-testid="stDataFrame"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; overflow: hidden; }
[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0; }
html, body, .stApp, [data-testid="stSidebar"],
[data-testid="stMetric"], .white-card, .launch-card, .launch-header {
  transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
}
</style>"""
    st.markdown(css, unsafe_allow_html=True)
_inject_css(st.session_state.dark_mode)


# ---------------------------------------------------------------- Helpers
def fmt(value, unit: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}{(' ' + unit) if unit else ''}"


def has_cols(frame: pd.DataFrame, cols) -> bool:
    return not frame.empty and set(cols).issubset(frame.columns)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def pretty_table(frame: pd.DataFrame, rename: dict, digits: int = 2) -> pd.DataFrame:
    cols = [c for c in rename if c in frame.columns]
    out = frame[cols].rename(columns=rename).copy()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(digits)
    return out


def style_fig(fig, height: int = 400):
    dark = st.session_state.get("dark_mode", False)
    axis_text = "#94A3B8" if dark else "#475569"
    body_text = "#CBD5E1" if dark else "#334155"
    title_color = "#F1F5F9" if dark else "#0F172A"
    grid_color = "#272F42" if dark else "#F1F5F9"
    zero_color = "#334155" if dark else "#E2E8F0"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color=body_text),
        title_font=dict(size=15, color=title_color, family="sans-serif"),
        height=height,
        colorway=PALETTE,
        margin=dict(l=10, r=24, t=50, b=10),
        hoverlabel=dict(bgcolor="#1E293B" if dark else "#FFFFFF",
                        font_color="#E2E8F0" if dark else "#0F172A",
                        font_size=13, bordercolor="#334155" if dark else "#E2E8F0"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=12, color=axis_text), title_text=""),
    )
    for axis in (fig.update_xaxes, fig.update_yaxes):
        axis(
            title_font_size=13, title_font_color=axis_text,
            tickfont=dict(size=12, color=axis_text), automargin=True,
            gridcolor=grid_color, zerolinecolor=zero_color, linecolor=zero_color
        )
    return fig


def show(fig, height: int = 400):
    st.plotly_chart(style_fig(fig, height), **STRETCH)


def hbar(frame, y, x, x_label, title, color=None, barmode="group", height=None):
    data = frame.dropna(subset=[x])
    if data.empty:
        st.info(f"No data for “{title}”.")
        return
    fig = px.bar(data, x=x, y=y, color=color, orientation="h", title=title, barmode=barmode,
                 labels={x: x_label, y: ""}, color_discrete_sequence=PALETTE)
    if color is None:
        fig.update_traces(marker_color=ORANGE_PRIMARY)
    fig.update_traces(texttemplate="%{x:,.3g}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(categoryorder="total ascending")
    show(fig, height or max(280, 60 + 44 * len(data)))


def benchmark_only(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"model", "gen_tok_s"}.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    out["gen_tok_s"] = pd.to_numeric(out["gen_tok_s"], errors="coerce")
    return out[out["model"].notna() & out["gen_tok_s"].notna()].reset_index(drop=True)


def power_basis(frame: pd.DataFrame):
    if "method" not in frame.columns or frame["method"].dropna().empty:
        return "unknown", "Hardware power source unverified."
    methods = sorted(frame["method"].dropna().astype(str).unique())
    joined = ", ".join(methods)
    if any("cpu_util_model" in m for m in methods):
        return "estimate", f"Power basis: **{joined}** (Local hardware CPU-utilization model)."
    return "measured", f"Power basis: **{joined}** (RAPL CPU package / NVML GPU hardware counters)."


ESTIMATE_COLUMNS = {
    "option": "Architecture Candidate", "kind_label": "Type", "servers": "Servers Needed",
    "utilisation_pct": "Avg Utilization (%)", "kwh": "Energy (kWh/yr)",
    "cost_usd": "Annual Cost ($)", "kg_co2": "Carbon (kg CO₂/yr)",
    "wh_per_query": "Energy/Query (Wh)", "note": "Basis",
}


# ---------------------------------------------------------------- Data Ingestion
raw = pd.DataFrame()
try:
    raw = load_results()
except Exception as exc:
    st.error(f"Could not load benchmark files: {exc}")
bench = benchmark_only(raw)
measured = not bench.empty

profiles = placeholder_profiles()
if measured:
    try:
        try:
            built = build_profiles(bench, quality_csv=str(QUALITY_AUTO))
        except TypeError:
            built = build_profiles(bench)
        if built:
            profiles = built
        else:
            measured = False
    except Exception as exc:
        measured = False
        st.warning(f"Could not construct model profiles: {exc}")

cloud: list = []
cloud_source_note = "none"
if CLOUD_PATH.exists():
    try:
        loaded = json.loads(CLOUD_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, list) and loaded:
            cloud = [c for c in loaded if isinstance(c, dict)]
            cloud_source_note = "custom"
    except (json.JSONDecodeError, OSError):
        pass

if not cloud and CLOUD_BACKUP.exists():
    try:
        loaded_backup = json.loads(CLOUD_BACKUP.read_text(encoding="utf-8"))
        if isinstance(loaded_backup, list) and loaded_backup:
            cloud = [c for c in loaded_backup if isinstance(c, dict)]
            cloud_source_note = "reference baseline"
    except (json.JSONDecodeError, OSError):
        pass

# ---------------------------------------------------------------- Sidebar Controls
DEFAULTS = dict(users=15000, active=0.20, qpd=5.0, days=240, hours=12, peak=8.0, tin=1500, tout=350,
                ci=0.71, price=0.10, tu=0.70, aon=True, bg=1.0, capex=1500, with_cloud=True)
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.toggle("🌙 Dark mode", key="dark_mode",
              help="Click to switch the whole dashboard between light and dark themes.")
    st.divider()
    st.markdown("### ⚙️ Scenario Parameters")
    st.caption("Change settings to immediately update projections.")

    # Data status pill
    _data_status = "🟢 Real data" if measured else "🟡 Placeholder"
    _data_tip = f"{len(bench):,} telemetry rows · {bench['model'].nunique() if measured else 5} models" if measured else "Run benchmarks first"
    st.markdown(
        f"<div class='status-pill' style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; "
        f"padding:7px 12px; font-size:0.84rem; color:#166534; margin-bottom:8px;'>"
        f"{_data_status} &nbsp;·&nbsp; {_data_tip}</div>",
        unsafe_allow_html=True
    )
    st.button("↺ Reset Defaults", on_click=lambda: st.session_state.update(DEFAULTS), **STRETCH)

    with st.expander("📘 Glossary & method", expanded=False):
        st.markdown("""
        <div class="info-tile">
          <h5>Wh / query</h5>
          <p>Electricity one user question costs, prefill plus generated tokens, across the whole fleet.</p>
          <span class="formula">annual kWh ÷ annual queries</span>
        </div>
        <div class="info-tile">
          <h5>J / token</h5>
          <p>Active energy per generated token. Lower is greener: it separates model efficiency from traffic volume.</p>
          <span class="formula">(active W − idle W) ÷ tok/s</span>
        </div>
        <div class="info-tile">
          <h5>Idle tax (×)</h5>
          <p>How many times more electricity a 24/7 server burns compared with the same box duty-cycled.</p>
          <span class="formula">always-on kWh ÷ duty-cycled kWh</span>
        </div>
        <div class="info-tile">
          <h5>EEI (points / J)</h5>
          <p>Energy Efficiency Index: answer quality bought per joule. Higher is better.</p>
          <span class="formula">quality score ÷ (J / token)</span>
        </div>
        <div class="info-tile">
          <h5>Pareto optimal</h5>
          <p>A model is on the frontier when nothing else beats it on both quality and energy per token.</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Measured rows come from local hardware counters (RAPL / NVML). Until benchmarks are run, "
            "the dashboard falls back to documented placeholder profiles, flagged 🟡 above."
        )

    with st.expander("👥 Community & Workload", expanded=True):
        users = st.number_input("Community (Users)", 100, 1_000_000, step=500, key="users")
        active = st.slider("Daily Active Share", 0.01, 1.0, step=0.01, key="active")
        qpd = st.number_input("Queries / Active User / Day", 1.0, 100.0, step=1.0, key="qpd")
        days = st.number_input("Operating Days / Year", 1, 365, key="days")
        hours = st.number_input("Service Hours / Day", 1, 24, key="hours")
        peak = st.number_input("Peak-to-Average Load Multiplier", 1.0, 50.0, step=0.5, key="peak")
        tin = st.number_input("Input Tokens / Query", 10, 32000, step=100, key="tin")
        tout = st.number_input("Output Tokens / Query", 10, 4000, step=25, key="tout")

    with st.expander("⚡ Electricity Tariff & Carbon", expanded=True):
        ci = st.number_input("Grid Carbon (kg CO₂/kWh)", 0.0, 1.5, step=0.01, key="ci")
        price = st.number_input("Electricity Rate ($/kWh)", 0.0, 1.0, step=0.01, key="price")

    with st.expander("🖥️ Fleet Hardware & Duty Cycle", expanded=True):
        tu = st.slider("Target Peak Utilization", 0.2, 0.95, step=0.05, key="tu")
        aon = st.checkbox("Servers Powered 24/7 (Always-On)", key="aon")
        bg = st.number_input("Batching Concurrency Gain", 1.0, 16.0, step=0.5, key="bg")
        capex = st.number_input("Hardware Cost / Server ($)", 0, 100000, step=100, key="capex")

    with st.expander("☁️ Cloud Baselines", expanded=False):
        use_cloud = st.checkbox("Include Cloud Options", key="with_cloud")

W = C.Workload(users, active, qpd, days, tin, tout, hours, peak)
G = C.Grid(carbon_kg_per_kwh=ci, price_per_kwh=price)


def estimate(workload, grid, tu_, aon_, bg_, capex_, with_cloud=True):
    rows, problems = [], []
    for p in profiles:
        try:
            rows.append(C.local_estimate(workload, p, grid, tu_, aon_, bg_, capex_))
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            problems.append(f"{getattr(p, 'name', 'profile')}: {exc!r}")
    if with_cloud and use_cloud:
        for o in cloud:
            try:
                rows.append(C.cloud_estimate(workload, o, grid, tu_, aon_))
            except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
                problems.append(f"{o.get('name', 'cloud option')}: missing field {exc!r}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["kind_label"] = frame["kind"].map(KIND_LABELS).fillna(frame["kind"])
        if "utilisation" in frame.columns:
            frame["utilisation_pct"] = (frame["utilisation"] * 100).where(frame["servers"] > 0)
        frame["servers"] = frame["servers"].where(frame["servers"] > 0)
    return frame, problems


# ---------------------------------------------------------------- Model statistics (single source of truth)
def model_key(profile_name: str) -> str:
    """'qwen3:1.7b (ollama)' -> 'qwen3:1.7b' (matches the `model` column in the CSVs)."""
    return str(profile_name).rsplit(" (", 1)[0]


def md_inline(text: str) -> str:
    """Render **bold** inside raw HTML blocks (Markdown is not parsed inside HTML)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", str(text))


def build_model_stats() -> pd.DataFrame:
    """One row per benchmarked model, with the campus case study run for each one.

    Active energy per token is the median of the per-run `j_per_out_token` column
    (energy above idle ÷ output tokens). Campus figures use the fixed case-study
    workload (scenarios/campus_chatbot.py) and electricity only (no hardware cost).
    """
    if not measured:
        return pd.DataFrame()
    j_med, runs, wall = {}, {}, {}
    if "j_per_out_token" in bench.columns:
        j_med = pd.to_numeric(bench["j_per_out_token"], errors="coerce").groupby(bench["model"]).median().dropna().to_dict()
    runs = bench.groupby("model").size().to_dict()
    if "wall_s" in bench.columns:
        wall = pd.to_numeric(bench["wall_s"], errors="coerce").groupby(bench["model"]).median().dropna().to_dict()
    manual = {}
    mdf = read_csv_safe(QUALITY_MANUAL)
    if has_cols(mdf, ["model", "manual_score"]):
        manual = pd.to_numeric(mdf["manual_score"], errors="coerce").groupby(mdf["model"]).mean().dropna().to_dict()

    rows = []
    for p in profiles:
        key = model_key(p.name)
        j = j_med.get(key)
        j_src = "measured"
        if j is None or not j > 0:
            j, j_src = max(p.active_w - p.idle_w, 0) / max(p.gen_tok_s, 0.1), "derived"
        duty = C.local_estimate(CW, p, CG, 0.7, False, 1.0, 0.0)
        always = C.local_estimate(CW, p, CG, 0.7, True, 1.0, 0.0)
        q = p.quality if p.quality is not None and pd.notna(p.quality) else None
        rows.append(dict(
            model=key, profile=p.name, params_b=p.params_b, runs=runs.get(key, 0), wall_s=wall.get(key),
            tok_s=p.gen_tok_s, active_w=p.active_w, idle_w=p.idle_w, j_per_tok=float(j), j_src=j_src,
            quality=q, manual=manual.get(key),
            eei=S.compute_efficiency_index(q, float(j)) if q is not None else None,
            duty_kwh=duty["kwh"], duty_usd=duty["cost_usd"], duty_co2=duty["kg_co2"], servers=duty["servers"],
            always_kwh=always["kwh"], always_usd=always["cost_usd"], always_co2=always["kg_co2"],
        ))
    out = pd.DataFrame(rows)
    out["idle_tax"] = out["always_kwh"] / out["duty_kwh"].where(out["duty_kwh"] > 0)
    return out


ms = build_model_stats()
POWER_KIND, POWER_NOTE = power_basis(bench) if measured else ("none", "No benchmark data yet.")
CLOUD_PLACEHOLDER = any("placeholder" in str(o.get("name", "")).lower() for o in cloud)

# Headline facts — every number below is computed, never typed in
small = large = None
if not ms.empty:
    ranked = ms.dropna(subset=["eei"]).sort_values("eei", ascending=False)
    small = (ranked if not ranked.empty else ms.sort_values("j_per_tok")).iloc[0]
    large = ms.sort_values(["params_b", "j_per_tok"], ascending=False, na_position="last").iloc[0]
    if small["model"] == large["model"]:
        large = None
tax_min = ms["idle_tax"].min() if not ms.empty else None
tax_max = ms["idle_tax"].max() if not ms.empty else None
tax_avg = ms["idle_tax"].mean() if not ms.empty else None


def fmt_q(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.0f}"


def compare_word(a, b, tol: float = 2.0) -> str:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return "n/a"
    return "similar" if abs(a - b) <= tol else ("higher" if a > b else "lower")


FINDINGS: list[str] = []
if small is not None and large is not None:
    pct = (1 - small["j_per_tok"] / large["j_per_tok"]) * 100
    FINDINGS.append(
        f"**{small['model']}** uses **{pct:.0f}% less active energy per token** than **{large['model']}** "
        f"({small['j_per_tok']:.2f} vs {large['j_per_tok']:.2f} J/token) and runs "
        f"{small['tok_s'] / large['tok_s']:.1f}× faster. Quality: automatic {fmt_q(small['quality'])} vs "
        f"{fmt_q(large['quality'])}, human review {fmt_q(small['manual'])} vs {fmt_q(large['manual'])}."
    )
if tax_min is not None:
    FINDINGS.append(
        f"Keeping servers on 24/7 uses **{tax_min:.1f}–{tax_max:.1f}×** the energy of a "
        f"{CW.hours_per_day}h/day, {CW.days}-day schedule — idle power dominates always-on deployments."
    )
if small is not None and large is not None:
    FINDINGS.append(
        f"Campus chatbot for a year: **{small['model']} on a schedule ≈ {small['duty_kwh']:,.0f} kWh** vs "
        f"**{large['model']} left on 24/7 ≈ {large['always_kwh']:,.0f} kWh** "
        f"({large['always_kwh'] / small['duty_kwh']:.1f}× more)."
    )

# Base estimates for the sidebar scenario (local only, so placeholder cloud rows never become a headline)
res_base, _ = estimate(W, G, tu, aon, bg, capex, with_cloud=True)
res_local = res_base[res_base["kind"] == "local"] if not res_base.empty else res_base

# ---------------------------------------------------------------- Landing page (shown before the dashboard)
st.session_state.setdefault("entered", False)


def _toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


def _enter_dashboard():
    st.session_state.entered = True


def _go_home():
    st.session_state.entered = False


def render_landing() -> None:
    dark = st.session_state.dark_mode
    P = dict(
        surface="#111A2E" if dark else "#FFFFFF",
        surface2="#16213A" if dark else "#FFF7ED",
        border="#24324D" if dark else "#E2E8F0",
        text="#F1F5F9" if dark else "#0F172A",
        muted="#A9B7CB" if dark else "#475569",
        accent="#FB923C" if dark else "#EA580C",
        hero=("radial-gradient(900px 380px at 0% 0%, rgba(249,115,22,0.22), transparent 60%),"
              "linear-gradient(135deg, #111A2E 0%, #0F172A 100%)") if dark else
             ("radial-gradient(900px 380px at 0% 0%, rgba(249,115,22,0.18), transparent 60%),"
              "linear-gradient(135deg, #FFFFFF 0%, #FFF7ED 100%)"),
        shadow="0 10px 30px rgba(0,0,0,0.35)" if dark else "0 10px 30px rgba(15,23,42,0.06)",
    )
    css = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.stApp .lp-nav { display:flex; align-items:center; gap:10px; padding:4px 0 2px 0; }
.stApp .lp-logo { font-size:1.15rem !important; font-weight:800 !important; color:__text__ !important; letter-spacing:-0.01em; }
.stApp .lp-pill { font-size:0.72rem !important; font-weight:700 !important; color:__accent__ !important;
  border:1px solid #F97316; border-radius:999px; padding:2px 10px; text-transform:uppercase; letter-spacing:0.06em; }
.stApp .lp-hero { background:__hero__; border:1px solid __border__; border-radius:20px; padding:46px 44px 40px 44px;
  box-shadow:__shadow__; margin:10px 0 18px 0; }
.stApp .lp-kicker { color:__accent__ !important; font-weight:700 !important; font-size:0.8rem !important;
  letter-spacing:0.12em; text-transform:uppercase; margin-bottom:10px; }
.stApp .lp-hero h1 { color:__text__ !important; font-size:3rem !important; line-height:1.08 !important;
  font-weight:850 !important; letter-spacing:-0.035em; margin:0 0 14px 0 !important; padding:0 !important; border:none !important; }
.stApp .lp-hero h1 span { color:#F97316 !important; }
.stApp .lp-hero p { color:__muted__ !important; font-size:1.1rem !important; line-height:1.6 !important; max-width:720px; margin:0 !important; }
.stApp .lp-stat { background:__surface__; border:1px solid __border__; border-radius:14px; padding:18px 20px; height:100%; box-shadow:__shadow__; }
.stApp .lp-stat .v { color:__text__ !important; font-size:1.9rem !important; font-weight:800 !important; letter-spacing:-0.02em; line-height:1.1; }
.stApp .lp-stat .l { color:__muted__ !important; font-size:0.85rem !important; margin-top:4px; }
.stApp .lp-section { color:__text__ !important; font-size:1.45rem !important; font-weight:800 !important;
  letter-spacing:-0.02em; margin:30px 0 4px 0; }
.stApp .lp-sub { color:__muted__ !important; font-size:0.95rem !important; margin:0 0 14px 0; }
.stApp .lp-card { background:__surface__; border:1px solid __border__; border-radius:14px; padding:20px 22px; height:100%;
  box-shadow:__shadow__; transition:transform .18s ease, border-color .18s ease; }
.stApp .lp-card:hover { transform:translateY(-3px); border-color:#F97316; }
.stApp .lp-card .ic { font-size:1.6rem; margin-bottom:6px; }
.stApp .lp-card .n { color:__accent__ !important; font-size:0.75rem !important; font-weight:800 !important; letter-spacing:0.1em; }
.stApp .lp-card h4 { color:__text__ !important; font-size:1.05rem !important; font-weight:750 !important; margin:4px 0 6px 0 !important; }
.stApp .lp-card p { color:__muted__ !important; font-size:0.9rem !important; line-height:1.5 !important; margin:0 !important; }
.stApp .lp-chip { display:inline-block; background:__surface2__; border:1px solid __border__; color:__text__ !important;
  border-radius:999px; padding:6px 14px; margin:0 8px 8px 0; font-size:0.85rem !important; font-weight:600 !important; }
.stApp .lp-cta { background:__surface2__; border:1px solid #F97316; border-radius:18px; padding:28px 30px; margin-top:28px; }
.stApp .lp-cta h3 { color:__text__ !important; font-size:1.4rem !important; font-weight:800 !important; margin:0 0 4px 0 !important; border:none !important; padding:0 !important; }
.stApp .lp-cta p { color:__muted__ !important; margin:0 !important; }
.stApp .lp-foot { color:__muted__ !important; font-size:0.8rem !important; text-align:center; margin:26px 0 6px 0; }
</style>"""
    for k, v in P.items():
        css = css.replace(f"__{k}__", v)
    st.markdown(css, unsafe_allow_html=True)

    # Nav bar
    n1, n2 = st.columns([6, 1])
    with n1:
        st.markdown("<div class='lp-nav'><span class='lp-logo'>⚡ AI Energy Lab</span>"
                    "<span class='lp-pill'>Project 74</span></div>", unsafe_allow_html=True)
    with n2:
        st.button("☀️ Light" if dark else "🌙 Dark", on_click=_toggle_theme, key="lp_theme", **STRETCH)

    # Hero
    st.markdown("""
<div class="lp-hero">
  <div class="lp-kicker">Energy · Cost · Carbon of AI systems</div>
  <h1>What does an AI answer<br><span>really cost?</span></h1>
  <p>We benchmarked language models on our own hardware, priced cloud alternatives, and built a
     calculator that turns any planned AI deployment into kWh, dollars and CO₂.</p>
</div>""", unsafe_allow_html=True)

    c1, c2, _sp = st.columns([1.2, 1.2, 3.6])
    with c1:
        st.button("🚀 Launch dashboard", type="primary", on_click=_enter_dashboard, key="lp_enter_top", **STRETCH)
    with c2:
        st.link_button("⭐ View on GitHub", "https://github.com/adityajyotir2007-hub/ai-energy-lab", **STRETCH)

    # Stats strip — all computed from the benchmark files
    if not ms.empty:
        stats = [
            (f"{len(bench):,}", "benchmark runs"),
            (f"{len(ms)}", "models compared"),
            (f"{ms['j_per_tok'].min():.2f}–{ms['j_per_tok'].max():.2f}", "joules per token"),
            (f"{tax_min:.1f}–{tax_max:.1f}×", "energy cost of 24/7 servers"),
        ]
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        for col, (v, l) in zip(st.columns(4), stats):
            with col:
                st.markdown(f"<div class='lp-stat'><div class='v'>{v}</div><div class='l'>{l}</div></div>",
                            unsafe_allow_html=True)

    # How it works
    st.markdown("<div class='lp-section'>How it works</div>"
                "<div class='lp-sub'>Three steps from raw measurements to a deployment decision.</div>",
                unsafe_allow_html=True)
    steps = [
        ("01", "📏", "Measure", "Run each model through the same prompts with Ollama, timing every token and logging power."),
        ("02", "🧮", "Model", "Feed the measurements into a calculator that sizes a server fleet for any workload."),
        ("03", "✅", "Decide", "Compare local vs cloud, small vs large, scheduled vs always-on — in kWh, $ and CO₂."),
    ]
    for col, (n, ic, h, p_) in zip(st.columns(3), steps):
        with col:
            st.markdown(f"<div class='lp-card'><div class='n'>STEP {n}</div><div class='ic'>{ic}</div>"
                        f"<h4>{h}</h4><p>{p_}</p></div>", unsafe_allow_html=True)

    # Headline finding with a real chart
    if small is not None and large is not None:
        st.markdown("<div class='lp-section'>The headline finding</div>"
                    f"<div class='lp-sub'>A campus chatbot for one academic year ({CW.users:,} students, {CW.days} days).</div>",
                    unsafe_allow_html=True)
        f1, f2 = st.columns([1.5, 1])
        with f1:
            hl = pd.DataFrame({
                "Setup": [f"{small['model']} · scheduled", f"{large['model']} · scheduled", f"{large['model']} · 24/7"],
                "kWh": [small["duty_kwh"], large["duty_kwh"], large["always_kwh"]],
            })
            fig_hl = px.bar(hl, x="kWh", y="Setup", orientation="h", text="kWh",
                            color="Setup", color_discrete_sequence=[EMERALD, AMBER, ROSE],
                            title="Energy per year (kWh)")
            fig_hl.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
            fig_hl.update_layout(showlegend=False)
            fig_hl.update_yaxes(title="", categoryorder="array", categoryarray=list(hl["Setup"])[::-1])
            show(fig_hl, 260)
        with f2:
            ratio = large["always_kwh"] / small["duty_kwh"]
            st.markdown(f"""
<div class="lp-card">
  <div class="n">SMALL MODEL, RUN WELL</div>
  <h4>{ratio:.1f}× less energy</h4>
  <p><strong>{small['model']}</strong> on a {CW.hours_per_day}h schedule uses
     <strong>{small['duty_kwh']:,.0f} kWh</strong> a year. <strong>{large['model']}</strong> left on 24/7 uses
     <strong>{large['always_kwh']:,.0f} kWh</strong> — about
     <strong>${large['always_usd'] - small['duty_usd']:,.0f}</strong> more in electricity.</p>
</div>""", unsafe_allow_html=True)

    # Features
    st.markdown("<div class='lp-section'>Inside the dashboard</div>"
                "<div class='lp-sub'>Nine tools, one set of numbers.</div>", unsafe_allow_html=True)
    feats = [
        ("🤖", "Green Auditor", "Answer four questions, get a deployment plan."),
        ("⚖️", "Pareto Frontier", "Which models give the most quality per joule."),
        ("🏫", "Campus Chatbot", "A full academic year, with a sleep-schedule slider."),
        ("🧮", "Calculator", "Size a fleet and price it for any workload."),
        ("📊", "Telemetry", "Every benchmark run — speed, runtime and energy."),
        ("⚔️", "Model Arena", "Run two models live and judge the answers."),
    ]
    for row in (feats[:3], feats[3:]):
        for col, (ic, h, p_) in zip(st.columns(3), row):
            with col:
                st.markdown(f"<div class='lp-card'><div class='ic'>{ic}</div><h4>{h}</h4><p>{p_}</p></div>",
                            unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("<div class='lp-section'>Built with</div>", unsafe_allow_html=True)
    stack = ["Python", "Ollama", "Hugging Face Transformers", "CodeCarbon", "psutil", "Pandas", "Plotly", "Streamlit"]
    st.markdown("".join(f"<span class='lp-chip'>{s}</span>" for s in stack), unsafe_allow_html=True)

    # Closing CTA
    st.markdown("""
<div class="lp-cta">
  <h3>Plan your own AI deployment</h3>
  <p>Set your users, hours and grid — see the energy, cost and carbon in seconds.</p>
</div>""", unsafe_allow_html=True)
    b1, _sp2 = st.columns([1.2, 4.8])
    with b1:
        st.button("🚀 Launch dashboard", type="primary", on_click=_enter_dashboard, key="lp_enter_bottom", **STRETCH)

    note = ("Power figures are estimated from CPU utilisation; runtime and token speed are measured."
            if POWER_KIND == "estimate" else "Power read from hardware energy counters.")
    st.markdown(f"<div class='lp-foot'>{note}<br>AI Energy Lab · Project 74</div>", unsafe_allow_html=True)


if not st.session_state.entered:
    render_landing()
    st.stop()

# ---------------------------------------------------------------- Top bar: home, data banner, theme toggle
_tb1, _tb_home, _tb2 = st.columns([5, 1, 1])
with _tb_home:
    st.button("🏠 Home", on_click=_go_home, key="home_btn", **STRETCH)
with _tb2:
    st.button("☀️ Light mode" if st.session_state.dark_mode else "🌙 Dark mode",
              on_click=_toggle_theme, key="theme_btn", **STRETCH)
with _tb1:
    if POWER_KIND == "estimate":
        st.caption("⚠️ **Power is estimated** from CPU utilisation on this machine (no RAPL/NVML counters). "
                   "Runtime and tokens/s are measured. See *Telemetry* for how to get metered watts.")
    elif POWER_KIND == "measured":
        st.caption("✅ Power read from hardware energy counters.")

# ---------------------------------------------------------------- Primary Tabs
tab_launch, tab_wizard, tab_pareto, tab_campus, tab_calc, tab_telemetry, tab_quality, tab_arena, tab_report = st.tabs([
    "🚀 Launchpad",
    "🤖 Green Auditor",
    "⚖️ Pareto Frontier",
    "🏫 Campus Chatbot",
    "🧮 Calculator",
    "📊 Telemetry",
    "🎯 Quality",
    "⚔️ Model Arena",
    "📄 Report"
])


# ================================================================ 1. 🚀 Launchpad
with tab_launch:
    st.markdown("""
    <div class="launch-header">
      <div class="kicker">⚡ Project 74 · AI Energy Lab</div>
      <h1>AI Systems Energy, Cost &amp; Carbon Analysis</h1>
      <p>Every AI answer costs electricity and money. We benchmarked local models on our own machine,
         compared cloud options, and tested whether <strong>a small model run well</strong> beats
         <strong>a large model run wastefully</strong>.</p>
    </div>
    """, unsafe_allow_html=True)

    _n_runs = len(bench) if measured else 0
    _n_models = bench["model"].nunique() if measured else 0
    s_col1, s_col2, s_col3, s_col4 = st.columns([1.1, 1.1, 1.4, 2.0])
    with s_col1:
        st.markdown(f"<span class='badge badge-orange'>Runs</span> **{_n_runs:,}** benchmarks", unsafe_allow_html=True)
    with s_col2:
        st.markdown(f"<span class='badge badge-blue'>Models</span> **{_n_models}** compared", unsafe_allow_html=True)
    with s_col3:
        _win = small["model"] if small is not None else "—"
        st.markdown(f"<span class='badge badge-emerald'>Best EEI</span> **{_win}**", unsafe_allow_html=True)
    with s_col4:
        _tax = f"{tax_min:.1f}–{tax_max:.1f}×" if tax_min is not None else "—"
        st.markdown(f"<span class='badge badge-rose'>Idle trap</span> 24/7 uses **{_tax}** the energy", unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    if not ms.empty:
        _best = ms.sort_values("duty_kwh").iloc[0]
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("Campus chatbot · best case", f"{_best['duty_kwh']:,.0f} kWh/yr",
                   help=f"{_best['model']}, {CW.hours_per_day}h/day, {CW.days} days")
        _m2.metric("Electricity cost", f"${_best['duty_usd']:,.0f}/yr",
                   help=f"At ${CG.price_per_kwh:.2f}/kWh, hardware excluded")
        _m3.metric("CO₂", f"{_best['duty_co2']:,.0f} kg/yr", help=f"At {CG.carbon_kg_per_kwh} kg CO₂/kWh")
        _fast = ms.sort_values("tok_s", ascending=False).iloc[0]
        _m4.metric("Fastest model", f"{_fast['tok_s']:.1f} tok/s", help=f"{_fast['model']} (median)")
    else:
        st.info("No benchmark data found in `data/results/`. Run `python -m aienergy.bench_ollama` first.")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Core finding: computed head-to-head
    if small is not None and large is not None:
        st.markdown("### 🏆 Small Model Run Well vs. Large Model Run Wastefully")
        st.caption(f"Campus case study: {CW.users:,} users · {CW.days} days · electricity at ${CG.price_per_kwh:.2f}/kWh.")
        e_ratio = large["always_kwh"] / small["duty_kwh"]
        usd_gap = large["always_usd"] - small["duty_usd"]
        j_ratio = large["j_per_tok"] / small["j_per_tok"]
        sp_ratio = small["tok_s"] / large["tok_s"]
        qa, qm = compare_word(small["quality"], large["quality"]), compare_word(small["manual"], large["manual"])

        def _row(icon, label, value):
            return f"<div>{icon} <strong>{label}:</strong> {value}</div>"

        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.markdown(f"""
<div class="cmp-card cmp-good" style="background:#FFFFFF; border:2px solid #10B981; border-radius:12px; padding:18px 20px;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h4 style="margin:0; color:#059669;">🌟 {small['model']}</h4>
    <span class="badge badge-emerald">Run well</span>
  </div>
  <p style="margin:6px 0 10px 0; color:#64748B; font-size:0.9rem;">Best quality-per-joule · sleeps outside {CW.hours_per_day}h service</p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.9rem; color:#0F172A;">
    {_row("🎯", "Quality", f"{fmt_q(small['quality'])} auto · {fmt_q(small['manual'])} human")}
    {_row("⚡", "Speed", f"{small['tok_s']:.1f} tok/s")}
    {_row("🔋", "Energy", f"{small['j_per_tok']:.2f} J/token")}
    {_row("🌱", "Year", f"{small['duty_kwh']:,.0f} kWh")}
    {_row("💰", "Cost", f"${small['duty_usd']:,.0f}/yr")}
    {_row("🖥️", "Servers", f"{int(small['servers'])}")}
  </div>
  <div class="cmp-foot-good" style="margin-top:10px; padding-top:8px; border-top:1px solid #E2E8F0; color:#047857; font-size:0.85rem; font-weight:600;">
    ✅ {j_ratio:.1f}× less energy per token, {sp_ratio:.1f}× faster.
  </div>
</div>""", unsafe_allow_html=True)
        with h_col2:
            st.markdown(f"""
<div class="cmp-card cmp-bad" style="background:#FFFFFF; border:2px solid #E11D48; border-radius:12px; padding:18px 20px;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h4 style="margin:0; color:#E11D48;">⚠️ {large['model']}</h4>
    <span class="badge badge-rose">Run wastefully</span>
  </div>
  <p style="margin:6px 0 10px 0; color:#64748B; font-size:0.9rem;">Largest model tested · left on 24/7, all year</p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:0.9rem; color:#0F172A;">
    {_row("🎯", "Quality", f"{fmt_q(large['quality'])} auto · {fmt_q(large['manual'])} human")}
    {_row("⚡", "Speed", f"{large['tok_s']:.1f} tok/s")}
    {_row("🔋", "Energy", f"{large['j_per_tok']:.2f} J/token")}
    {_row("🌱", "Year", f"{large['always_kwh']:,.0f} kWh")}
    {_row("💰", "Cost", f"${large['always_usd']:,.0f}/yr")}
    {_row("🖥️", "Servers", f"{int(large['servers'])}")}
  </div>
  <div class="cmp-foot-bad" style="margin-top:10px; padding-top:8px; border-top:1px solid #E2E8F0; color:#BE123C; font-size:0.85rem; font-weight:600;">
    ❌ {e_ratio:.1f}× the energy and ${usd_gap:,.0f} more per year.
  </div>
</div>""", unsafe_allow_html=True)

        # Where does the waste come from? (model choice vs schedule)
        from_model = large["duty_kwh"] - small["duty_kwh"]
        from_sched = large["always_kwh"] - large["duty_kwh"]
        split = pd.DataFrame({
            "Cause": ["Bigger model", "Always-on schedule"],
            "kWh": [max(from_model, 0), max(from_sched, 0)],
        })
        fig_split = px.bar(split, x="kWh", y="Cause", orientation="h", text="kWh",
                           title="Where the extra energy comes from (kWh/yr)",
                           color="Cause", color_discrete_sequence=[ROSE, AMBER])
        fig_split.update_traces(texttemplate="%{x:,.0f}", textposition="outside", cliponaxis=False)
        fig_split.update_layout(showlegend=False)
        fig_split.update_yaxes(title="")
        show(fig_split, 220)
        st.caption(f"Quality is **{qa}** on the automatic check and **{qm}** on human review for "
                   f"{small['model']} vs {large['model']}.")

    # Takeaways
    if FINDINGS:
        st.markdown("### 📌 Key Takeaways")
        cols = st.columns(3)
        titles = ["1. Smaller can be enough", "2. The 24/7 idle trap", "3. Cloud vs local"]
        bodies = FINDINGS[:2] + [
            "Cloud APIs have no idle draw on your side, so they suit low, bursty traffic. "
            + ("**Cloud prices here are placeholders** — add real quotes to `data/cloud_options.json`."
               if CLOUD_PLACEHOLDER else "Compare in the Calculator tab.")
        ]
        for col, t, b in zip(cols, titles, bodies):
            with col:
                st.markdown(f"<div class='white-card'><h4>{t}</h4><p>{md_inline(b)}</p></div>",
                            unsafe_allow_html=True)

    # How to read
    with st.expander("🧭 How to read these numbers", expanded=False):
        _g1, _g2, _g3, _g4 = st.columns(4)
        tiles = [
            ("⚡ Energy per query", "What one question costs, prompt + answer.", "Wh = kWh/yr ÷ queries/yr"),
            ("🔌 Idle tax", "Servers left on overnight burn power waiting.", "always-on ÷ scheduled"),
            ("⚖️ EEI", "Quality points per joule. Higher is better.", "quality ÷ (J/token)"),
            ("🌱 Carbon", "Energy × grid carbon intensity.", "kg CO₂ = kWh × kg/kWh"),
        ]
        for col, (h, p_, f_) in zip([_g1, _g2, _g3, _g4], tiles):
            with col:
                st.markdown(f"<div class='info-tile'><h5>{h}</h5><p>{p_}</p><span class='formula'>{f_}</span></div>",
                            unsafe_allow_html=True)

    # Provenance
    with st.expander("🔬 Where the data comes from", expanded=False):
        _prompt_kinds = bench["prompt_id"].nunique() if (measured and "prompt_id" in bench.columns) else "—"
        _cloud_txt = (f"{len(cloud)} options — **placeholder prices, not verified**" if CLOUD_PLACEHOLDER
                      else f"{len(cloud)} options from `data/cloud_options.json`")
        st.markdown(f"""
| Input | Source |
| --- | --- |
| Benchmarks | {_n_runs:,} runs · {_n_models} models · {_prompt_kinds} prompts (Ollama, this machine) |
| Power | {POWER_NOTE} |
| Quality | Automatic rule score + human audit ({len(read_csv_safe(QUALITY_MANUAL))} ratings) |
| Cloud | {_cloud_txt} |
| Campus case | {CW.users:,} users · {CW.active_frac:.0%} active · {CW.queries_per_active_day} q/day · {CW.days} days · {CW.hours_per_day} h/day |
| Calculator | Sidebar settings: grid {ci} kg CO₂/kWh · ${price:.2f}/kWh |
""")

    st.markdown("### 🎯 Explore")
    l_cols = st.columns(4)
    cards = [("🤖", "Green Auditor", "4 questions → a deployment plan."),
             ("⚖️", "Pareto Frontier", "Quality vs energy per token."),
             ("🏫", "Campus Chatbot", "One academic year, with a sleep-schedule slider."),
             ("⚔️", "Model Arena", "Run two models live and judge them.")]
    for col, (ic, h, p_) in zip(l_cols, cards):
        with col:
            st.markdown(f"<div class='launch-card'><div style='font-size:1.8rem; margin-bottom:4px;'>{ic}</div>"
                        f"<h3>{h}</h3><p>{p_}</p></div>", unsafe_allow_html=True)


# ================================================================ 2. 🤖 AI Green Auditor
with tab_wizard:
    st.subheader("🤖 Green Auditor")
    st.caption("Answer 4 questions. The advice uses your benchmark results.")

    w_col1, w_col2 = st.columns([1, 1.25])
    with w_col1:
        st.markdown("##### 1. Your situation")
        wiz_case = st.selectbox(
            "Use case",
            ["Campus Student Assistant (Coursework, policy, FAQs)",
             "Faculty Research Assistant (Document synthesis, coding)",
             "Departmental Helpdesk (Low-volume administrative triage)",
             "High-Throughput Campus Lab (Heavy concurrent student load)"],
            key="w_case"
        )
        wiz_vol = st.select_slider(
            "Daily queries", options=[500, 2000, 5000, 15000, 30000, 60000], value=15000,
            format_func=lambda v: f"{v:,} queries/day", key="w_vol"
        )
        wiz_privacy = st.radio(
            "Privacy",
            ["Strict Local Requirement (FERPA / GDPR / Private on-prem)",
             "Cloud API Permissible (Public university documents only)"],
            key="w_privacy"
        )
        wiz_hours = st.radio(
            "Schedule",
            ["Business Hours Duty-Cycled (7 AM – 10 PM, sleep off-hours)",
             "Continuous Always-On (24/7 unmonitored availability)"],
            key="w_hours"
        )

    with w_col2:
        st.markdown("##### 2. Recommendation")
        _jmap = {r.profile: r.j_per_tok for r in ms.itertuples()} if not ms.empty else None
        rec = S.ai_green_auditor_recommendation(
            workload_type=wiz_case, query_volume_daily=wiz_vol, data_privacy_level=wiz_privacy,
            operating_mode=wiz_hours, profiles=profiles, cloud_options=cloud,
            j_per_tok=_jmap, idle_tax=tax_avg,
        )
        st.markdown(f"""
<div class="rec-card" style="background:#FFFFFF; border:1.5px solid #FDBA74; border-top:4px solid #F97316; border-radius:12px; padding:18px 20px;">
  <div class="rec-kicker" style="font-size:0.78rem; color:#EA580C; font-weight:700; text-transform:uppercase;">Recommendation</div>
  <h3 style="margin:4px 0 8px 0; color:#0F172A;">🎯 {rec['recommended_architecture']}</h3>
  <p style="color:#334155; font-size:0.92rem; line-height:1.5;">{md_inline(rec['primary_rationale'])}</p>
  <div class="rec-note" style="background:#FFF7ED; border-left:3px solid #F97316; padding:10px 14px; border-radius:6px; margin:10px 0; font-size:0.88rem; color:#7C2D12;">
    <strong>⏰ Schedule:</strong> {md_inline(rec['duty_cycle_advice'])}
  </div>
  <div class="rec-meta" style="display:flex; flex-wrap:wrap; gap:14px; font-size:0.85rem; color:#64748B; margin-top:8px;">
    <div>🚀 <strong>Fastest:</strong> {rec['fastest_alternative']}</div>
    <div>🌱 <strong>Lowest J/token:</strong> {rec['lowest_power_alternative']}</div>
  </div>
</div>""", unsafe_allow_html=True)


# ================================================================ 3. ⚖️ Pareto Frontier
with tab_pareto:
    st.subheader("⚖️ Quality vs Energy")
    st.caption("Top-left is best: high quality, low energy per token. Orange = no other model beats it on both.")

    if not ms.empty and ms["quality"].notna().any():
        q_src = st.radio("Quality score", ["Automatic check", "Human review"], horizontal=True, key="pareto_q")
        qcol = "quality" if q_src == "Automatic check" else "manual"
        pdata = [{"model": r.model, "quality": getattr(r, qcol), "j_per_tok": r.j_per_tok,
                  "tok_s": r.tok_s, "params_b": r.params_b}
                 for r in ms.itertuples() if getattr(r, qcol) is not None and pd.notna(getattr(r, qcol))]
        pdf = pd.DataFrame(S.compute_pareto_frontier(pdata))
        pdf["eei"] = [S.compute_efficiency_index(q, j) for q, j in zip(pdf["quality"], pdf["j_per_tok"])]
        pdf["Frontier"] = pdf["pareto_optimal"].map({True: "On frontier", False: "Dominated"})

        p_c1, p_c2 = st.columns([1.6, 1])
        with p_c1:
            fig_p = px.scatter(
                pdf, x="j_per_tok", y="quality", text="model", color="Frontier",
                color_discrete_map={"On frontier": ORANGE_PRIMARY, "Dominated": "#94A3B8"},
                size=pdf["params_b"].fillna(1).astype(float), size_max=28,
                labels={"j_per_tok": "Active energy (J/token) — lower is greener",
                        "quality": f"{q_src} (0–100)"},
                hover_data={"tok_s": ":.1f", "eei": ":.1f", "params_b": True},
                title="Bubble size = parameters (B)",
            )
            front = pdf[pdf["pareto_optimal"]].sort_values("j_per_tok")
            fig_p.add_trace(go.Scatter(x=front["j_per_tok"], y=front["quality"], mode="lines",
                                       line=dict(color=ORANGE_PRIMARY, dash="dot"), showlegend=False,
                                       hoverinfo="skip"))
            fig_p.update_traces(textposition="top center", selector=dict(mode="markers+text"))
            show(fig_p, 400)
        with p_c2:
            st.markdown("##### 🏆 Efficiency ranking")
            st.caption("EEI = quality ÷ J/token.")
            st.dataframe(pdf.sort_values("eei", ascending=False)[["model", "quality", "j_per_tok", "eei", "Frontier"]]
                         .rename(columns={"model": "Model", "quality": "Quality", "j_per_tok": "J/token", "eei": "EEI"})
                         .round(2), hide_index=True, **STRETCH)
    else:
        st.info("Needs benchmark data and quality scores.")


# ================================================================ 4. 🏫 Campus Chatbot
with tab_campus:
    st.subheader("🏫 Campus Chatbot: One Academic Year")
    st.caption(f"{CW.users:,} students · {CW.active_frac:.0%} active daily · {CW.queries_per_active_day} questions each · "
               f"{CW.in_tokens:,} in / {CW.out_tokens} out tokens · ${CG.price_per_kwh:.2f}/kWh · {CG.carbon_kg_per_kwh} kg CO₂/kWh.")

    c_s1, c_s2 = st.columns([1.3, 1])
    with c_s1:
        duty_hours = st.slider("Service hours per day", min_value=4, max_value=24, value=CW.hours_per_day, step=1,
                               help="Servers sleep outside these hours.")
    with c_s2:
        open_all_year = st.checkbox("Open all year (365 days, includes breaks)", value=False,
                                    help=f"Default is {CW.days} teaching days. More days also means more questions.")

    sim_w = replace(CW, hours_per_day=duty_hours, days=365 if open_all_year else CW.days)
    s_biz, _ = estimate(sim_w, CG, 0.7, False, 1.0, 0.0, with_cloud=False)
    s_always, _ = estimate(sim_w, CG, 0.7, True, 1.0, 0.0, with_cloud=False)

    if not s_biz.empty and not s_always.empty:
        m_sim = s_biz[["option", "kwh", "cost_usd", "kg_co2"]].rename(
            columns={"kwh": "kwh_biz", "cost_usd": "cost_biz", "kg_co2": "co2_biz"}).merge(
            s_always[["option", "kwh", "cost_usd", "kg_co2"]].rename(
                columns={"kwh": "kwh_always", "cost_usd": "cost_always", "kg_co2": "co2_always"}), on="option")
        m_sim["kwh_saved"] = m_sim["kwh_always"] - m_sim["kwh_biz"]
        m_sim["cost_saved"] = m_sim["cost_always"] - m_sim["cost_biz"]
        m_sim["multiplier"] = m_sim["kwh_always"] / m_sim["kwh_biz"].where(m_sim["kwh_biz"] > 0)
        m_sim["option"] = m_sim["option"].map(model_key)

        pick = st.selectbox("Model", m_sim["option"].tolist(),
                            index=m_sim["option"].tolist().index(small["model"]) if small is not None and small["model"] in m_sim["option"].tolist() else 0,
                            key="campus_model")
        r = m_sim[m_sim["option"] == pick].iloc[0]
        eq = S.compute_equivalents(r["kwh_saved"], r["co2_always"] - r["co2_biz"])
        b_eq = S.compute_budget_equivalents(r["cost_saved"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Scheduled", f"{r['kwh_biz']:,.0f} kWh/yr")
        k2.metric("Always-on 24/7", f"{r['kwh_always']:,.0f} kWh/yr", f"{r['multiplier']:.2f}×", delta_color="inverse")
        k3.metric("Saved by scheduling", f"${r['cost_saved']:,.0f}/yr")
        k4.metric("CO₂ avoided", f"{r['co2_always'] - r['co2_biz']:,.0f} kg/yr")

        if r["kwh_saved"] > 0:
            st.markdown(f"""
<div class="savings-box" style="background:#FFF7ED; border:1.5px solid #FDBA74; border-radius:10px; padding:14px 18px; margin:10px 0;">
  <h4 style="margin:0 0 4px 0; color:#EA580C;">💡 What scheduling {pick} saves each year</h4>
  <p style="margin:0; color:#7C2D12; font-size:0.92rem;">
    <strong>{r['kwh_saved']:,.0f} kWh</strong> ≈ <strong>{eq['smartphones_charged']:,}</strong> phone charges ·
    <strong>{eq['gas_car_miles']:,.0f}</strong> car miles of CO₂ ·
    <strong>${r['cost_saved']:,.0f}</strong> ≈ <strong>{b_eq['cafeteria_meals']:,}</strong> campus meals.
  </p>
</div>""", unsafe_allow_html=True)
        else:
            st.info("At 24 service hours the schedule is already always-on, so there is nothing to save.")

        long = pd.concat([
            m_sim[["option", "kwh_biz"]].rename(columns={"kwh_biz": "kwh"}).assign(Mode=f"Scheduled ({duty_hours}h)"),
            m_sim[["option", "kwh_always"]].rename(columns={"kwh_always": "kwh"}).assign(Mode="Always-on (24/7)"),
        ], ignore_index=True)
        hbar(long, "option", "kwh", "kWh per year", "Energy per year: scheduled vs always-on", color="Mode")

        with st.expander("📋 Full table", expanded=False):
            st.dataframe(pretty_table(m_sim, {
                "option": "Model", "kwh_biz": f"Scheduled kWh/yr", "kwh_always": "24/7 kWh/yr",
                "kwh_saved": "kWh saved", "cost_saved": "Electricity saved ($/yr)", "multiplier": "24/7 ÷ scheduled (×)",
            }, 2), hide_index=True, **STRETCH)


# ================================================================ 5. 🧮 Calculator
with tab_calc:
    st.subheader("🧮 Deployment Calculator")
    st.caption("Set your own scenario in the sidebar. Costs include hardware spread over 4 years.")

    res_c, problems = estimate(W, G, tu, aon, bg, capex, with_cloud=True)
    if CLOUD_PLACEHOLDER and use_cloud:
        st.warning("Cloud rows use **placeholder** prices and energy figures. Replace them in "
                   "`data/cloud_options.json` with real quotes before drawing conclusions.")
    for msg in problems:
        st.caption(f"Skipped: {msg}")

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Questions / year", f"{W.total_queries:,.0f}")
    w2.metric("Tokens / year", f"{W.total_queries * (tin + tout) / 1e6:,.1f} M")
    w3.metric("Average load", f"{W.avg_qps:,.2f} q/s")
    w4.metric("Peak load", f"{W.peak_qps:,.2f} q/s", help="Used to size the server fleet.")

    if not res_c.empty:
        c_metric = st.radio("Compare by", ["Energy (kWh/year)", "Cost (USD/year)", "Carbon (kg CO₂/year)", "Energy per query (Wh)"],
                            horizontal=True, key="calc_m")
        c_key = {"Energy (kWh/year)": "kwh", "Cost (USD/year)": "cost_usd",
                 "Carbon (kg CO₂/year)": "kg_co2", "Energy per query (Wh)": "wh_per_query"}[c_metric]
        plot_df = res_c.assign(option=res_c["option"].astype(str).map(model_key))
        hbar(plot_df, "option", c_key, c_metric, c_metric, color="kind_label")
        with st.expander("📋 Full table", expanded=False):
            st.dataframe(pretty_table(res_c.sort_values(c_key), ESTIMATE_COLUMNS, 2), hide_index=True, **STRETCH)


# ================================================================ 6. 📊 Telemetry
with tab_telemetry:
    st.subheader("📊 Benchmark Telemetry")
    st.caption("One dot per run. Speed and runtime are timed; energy follows the power basis below.")

    if POWER_KIND == "estimate":
        st.warning(
            "**Power is estimated, not metered.** This machine exposed no RAPL or NVML energy counters, so watts come "
            "from a CPU-utilisation model. For metered results, re-run the benchmark on Linux (RAPL), on an NVIDIA GPU "
            "(NVML), or log a wall-plug power meter alongside the run."
        )
    elif POWER_KIND == "measured":
        st.success(POWER_NOTE)

    if not ms.empty:
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Models", f"{len(ms)}")
        t2.metric("Runs", f"{len(bench):,}")
        t3.metric("Prompts", f"{bench['prompt_id'].nunique()}" if "prompt_id" in bench.columns else "—")
        t4.metric("Median runtime", f"{pd.to_numeric(bench['wall_s'], errors='coerce').median():.1f} s"
                  if "wall_s" in bench.columns else "—")

        runs_df = bench.copy()
        for c in ["gen_tok_s", "j_per_out_token", "avg_w", "wall_s", "out_tokens"]:
            if c in runs_df.columns:
                runs_df[c] = pd.to_numeric(runs_df[c], errors="coerce")
        if has_cols(runs_df, ["gen_tok_s", "j_per_out_token"]):
            fig_r = px.scatter(runs_df.dropna(subset=["gen_tok_s", "j_per_out_token"]),
                               x="gen_tok_s", y="j_per_out_token", color="model",
                               hover_data=[c for c in ["prompt_id", "out_tokens", "wall_s", "avg_w"] if c in runs_df.columns],
                               labels={"gen_tok_s": "Speed (tokens/s)", "j_per_out_token": "Active energy (J/token)"},
                               title="Every run: faster models spend less energy per token")
            fig_r.update_traces(marker=dict(size=10, opacity=0.8))
            show(fig_r, 380)

        t_c1, t_c2 = st.columns(2)
        with t_c1:
            hbar(ms, "model", "tok_s", "Tokens / second", "Speed (median)")
        with t_c2:
            hbar(ms, "model", "j_per_tok", "J / output token", "Active energy (median)")

        with st.expander("📋 Summary table", expanded=False):
            st.dataframe(pretty_table(ms, {
                "model": "Model", "params_b": "Params (B)", "runs": "Runs", "tok_s": "tok/s (median)",
                "wall_s": "Runtime s (median)", "j_per_tok": "J/token (median)", "active_w": "Avg W", "idle_w": "Idle W",
            }, 2), hide_index=True, **STRETCH)


# ================================================================ 7. 🎯 Quality
with tab_quality:
    st.subheader("🎯 Answer Quality")
    st.caption("Saving energy only counts if the answers are still good.")

    man_df = read_csv_safe(QUALITY_MANUAL)
    at_df = read_csv_safe(QUALITY_AUTO)
    if at_df.empty and has_cols(man_df, ["model", "prompt_id", "automatic_score"]):
        at_df = man_df[["model", "prompt_id", "automatic_score"]].dropna().rename(columns={"automatic_score": "score"})

    q_c1, q_c2 = st.columns(2)
    with q_c1:
        if has_cols(at_df, ["model", "score"]):
            at_df["score"] = pd.to_numeric(at_df["score"], errors="coerce")
            hbar(at_df.groupby("model", as_index=False)["score"].mean(), "model", "score", "Score (0–100)", "Automatic check")
    with q_c2:
        if has_cols(man_df, ["model", "manual_score"]):
            man_df["manual_score"] = pd.to_numeric(man_df["manual_score"], errors="coerce")
            hbar(man_df.groupby("model", as_index=False)["manual_score"].mean(), "model", "manual_score",
                 "Score (0–100)", "Human review")

    if has_cols(man_df, ["model", "prompt_id", "manual_score"]):
        piv = man_df.pivot_table(index="model", columns="prompt_id", values="manual_score", aggfunc="mean")
        if not piv.empty:
            fig_hm = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="Oranges",
                               zmin=0, zmax=100, title="Human score by prompt")
            show(fig_hm, 260)


# ================================================================ 8. ⚔️ Model Arena
with tab_arena:
    render_model_arena()


# ================================================================ 9. 📄 Report
with tab_report:
    st.subheader("📄 Sustainability Report")
    st.caption("Built from the same numbers as the tabs above. Download and share.")

    top_m = [{"model": r.model, "params_b": r.params_b, "tok_s": r.tok_s, "j_per_tok": r.j_per_tok,
              "watts": r.active_w, "idle_w": r.idle_w} for r in ms.itertuples()] if not ms.empty else []

    campus = {"days": CW.days, "hours": CW.hours_per_day, "qpd": CW.queries_per_active_day}
    if small is not None:
        extra_kwh = small["always_kwh"] - small["duty_kwh"]
        extra_co2 = small["always_co2"] - small["duty_co2"]
        eq_r = S.compute_equivalents(extra_kwh, extra_co2)
        campus.update({
            "model": small["model"],
            "biz_kwh": f"{small['duty_kwh']:,.0f}", "always_kwh": f"{small['always_kwh']:,.0f}",
            "multiplier": f"{small['idle_tax']:.2f}×",
            "smartphones": f"{eq_r['smartphones_charged']:,}", "car_miles": f"{eq_r['gas_car_miles']:,.0f}",
            "trees": f"{eq_r['tree_seedlings_10yr']:,.0f}",
        })

    recs = None
    if small is not None:
        recs = [f"Use **{small['model']}** — best quality per joule in our tests.",
                f"Sleep servers outside the {CW.hours_per_day} service hours and during breaks.",
                "Verify cloud prices (currently placeholders) before choosing cloud." if CLOUD_PLACEHOLDER
                else "Compare cloud quotes against the local figures.",
                "Confirm power with RAPL/NVML or a wall-plug meter before budgeting." if POWER_KIND == "estimate"
                else "Re-run benchmarks after any hardware change."]

    gen_rep = S.generate_executive_markdown_report(
        workload_summary={"total_queries": W.total_queries,
                          "total_tokens_m": W.total_queries * (tin + tout) / 1e6, "users": users},
        top_models=top_m, campus_scenario=campus, power_method_note=POWER_NOTE,
        findings=FINDINGS, recommendations=recs,
    )

    st.download_button("📥 Download report (.md)", data=gen_rep.encode("utf-8"),
                       file_name="AI_Energy_Lab_Report.md", mime="text/markdown", **STRETCH)
    with st.expander("👁️ Preview", expanded=True):
        st.markdown(gen_rep)
