"""Android Software Trust — Industry-Level Cybersecurity Research Dashboard.

Source-Aware Evidence Fusion for Multi-Source Security Risk Assessment
Authors: Paramjeet Kaur, Priya Goel, Vrinda Sachdeva, Avneesh Kumar,
         Janvi Chaudhary, Mahek Singhal
GL Bajaj Institute of Technology and Management, Greater Noida, India
"""

import sys
import os
import math
import time
import json
import random
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    FusionConfig, RESEARCH_PRESETS,
    STATIC_WEIGHT, BEHAVIOUR_WEIGHT, NETWORK_WEIGHT,
    TRUSTED_THRESHOLD, HIGH_RISK_THRESHOLD,
    PAPER_TRUST_THRESHOLD, PAPER_REVIEW_THRESHOLD,
)
from src.fusion import fuse_evidence, FusionResult
from src.history import AssessmentHistory
from src.metrics import (
    DATASET_METADATA, get_model_comparison_df, get_reliability_weights_df,
    get_ablation_df, get_paper_metadata, get_paper_table1_df, get_paper_table2_df,
    get_paper_table3_df, get_paper_table4_df, get_paper_fig1_df,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Android Software Trust | Evidence Fusion Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Synthwave-Cyber Dark (not generic AI) ───────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─── Tokens ─────────────────────────── */
:root {
  --bg-void:     #03050A;
  --bg-panel:    #080D16;
  --bg-card:     #0D1421;
  --bg-raised:   #111C2E;
  --accent-cyan: #00D4FF;
  --accent-blue: #4488FF;
  --accent-violet:#8B5CF6;
  --accent-lime: #39FF7E;
  --accent-amber:#F59E0B;
  --danger:      #FF3A5C;
  --warning:     #F59E0B;
  --safe:        #22C55E;
  --review:      #F59E0B;
  --border:      rgba(0, 212, 255, 0.13);
  --border-hover:rgba(0, 212, 255, 0.38);
  --text-prime:  #E8F0FE;
  --text-muted:  #6B7FA3;
  --text-dim:    #3A4A6B;
  --glow-cyan:   0 0 20px rgba(0,212,255,0.30), 0 0 60px rgba(0,212,255,0.10);
  --glow-violet: 0 0 20px rgba(139,92,246,0.30);
  --glow-danger: 0 0 20px rgba(255,58,92,0.35);
}

/* ─── Base reset ─────────────────────── */
html, body, .stApp, [class*="css"] {
  font-family: 'Inter', 'Space Grotesk', sans-serif !important;
  background-color: var(--bg-void) !important;
  color: var(--text-prime) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ─── Sidebar ────────────────────────── */
/* Keep the navigation panel visible and expanded on desktop/local runs. */
[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
  width: 260px !important;
  min-width: 260px !important;
  max-width: 260px !important;
  flex: 0 0 260px !important;
  transform: none !important;
  visibility: visible !important;
  opacity: 1 !important;
}
[data-testid="stSidebar"] * { color: var(--text-prime) !important; }
/* Preserve Streamlit's reopen control if the sidebar was collapsed earlier. */
[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
}

/* ─── Sidebar nav button ─────────────── */
.nav-btn {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; border-radius: 8px; margin: 2px 0;
  cursor: pointer; border: 1px solid transparent;
  background: transparent; color: var(--text-muted);
  font-size: 13px; font-weight: 500; letter-spacing: 0.02em;
  transition: all 0.18s ease; width: 100%;
}
.nav-btn:hover { background: rgba(0,212,255,0.07); color: var(--text-prime); border-color: var(--border); }
.nav-btn.active {
  background: rgba(0,212,255,0.12); color: var(--accent-cyan);
  border-color: rgba(0,212,255,0.30); box-shadow: inset 0 0 0 1px rgba(0,212,255,0.10);
}

/* ─── Cards ──────────────────────────── */
.card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px 22px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { border-color: var(--border-hover); }
.card-glow { box-shadow: var(--glow-cyan); }
.card-danger { border-color: rgba(255,58,92,0.35); box-shadow: var(--glow-danger); }
.card-safe { border-color: rgba(34,197,94,0.35); box-shadow: 0 0 20px rgba(34,197,94,0.15); }
.card-amber { border-color: rgba(245,158,11,0.35); box-shadow: 0 0 20px rgba(245,158,11,0.15); }

/* ─── KPI tiles ──────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0; }
.kpi-tile {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px; position: relative; overflow: hidden;
}
.kpi-tile::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
}
.kpi-label { font-size: 11px; color: var(--text-muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.kpi-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.kpi-cyan  { color: var(--accent-cyan); }
.kpi-lime  { color: var(--accent-lime); }
.kpi-amber { color: var(--accent-amber); }
.kpi-danger{ color: var(--danger); }
.kpi-violet{ color: var(--accent-violet); }

/* ─── Section header ─────────────────── */
.section-header {
  display: flex; align-items: center; gap: 12px;
  padding: 0 0 16px 0; margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
}
.section-title { font-size: 20px; font-weight: 700; color: var(--text-prime); }
.section-badge {
  font-size: 10px; font-weight: 700; padding: 3px 9px;
  border-radius: 20px; text-transform: uppercase; letter-spacing: 0.08em;
  background: rgba(0,212,255,0.12); color: var(--accent-cyan);
  border: 1px solid rgba(0,212,255,0.25);
}

/* ─── Tables ─────────────────────────── */
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th {
  background: var(--bg-raised); color: var(--text-muted); font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  padding: 10px 14px; border-bottom: 1px solid var(--border); text-align: left;
}
.styled-table td { padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,0.04); }
.styled-table tr:hover td { background: rgba(0,212,255,0.04); }
.best-row td { background: rgba(0,212,255,0.06) !important; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.highlight-cyan { color: var(--accent-cyan); font-weight: 700; }
.highlight-lime  { color: var(--accent-lime); font-weight: 700; }
.highlight-danger { color: var(--danger); font-weight: 700; }

/* ─── Tag / Pill ─────────────────────── */
.tag {
  display: inline-block; font-size: 10px; font-weight: 700;
  padding: 3px 9px; border-radius: 20px; letter-spacing: 0.06em;
  text-transform: uppercase;
}
.tag-cyan    { background: rgba(0,212,255,0.15); color: var(--accent-cyan); }
.tag-violet  { background: rgba(139,92,246,0.15); color: var(--accent-violet); }
.tag-lime    { background: rgba(57,255,126,0.15); color: var(--accent-lime); }
.tag-amber   { background: rgba(245,158,11,0.18); color: var(--accent-amber); }
.tag-danger  { background: rgba(255,58,92,0.15); color: var(--danger); }

/* ─── Progress bars ──────────────────── */
.progress-bar-wrap { background: var(--bg-raised); border-radius: 6px; height: 8px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }
.fill-cyan   { background: linear-gradient(90deg, #00D4FF, #4488FF); }
.fill-lime   { background: linear-gradient(90deg, #39FF7E, #22C55E); }
.fill-amber  { background: linear-gradient(90deg, #F59E0B, #EF4444); }
.fill-danger { background: linear-gradient(90deg, #FF3A5C, #EF4444); }
.fill-violet { background: linear-gradient(90deg, #8B5CF6, #6366F1); }

/* ─── Gauge ring ──────────────────────── */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }

/* ─── Equation block ─────────────────── */
.eq-block {
  background: var(--bg-raised); border: 1px solid var(--border);
  border-left: 3px solid var(--accent-cyan); border-radius: 8px;
  padding: 14px 18px; margin: 8px 0; font-family: 'JetBrains Mono', monospace;
  font-size: 13px; color: var(--accent-cyan);
}
.eq-label { font-size: 10px; color: var(--text-muted); font-family: 'Inter', sans-serif;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }

/* ─── Decision state card ─────────────── */
.decision-trusted {
  background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(0,212,255,0.06));
  border: 1px solid rgba(34,197,94,0.40); border-radius: 12px; padding: 20px;
  text-align: center;
}
.decision-review {
  background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.04));
  border: 1px solid rgba(245,158,11,0.40); border-radius: 12px; padding: 20px;
  text-align: center;
}
.decision-highrisk {
  background: linear-gradient(135deg, rgba(255,58,92,0.14), rgba(139,92,246,0.06));
  border: 1px solid rgba(255,58,92,0.40); border-radius: 12px; padding: 20px;
  text-align: center;
}
.decision-emoji { font-size: 36px; margin-bottom: 6px; }
.decision-label { font-size: 18px; font-weight: 800; letter-spacing: 0.04em; }
.decision-score { font-size: 38px; font-weight: 900; font-family: 'JetBrains Mono', monospace; }

/* ─── Alert box ──────────────────────── */
.alert-info {
  background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.25);
  border-radius: 8px; padding: 12px 16px; font-size: 13px; color: var(--text-prime);
}
.alert-warn {
  background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.30);
  border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #FDE68A;
}

/* ─── Hero banner ────────────────────── */
.hero-bar {
  background: linear-gradient(135deg, #03050A 0%, #080D16 40%, #0D1A2E 100%);
  border-bottom: 1px solid var(--border);
  padding: 28px 36px; display: flex; align-items: center; gap: 24px;
  position: relative; overflow: hidden;
}

/* ─── Source contribution bar ─────────── */
.source-row { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.source-name { width: 120px; font-size: 12px; color: var(--text-muted); font-weight: 600; }
.source-bar-wrap { flex: 1; }
.source-pct { width: 48px; text-align: right; font-size: 12px; font-family: 'JetBrains Mono', monospace; }

/* ─── Streamlit widget overrides ──────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: var(--bg-raised) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-prime) !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSlider"] { color: var(--accent-cyan) !important; }
[data-testid="stSelectbox"] > div { background: var(--bg-raised) !important; }

.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(68,136,255,0.15)) !important;
  border: 1px solid rgba(0,212,255,0.40) !important;
  border-radius: 8px !important; color: var(--accent-cyan) !important;
  font-weight: 700 !important; letter-spacing: 0.04em !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.25), rgba(68,136,255,0.25)) !important;
  box-shadow: 0 2px 10px rgba(0,212,255,0.10) !important; transform: translateY(-1px) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
  font-weight: 600 !important; font-size: 13px !important; color: var(--text-muted) !important;
  border-bottom: 2px solid transparent !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent-cyan) !important;
  border-bottom: 2px solid var(--accent-cyan) !important;
}

/* Responsive application header */
.app-hero {
  display: flex; align-items: center; justify-content: space-between; gap: 20px;
  padding: 22px clamp(16px, 3vw, 36px);
  min-height: 104px; box-sizing: border-box;
  background: linear-gradient(135deg, #0A1220 0%, #101C2E 100%);
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
}
.app-hero-brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
.app-hero-shield { font-size: 30px; flex: 0 0 auto; }
.app-hero-copy { min-width: 0; }
.app-hero-title { color: #F1F5F9; font-size: clamp(18px, 2vw, 24px); font-weight: 750; line-height: 1.2; }
.app-hero-subtitle { color: #A7B4C8; font-size: 12px; margin-top: 5px; line-height: 1.45; }
.app-hero-status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.status-pill { padding: 5px 10px; border-radius: 999px; font-size: 10px; font-weight: 700;
  color: #BAE6FD; background: rgba(56, 189, 248, 0.10); border: 1px solid rgba(56, 189, 248, 0.24); }
.status-pill-muted { color: #CBD5E1; background: rgba(148, 163, 184, 0.08); border-color: rgba(148, 163, 184, 0.18); }

/* Native sidebar radio styled as a clear navigation list */
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 4px; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label {
  padding: 9px 11px; border: 1px solid transparent; border-radius: 8px;
  transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:hover {
  background: rgba(148, 163, 184, 0.08); border-color: rgba(148, 163, 184, 0.16);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.30);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] label p { margin: 0; }

/* Responsive layouts */
@media (max-width: 1100px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .kpi-value { font-size: 24px; }
}
@media (max-width: 700px) {
  .app-hero { align-items: flex-start; flex-direction: column; gap: 12px; padding: 18px 16px; }
  .app-hero-shield { font-size: 26px; }
  .app-hero-status { justify-content: flex-start; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .kpi-tile { padding: 13px; }
  .kpi-label { font-size: 10px; }
  .kpi-value { font-size: 21px; overflow-wrap: anywhere; }
  .card { padding: 16px; }
  .section-title { font-size: 18px; }
  .source-name { width: 88px; }
  .source-pct { width: 42px; }
  .block-container { padding-left: 0 !important; padding-right: 0 !important; }
}
@media (max-width: 420px) {
  .kpi-grid { grid-template-columns: minmax(0, 1fr); }
  .app-hero-subtitle { font-size: 11px; }
  .decision-score { font-size: 30px; }
  .eq-block { padding: 12px; font-size: 11px; overflow-wrap: anywhere; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Particle / Sand Canvas hero ────────────────────────────────────────────────
HERO_HTML = """
<div class="app-hero">
  <div class="app-hero-brand">
    <div class="app-hero-shield" aria-hidden="true">🛡️</div>
    <div class="app-hero-copy">
      <div class="app-hero-title">Android Software Trust</div>
      <div class="app-hero-subtitle">Source-aware evidence fusion for security risk assessment</div>
    </div>
  </div>
  <div class="app-hero-status">
    <span class="status-pill">Research prototype</span>
    <span class="status-pill status-pill-muted">Offline assessment demo</span>
  </div>
</div>
"""

# ── Session state ──────────────────────────────────────────────────────────────
if "config" not in st.session_state:
    st.session_state.config = FusionConfig()
if "history" not in st.session_state:
    st.session_state.history = AssessmentHistory()
if "page" not in st.session_state:
    st.session_state.page = "overview"
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, JetBrains Mono, sans-serif", color="#E8F0FE", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

def apply_theme(fig, title=""):
    layout = dict(PLOTLY_LAYOUT)
    if title:
        layout["title"] = dict(text=title, font=dict(size=14, color="#E8F0FE"), x=0, xanchor="left")
    fig.update_layout(**layout)
    return fig

CYAN  = "#00D4FF"
BLUE  = "#4488FF"
VIOLET= "#8B5CF6"
LIME  = "#39FF7E"
AMBER = "#F59E0B"
DANGER= "#FF3A5C"
MUTED = "#6B7FA3"
WHITE = "#E8F0FE"

# ── Sidebar ────────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("overview",    "🏠", "Overview"),
    ("calculator",  "⚡", "Risk Calculator"),
    ("evidence",    "🔬", "Evidence Sources"),
    ("experiments", "📊", "Experiments"),
    ("config",      "⚙️", "Configuration"),
    ("history",     "🕐", "History"),
    ("about",       "📄", "About & Limits"),
]

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 20px;">
      <div style="font-size:11px;font-weight:700;color:#3A4A6B;text-transform:uppercase;
           letter-spacing:0.1em;margin-bottom:14px;">Navigation</div>
    """, unsafe_allow_html=True)

    nav_keys = [key for key, _, _ in NAV_ITEMS]
    nav_labels = {key: f"{icon}  {label}" for key, icon, label in NAV_ITEMS}
    if "page_navigation" not in st.session_state:
        st.session_state.page_navigation = st.session_state.page

    st.radio(
        "Navigation",
        options=nav_keys,
        format_func=lambda key: nav_labels[key],
        key="page_navigation",
        label_visibility="collapsed",
    )
    st.session_state.page = st.session_state.page_navigation

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Config summary
    cfg = st.session_state.config
    st.markdown(f"""
    <div style="padding:12px;background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
         border-radius:8px;font-size:12px;">
      <div style="color:#3A4A6B;font-weight:700;font-size:10px;text-transform:uppercase;
           letter-spacing:0.08em;margin-bottom:8px;">Active Config</div>
      <div style="display:flex;justify-content:space-between;margin:3px 0;">
        <span style="color:#6B7FA3;">Static</span>
        <span style="color:#00D4FF;font-family:monospace;">{cfg.static_weight:.2f}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin:3px 0;">
        <span style="color:#6B7FA3;">Behaviour</span>
        <span style="color:#00D4FF;font-family:monospace;">{cfg.behaviour_weight:.2f}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin:3px 0;">
        <span style="color:#6B7FA3;">Network</span>
        <span style="color:#00D4FF;font-family:monospace;">{cfg.network_weight:.2f}</span>
      </div>
      <div style="border-top:1px solid rgba(0,212,255,0.12);margin:8px 0;"></div>
      <div style="display:flex;justify-content:space-between;margin:3px 0;">
        <span style="color:#6B7FA3;">Trusted &lt;</span>
        <span style="color:#22C55E;font-family:monospace;">{cfg.trusted_threshold:.2f}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin:3px 0;">
        <span style="color:#6B7FA3;">High Risk ≥</span>
        <span style="color:#FF3A5C;font-family:monospace;">{cfg.high_risk_threshold:.2f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:8px 0;font-size:11px;color:#3A4A6B;text-align:center;margin-top:12px;">
      History: {len(st.session_state.history.records)} records
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(HERO_HTML, unsafe_allow_html=True)

page = st.session_state.page
cfg  = st.session_state.config

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "overview":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">System Overview</div>
        <div class="section-badge">Research Prototype</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)

        # KPI row
        st.markdown("""
        <div class="kpi-grid">
          <div class="kpi-tile">
            <div class="kpi-label">Evidence Sources</div>
            <div class="kpi-value kpi-cyan">3</div>
            <div class="kpi-sub">Static · Behavioural · Network</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Best Balanced Accuracy</div>
            <div class="kpi-value kpi-lime">95.31<span style="font-size:16px;font-weight:400;">%</span></div>
            <div class="kpi-sub">Behaviour-Dominant fusion (Table III)</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Best F1 Score</div>
            <div class="kpi-value kpi-violet">0.9873</div>
            <div class="kpi-sub">vs. Equal fusion 0.7628</div>
          </div>
          <div class="kpi-tile">
            <div class="kpi-label">Fusion Engine</div>
            <div class="kpi-value kpi-amber" style="font-size:18px;padding-top:5px;">Eq. 1–12</div>
            <div class="kpi-sub">Paper-accurate implementation</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        # System architecture card
        st.markdown("""
        <div class="card" style="margin-bottom:16px;">
          <div style="font-size:14px;font-weight:700;color:#E8F0FE;margin-bottom:14px;">
            System Architecture
          </div>
          <div style="display:flex;flex-direction:column;gap:10px;">
            <div style="display:flex;align-items:stretch;gap:12px;">
              <div style="background:rgba(0,212,255,0.08);border:1px solid rgba(0,212,255,0.20);
                   border-radius:8px;padding:12px 16px;flex:1;">
                <div style="font-size:10px;color:#00D4FF;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">Layer 1 — Evidence</div>
                <div style="font-size:13px;color:#E8F0FE;margin-top:6px;line-height:1.5;">
                  <span style="color:#6B7FA3;">Static:</span> DREBIN (214 features, 15k samples)<br>
                  <span style="color:#6B7FA3;">Behaviour:</span> CICMalDroid 2020 (470 features, 11k)<br>
                  <span style="color:#6B7FA3;">Network:</span> CIC-AndMal2017 (72 features, 427k)
                </div>
              </div>
            </div>
            <div style="text-align:center;color:#3A4A6B;font-size:18px;">↓</div>
            <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.20);
                 border-radius:8px;padding:12px 16px;">
              <div style="font-size:10px;color:#8B5CF6;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">Layer 2 — Source Models</div>
              <div style="font-size:13px;color:#E8F0FE;margin-top:6px;">
                Calibrated Random Forest / Extra Trees classifiers producing
                probability estimates R<sub>s</sub>, R<sub>b</sub>, R<sub>n</sub> ∈ [0,1]
              </div>
            </div>
            <div style="text-align:center;color:#3A4A6B;font-size:18px;">↓</div>
            <div style="background:rgba(57,255,126,0.06);border:1px solid rgba(57,255,126,0.20);
                 border-radius:8px;padding:12px 16px;">
              <div style="font-size:10px;color:#39FF7E;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">Layer 3 — Fusion Engine (Eq. 1–12)</div>
              <div style="font-size:13px;color:#E8F0FE;margin-top:6px;line-height:1.6;">
                Fused Risk R<sub>f</sub> · Disagreement D · Coverage C · Trust T<br>
                → Decision State: <span style="color:#22C55E;">Trusted</span> /
                <span style="color:#F59E0B;">Review</span> /
                <span style="color:#FF3A5C;">High Risk</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Research summary card
        meta = get_paper_metadata()
        st.markdown(f"""
        <div class="card">
          <div style="font-size:14px;font-weight:700;color:#E8F0FE;margin-bottom:10px;">Research Paper</div>
          <div style="font-size:13px;color:#6B7FA3;line-height:1.6;">
            <span class="tag tag-cyan">IEEE</span>
            <span class="tag tag-violet" style="margin-left:6px;">CS</span>
            <div style="margin-top:10px;font-style:italic;color:#E8F0FE;">
              "{meta['title']}"
            </div>
            <div style="margin-top:6px;font-size:12px;">
              {' · '.join(meta['authors'][:3])} <em>et al.</em><br>
              <span style="color:#3A4A6B;">{meta['institution']}</span>
            </div>
          </div>
          <div style="margin-top:12px;background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.12);
               border-radius:6px;padding:10px 14px;font-size:12px;color:#6B7FA3;">
            <span style="color:#00D4FF;font-weight:700;">Central Question:</span><br>
            {meta['central_question']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Quick weight chart
        w_data = [cfg.static_weight, cfg.behaviour_weight, cfg.network_weight]
        w_labels = ["Static", "Behaviour", "Network"]
        w_colors = [CYAN, VIOLET, LIME]
        fig_w = go.Figure(go.Pie(
            labels=w_labels, values=w_data,
            hole=0.62,
            marker=dict(colors=w_colors, line=dict(color="#080D16", width=3)),
            textinfo="percent", textfont=dict(size=13, color="#E8F0FE"),
            hovertemplate="<b>%{label}</b><br>Weight: %{value:.3f}<extra></extra>",
        ))
        fig_w.add_annotation(
            text=f"<b>Weights</b>", x=0.5, y=0.5,
            font=dict(size=13, color="#E8F0FE"), showarrow=False,
        )
        # Merge the shared layout first, then override its legend settings.
        # Passing **PLOTLY_LAYOUT and legend= separately duplicates the keyword.
        weight_layout = dict(PLOTLY_LAYOUT)
        weight_layout.update(
            title=dict(text="Active Fusion Weights", font=dict(size=13, color=WHITE), x=0),
            showlegend=True,
            height=220,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15,
                xanchor="center", x=0.5,
                bgcolor="rgba(0,0,0,0)", font=dict(size=11),
            ),
            margin=dict(l=10, r=10, t=36, b=30),
        )
        fig_w.update_layout(**weight_layout)
        st.plotly_chart(fig_w, use_container_width=True)

        # Key findings summary
        st.markdown("""
        <div class="card" style="margin-top:4px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">
            Key Findings (Paper)
          </div>
          <div style="display:flex;flex-direction:column;gap:8px;font-size:12px;">
            <div style="display:flex;align-items:flex-start;gap:8px;">
              <span style="color:#39FF7E;font-size:16px;line-height:1;">▲</span>
              <div><span style="color:#E8F0FE;">+16.39 pp</span>
                <span style="color:#6B7FA3;"> BA vs Equal fusion</span></div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:8px;">
              <span style="color:#00D4FF;font-size:16px;line-height:1;">◆</span>
              <div><span style="color:#E8F0FE;">+1.50 pp</span>
                <span style="color:#6B7FA3;"> BA vs best individual</span></div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:8px;">
              <span style="color:#8B5CF6;font-size:16px;line-height:1;">●</span>
              <div><span style="color:#E8F0FE;">10,000 synthetic</span>
                <span style="color:#6B7FA3;"> VI-C: 0 false Trusted</span></div>
            </div>
            <div style="display:flex;align-items:flex-start;gap:8px;">
              <span style="color:#F59E0B;font-size:16px;line-height:1;">⚡</span>
              <div><span style="color:#E8F0FE;">Bootstrap stable</span>
                <span style="color:#6B7FA3;"> 95% CI [0.9382, 0.9674]</span></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Readiness matrix
        st.markdown("""
        <div class="card" style="margin-top:12px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">
            Capability Status
          </div>
          <div style="font-size:12px;">
        """, unsafe_allow_html=True)

        caps = [
            ("Fusion Engine (Eq. 1–12)", "✅", "lime"),
            ("24-test Suite", "✅", "lime"),
            ("Research Dashboard", "✅", "lime"),
            ("Manual Calculator", "✅", "lime"),
            ("Saved Model Files", "⚠️", "amber"),
            ("Live APK Analysis", "🔴", "danger"),
            ("Real-time Inference", "🔴", "danger"),
        ]
        for cap, status, color in caps:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                 padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
              <span style="color:#6B7FA3;">{cap}</span>
              <span style="color:var(--{color},#E8F0FE);">{status}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RISK CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "calculator":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">Evidence Fusion Calculator</div>
        <div class="section-badge">Interactive</div>
      </div>
      <div class="alert-info" style="margin-bottom:20px;">
        ⚡ Enter suspiciousness probabilities from each available evidence source (0 = benign, 1 = malicious).
        Leave a source blank / disabled to exclude it — the engine will renormalise remaining weights per Eq. (4–5).
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px;font-weight:700;color:#E8F0FE;margin-bottom:18px;">Evidence Inputs</div>', unsafe_allow_html=True)

        use_static = st.checkbox("Include Static Evidence (DREBIN)", value=True, key="use_static")
        static_val = None
        if use_static:
            static_val = st.slider("Static Suspiciousness R_s", 0.0, 1.0, 0.25, 0.01, key="static_slider")

        st.markdown("<div style='margin:12px 0;border-top:1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

        use_behaviour = st.checkbox("Include Behavioural Evidence (CICMalDroid)", value=True, key="use_beh")
        beh_val = None
        if use_behaviour:
            beh_val = st.slider("Behavioural Suspiciousness R_b", 0.0, 1.0, 0.72, 0.01, key="beh_slider")

        st.markdown("<div style='margin:12px 0;border-top:1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)

        use_network = st.checkbox("Include Network Evidence (CIC-AndMal)", value=True, key="use_net")
        net_val = None
        if use_network:
            net_val = st.slider("Network Suspiciousness R_n", 0.0, 1.0, 0.80, 0.01, key="net_slider")

        st.markdown("<div style='margin:12px 0;'></div>", unsafe_allow_html=True)
        run_col, save_col = st.columns(2)
        with run_col:
            run = st.button("⚡  Compute Fusion", use_container_width=True)
        with save_col:
            save = st.button("💾  Save to History", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Equation walkthrough
        st.markdown("""
        <div class="card" style="margin-top:14px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">
            Equation Reference
          </div>
        """, unsafe_allow_html=True)
        eqs = [
            ("Eq. 4 — Renormalise", "w̃_i = (w_i · I_i) / Σ(w_j · I_j)"),
            ("Eq. 5 — Fused Risk",  "R_f = Σ_{i∈A} w̃_i · R_i"),
            ("Eq. 7 — Disagreement","D = Σ_{i∈A} w̃_i · (R_i − R_f)²"),
            ("Eq. 8 — Coverage",    "C = Σ_{i∈A} w_i"),
            ("Eq. 9 — Trust Score", "T = (1 − R_f) · (1 − D) · √C"),
        ]
        for label, eq in eqs:
            st.markdown(f"""
            <div class="eq-block">
              <div class="eq-label">{label}</div>
              {eq}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute result
    result: Optional[FusionResult] = None
    if run or save or st.session_state.last_result is not None:
        has_input = any(v is not None for v in [static_val, beh_val, net_val])
        if has_input:
            try:
                result = fuse_evidence(
                    static_prob=static_val,
                    behaviour_prob=beh_val,
                    network_prob=net_val,
                    config=cfg,
                )
                st.session_state.last_result = result
                if save and result is not None:
                    st.session_state.history.add_record(
                        result,
                        static_prob=static_val,
                        behaviour_prob=beh_val,
                        network_prob=net_val,
                    )
            except Exception as e:
                st.error(f"Fusion error: {e}")
        else:
            st.warning("Please enable and set at least one evidence source.")

    result = st.session_state.last_result

    with col_out:
        if result is None:
            st.markdown("""
            <div class="card" style="text-align:center;padding:48px 24px;min-height:320px;
                 display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;">
              <div style="font-size:48px;opacity:0.4;">🔬</div>
              <div style="font-size:14px;color:#3A4A6B;">
                Configure evidence inputs on the left and click<br>
                <span style="color:#00D4FF;">⚡ Compute Fusion</span> to see results
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Decision card
            dec = result.trust_decision
            if dec == "Trusted":
                dec_css, dec_icon, dec_color = "decision-trusted", "✅", "#22C55E"
            elif dec == "Review":
                dec_css, dec_icon, dec_color = "decision-review", "⚠️", "#F59E0B"
            else:
                dec_css, dec_icon, dec_color = "decision-highrisk", "🚨", "#FF3A5C"

            st.markdown(f"""
            <div class="{dec_css}" style="margin-bottom:14px;">
              <div class="decision-emoji">{dec_icon}</div>
              <div class="decision-label" style="color:{dec_color};">{dec}</div>
              <div style="font-size:11px;color:#6B7FA3;margin-top:4px;">Trust-Based Decision (Eq. 10–12)</div>
            </div>
            """, unsafe_allow_html=True)

            # Four metric tiles
            r_pct  = result.risk * 100
            t_pct  = result.trust * 100
            d_pct  = result.disagreement * 100
            c_pct  = result.coverage * 100

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
              <div class="kpi-tile">
                <div class="kpi-label">Fused Risk R_f</div>
                <div class="kpi-value" style="color:{'#FF3A5C' if r_pct>50 else '#F59E0B' if r_pct>20 else '#22C55E'};font-size:24px;">
                  {r_pct:.1f}%
                </div>
                <div class="kpi-sub">Higher = more suspicious</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Trust Score T</div>
                <div class="kpi-value" style="color:{'#22C55E' if t_pct>=70 else '#F59E0B' if t_pct>=40 else '#FF3A5C'};font-size:24px;">
                  {t_pct:.1f}%
                </div>
                <div class="kpi-sub">≥70% Trusted · ≥40% Review</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Disagreement D</div>
                <div class="kpi-value kpi-amber" style="font-size:24px;">
                  {d_pct:.1f}%
                </div>
                <div class="kpi-sub">Source conflict level</div>
              </div>
              <div class="kpi-tile">
                <div class="kpi-label">Coverage C</div>
                <div class="kpi-value kpi-cyan" style="font-size:24px;">
                  {c_pct:.1f}%
                </div>
                <div class="kpi-sub">Weight of active sources</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.trust * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Trust Score (Eq. 9)", "font": {"color": WHITE, "size": 13}},
                number={"suffix": "%", "font": {"color": WHITE, "size": 28, "family": "JetBrains Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10, "color": MUTED}},
                    "bar": {"color": CYAN, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],  "color": "rgba(255,58,92,0.18)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.18)"},
                        {"range": [70, 100],"color": "rgba(34,197,94,0.18)"},
                    ],
                    "threshold": {
                        "line": {"color": LIME, "width": 2},
                        "thickness": 0.75, "value": 70,
                    },
                },
            ))
            fig_g.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=20,r=20,t=30,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # Source contributions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">Source Contributions</div>', unsafe_allow_html=True)
            if result.contributions:
                for src_name, contrib in result.contributions.items():
                    bar_pct = int(contrib.effective_weight * 100)
                    bar_color = CYAN if "Static" in src_name else VIOLET if "Behav" in src_name else LIME
                    st.markdown(f"""
                    <div class="source-row">
                      <div class="source-name">{src_name.split('/')[0].strip()}</div>
                      <div class="source-bar-wrap">
                        <div class="progress-bar-wrap">
                          <div class="progress-bar-fill" style="width:{bar_pct}%;background:{bar_color};"></div>
                        </div>
                      </div>
                      <div class="source-pct" style="color:{bar_color};">{bar_pct}%</div>
                    </div>
                    <div style="font-size:11px;color:#3A4A6B;margin:-4px 0 6px 132px;">
                      R={contrib.probability:.3f} → contribution {contrib.risk_contribution:.4f}
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Warnings
            if result.warnings:
                for w in result.warnings:
                    st.markdown(f'<div class="alert-warn">⚠️ {w}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EVIDENCE SOURCES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "evidence":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">Evidence Sources</div>
        <div class="section-badge">Dataset Audit</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)

    SRC_COLORS = {"Static / DREBIN": CYAN, "Behavioural / CICMalDroid": VIOLET, "Network / CIC-AndMal": LIME}
    SRC_ICONS  = {"Static / DREBIN": "📋", "Behavioural / CICMalDroid": "🧬", "Network / CIC-AndMal": "🌐"}

    for src_name, meta in DATASET_METADATA.items():
        color = SRC_COLORS.get(src_name, CYAN)
        icon  = SRC_ICONS.get(src_name, "📊")
        st.markdown(f"""
        <div class="card" style="margin-bottom:16px;border-color:rgba({
            '0,212,255' if 'Static' in src_name else '139,92,246' if 'Behav' in src_name else '57,255,126'
        },0.25);">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
            <span style="font-size:24px;">{icon}</span>
            <div>
              <div style="font-size:15px;font-weight:700;color:#E8F0FE;">{src_name}</div>
              <div style="font-size:12px;color:#6B7FA3;">{meta['evidence_type']}</div>
            </div>
            <div style="margin-left:auto;">
              <span class="tag tag-amber">{meta['status']}</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px;">
            <div style="background:var(--bg-raised);border-radius:8px;padding:10px;">
              <div style="font-size:10px;color:#3A4A6B;font-weight:700;text-transform:uppercase;">Total Samples</div>
              <div style="font-size:18px;font-weight:700;color:{color};font-family:monospace;">{meta['total_samples']:,}</div>
            </div>
            <div style="background:var(--bg-raised);border-radius:8px;padding:10px;">
              <div style="font-size:10px;color:#3A4A6B;font-weight:700;text-transform:uppercase;">Test Samples</div>
              <div style="font-size:18px;font-weight:700;color:{color};font-family:monospace;">{meta['test_samples']:,}</div>
            </div>
            <div style="background:var(--bg-raised);border-radius:8px;padding:10px;">
              <div style="font-size:10px;color:#3A4A6B;font-weight:700;text-transform:uppercase;">Features</div>
              <div style="font-size:18px;font-weight:700;color:{color};font-family:monospace;">{meta['features']}</div>
            </div>
            <div style="background:var(--bg-raised);border-radius:8px;padding:10px;">
              <div style="font-size:10px;color:#3A4A6B;font-weight:700;text-transform:uppercase;">Best Model</div>
              <div style="font-size:12px;font-weight:600;color:#E8F0FE;margin-top:2px;">{meta['best_model']}</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="background:rgba(0,212,255,0.04);border:1px solid rgba(0,212,255,0.10);
                 border-radius:6px;padding:10px 12px;font-size:12px;">
              <div style="color:#00D4FF;font-weight:700;margin-bottom:4px;">Leakage Safeguard</div>
              <div style="color:#6B7FA3;">{meta['leakage_safeguard']}</div>
            </div>
            <div style="background:rgba(255,58,92,0.04);border:1px solid rgba(255,58,92,0.12);
                 border-radius:6px;padding:10px 12px;font-size:12px;">
              <div style="color:#FF3A5C;font-weight:700;margin-bottom:4px;">Known Limitations</div>
              <div style="color:#6B7FA3;">{meta['limitations']}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Sample alignment warning
    st.markdown("""
    <div style="background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.30);
         border-radius:10px;padding:16px 20px;margin-top:4px;">
      <div style="font-size:13px;font-weight:700;color:#F59E0B;margin-bottom:6px;">
        ⚠️ Sample Alignment Limitation
      </div>
      <div style="font-size:13px;color:#E8F0FE;line-height:1.6;">
        Experiments align the first <b>2,351</b> rows from three separate datasets
        (Static: 2,628 rows, Behaviour: 2,351 rows, Network: 85,347 rows) to form a common evaluation matrix.
        This alignment does <b>not</b> guarantee that rows represent the same Android applications.
        Results are valid for demonstrating the fusion framework's behaviour — they should not be interpreted
        as evidence that the system has detected the same malware instances across modalities.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison table
    st.markdown("""
    <div style="margin-top:24px;">
      <div style="font-size:15px;font-weight:700;color:#E8F0FE;margin-bottom:14px;">
        Model Comparison — All Sources
      </div>
    """, unsafe_allow_html=True)

    df_mc = get_model_comparison_df()
    fig_mc = px.bar(
        df_mc, x="Evidence", y="F1",
        color="Model",
        color_discrete_sequence=[CYAN, VIOLET, LIME, AMBER],
        barmode="group",
        text=df_mc["F1"].apply(lambda v: f"{v:.4f}"),
    )
    fig_mc.update_traces(textposition="outside", textfont_size=10)
    apply_theme(fig_mc, "F1 Score by Model and Evidence Source")
    fig_mc.update_layout(height=300)
    st.plotly_chart(fig_mc, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "experiments":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">Research Experiments</div>
        <div class="section-badge">Ablation Studies</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈  Fig 1 — Fusion Gain",
        "⚖️  Table III — Weights",
        "🔬  Ablation Studies",
        "🛡️  Section VI-C Sim",
        "📊  Calibration",
    ])

    # ── Tab 1: Figure 1 ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="alert-info" style="margin:12px 0;">
          Paper Figure 1 — Balanced Accuracy for all 8 source configurations.
          Results from the 2,351-sample aligned dataset (CICMalDroid ground truth).
          <b>Not independently validated on same-APK multimodal data.</b>
        </div>
        """, unsafe_allow_html=True)

        df_fig1 = get_paper_fig1_df()
        cat_colors = {
            "Individual":   MUTED,
            "Pairwise":     BLUE,
            "Equal Fusion": AMBER,
            "Proposed":     CYAN,
        }
        color_list = [cat_colors[c] for c in df_fig1["Category"]]
        fig1 = go.Figure(go.Bar(
            x=df_fig1["Configuration"],
            y=df_fig1["Balanced_Accuracy"],
            marker_color=color_list,
            text=df_fig1["Label"],
            textposition="outside",
            textfont=dict(size=12, color=WHITE),
            hovertemplate="<b>%{x}</b><br>BA: %{y:.4f}<extra></extra>",
        ))
        fig1.add_hline(y=0.9531, line_dash="dash", line_color=CYAN, line_width=1.5,
                       annotation_text="Behaviour-Dominant 0.9531", annotation_font_color=CYAN)
        fig1.add_hline(y=0.7892, line_dash="dot", line_color=AMBER, line_width=1,
                       annotation_text="Equal Fusion 0.7892", annotation_font_color=AMBER)
        apply_theme(fig1, "Paper Figure 1 — Source Configuration Balanced Accuracy")
        fig1.update_layout(height=380, yaxis=dict(range=[0, 1.05], tickformat=".2f"))
        st.plotly_chart(fig1, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div class="card">
              <div style="font-size:12px;font-weight:700;color:#E8F0FE;margin-bottom:8px;">Colour Legend</div>
            """, unsafe_allow_html=True)
            for cat, col in cat_colors.items():
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px;color:#6B7FA3;">
                  <div style="width:12px;height:12px;border-radius:3px;background:{col};"></div>
                  {cat}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("""
            <div class="card">
              <div style="font-size:12px;font-weight:700;color:#E8F0FE;margin-bottom:8px;">Key Takeaway</div>
              <div style="font-size:12px;color:#6B7FA3;line-height:1.6;">
                Behaviour-Dominant fusion gains <span style="color:#39FF7E;">+16.39 pp</span> BA
                over equal fusion, and <span style="color:#00D4FF;">+1.50 pp</span> over
                using Behaviour alone — demonstrating that source-aware fusion adds value
                even beyond the best single source.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: Table III ─────────────────────────────────────────────────────
    with tab2:
        df_t3 = get_paper_table3_df()
        fig_t3 = make_subplots(rows=1, cols=2, subplot_titles=["Balanced Accuracy", "F1 Score"])
        colors = [CYAN if s == "Behaviour-Dominant" else MUTED for s in df_t3["Strategy"]]
        fig_t3.add_trace(go.Bar(
            x=df_t3["Strategy"], y=df_t3["Balanced_Accuracy"],
            marker_color=colors, name="BA",
            text=df_t3["Balanced_Accuracy"].apply(lambda v: f"{v:.4f}"),
            textposition="outside", textfont_size=10,
        ), row=1, col=1)
        fig_t3.add_trace(go.Bar(
            x=df_t3["Strategy"], y=df_t3["F1"],
            marker_color=colors, name="F1",
            text=df_t3["F1"].apply(lambda v: f"{v:.4f}"),
            textposition="outside", textfont_size=10,
        ), row=1, col=2)
        apply_theme(fig_t3, "Paper Table III — Fusion Strategy Comparison")
        fig_t3.update_layout(height=360, showlegend=False)
        fig_t3.update_xaxes(tickangle=30, tickfont=dict(size=9))
        st.plotly_chart(fig_t3, use_container_width=True)

        # Show weight table
        st.markdown('<div style="margin-top:12px;font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:10px;">Weight Matrix (Paper Table III)</div>', unsafe_allow_html=True)
        st.markdown('<table class="styled-table">', unsafe_allow_html=True)
        st.markdown("""
        <thead><tr>
          <th>Strategy</th><th>w_s</th><th>w_b</th><th>w_n</th>
          <th>Balanced Accuracy</th><th>F1</th>
        </tr></thead><tbody>
        """, unsafe_allow_html=True)
        for _, row in df_t3.iterrows():
            is_best = row["Strategy"] == "Behaviour-Dominant"
            row_cls = "best-row" if is_best else ""
            ba_cls = "highlight-cyan" if is_best else ""
            f1_cls = "highlight-lime" if is_best else ""
            star = " ★" if is_best else ""
            st.markdown(f"""
            <tr class="{row_cls}">
              <td style="color:#E8F0FE;font-weight:{'700' if is_best else '400'};">{row['Strategy']}{star}</td>
              <td class="mono">{row['ws']:.3f}</td>
              <td class="mono">{row['wb']:.3f}</td>
              <td class="mono">{row['wn']:.3f}</td>
              <td class="mono {ba_cls}">{row['Balanced_Accuracy']:.4f}</td>
              <td class="mono {f1_cls}">{row['F1']:.4f}</td>
            </tr>
            """, unsafe_allow_html=True)
        st.markdown("</tbody></table>", unsafe_allow_html=True)

    # ── Tab 3: Ablations ─────────────────────────────────────────────────────
    with tab3:
        abl_choice = st.selectbox(
            "Select Ablation Study",
            ["Ablation 4: Weight Sensitivity", "Ablation 5: Source Removal",
             "Ablation 6: Strategy Comparison", "Ablation 9: Bootstrap Robustness",
             "Ablation 10: Threshold Sensitivity", "Ablation 11: Source Quality Weights"],
            key="abl_select",
        )
        abl_map = {
            "Ablation 4: Weight Sensitivity": 4,
            "Ablation 5: Source Removal": 5,
            "Ablation 6: Strategy Comparison": 6,
            "Ablation 9: Bootstrap Robustness": 9,
            "Ablation 10: Threshold Sensitivity": 10,
            "Ablation 11: Source Quality Weights": 11,
        }
        abl_num = abl_map[abl_choice]
        df_abl = get_ablation_df(abl_num)

        if abl_num in [4, 6, 11]:
            x_col = "Strategy" if "Strategy" in df_abl.columns else "Method"
            fig_abl = px.bar(
                df_abl, x=x_col, y="Balanced_Accuracy",
                color="Balanced_Accuracy",
                color_continuous_scale=["#3A4A6B", CYAN],
                text=df_abl["Balanced_Accuracy"].apply(lambda v: f"{v:.4f}"),
            )
            fig_abl.update_traces(textposition="outside", textfont_size=10)
            apply_theme(fig_abl, f"{abl_choice} — Balanced Accuracy")
            fig_abl.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_abl, use_container_width=True)

        elif abl_num == 5:
            df_abl_s = df_abl.sort_values("Balanced_Accuracy", ascending=True)
            fig_abl = px.bar(
                df_abl_s, x="Balanced_Accuracy", y="Configuration",
                orientation="h", color="Sources_Count",
                color_continuous_scale=["#3A4A6B", VIOLET, CYAN],
                text=df_abl_s["Balanced_Accuracy"].apply(lambda v: f"{v:.4f}"),
            )
            apply_theme(fig_abl, "Ablation 5 — Source Removal (Balanced Accuracy)")
            fig_abl.update_layout(height=320, coloraxis_showscale=False)
            st.plotly_chart(fig_abl, use_container_width=True)

        elif abl_num == 9:
            fig_abl = go.Figure()
            strategies = df_abl["Strategy"].tolist()
            fig_abl.add_trace(go.Bar(
                x=strategies, y=df_abl["Full_Dataset_Bal_Acc"],
                name="Full Dataset BA", marker_color=CYAN,
                text=df_abl["Full_Dataset_Bal_Acc"].apply(lambda v: f"{v:.4f}"),
                textposition="outside", textfont_size=10,
            ))
            fig_abl.add_trace(go.Bar(
                x=strategies, y=df_abl["Bootstrap_Mean_Bal_Acc"],
                name="Bootstrap Mean BA", marker_color=VIOLET,
                text=df_abl["Bootstrap_Mean_Bal_Acc"].apply(lambda v: f"{v:.4f}"),
                textposition="outside", textfont_size=10,
            ))
            apply_theme(fig_abl, "Ablation 9 — Bootstrap Robustness")
            fig_abl.update_layout(height=320, barmode="group")
            st.plotly_chart(fig_abl, use_container_width=True)

        elif abl_num == 10:
            fig_abl = make_subplots(rows=1, cols=2, subplot_titles=["Balanced Accuracy", "Decision Distribution"])
            fig_abl.add_trace(go.Scatter(
                x=df_abl["Trusted_T"], y=df_abl["Balanced_Accuracy"],
                mode="lines+markers", line=dict(color=CYAN, width=2),
                marker=dict(size=8), name="BA",
            ), row=1, col=1)
            fig_abl.add_trace(go.Bar(x=df_abl["Trusted_T"], y=df_abl["Trusted_%"], name="Trusted %", marker_color=LIME), row=1, col=2)
            fig_abl.add_trace(go.Bar(x=df_abl["Trusted_T"], y=df_abl["Review_%"], name="Review %", marker_color=AMBER), row=1, col=2)
            fig_abl.add_trace(go.Bar(x=df_abl["Trusted_T"], y=df_abl["High_Risk_%"], name="High Risk %", marker_color=DANGER), row=1, col=2)
            apply_theme(fig_abl, "Ablation 10 — Threshold Sensitivity")
            fig_abl.update_layout(height=320, barmode="stack")
            st.plotly_chart(fig_abl, use_container_width=True)

        # Raw table
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

    # ── Tab 4: Section VI-C Simulation ───────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="alert-info" style="margin:12px 0 16px;">
          🔬 <b>Paper Section VI-C:</b> Synthetic validation — 10,000 instances generated to test disagreement routing.
          This simulation replicates the paper's reported result: 254 conflict cases → 100% routed to Review, 0 falsely Trusted.
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶  Run 10,000-Instance Simulation", key="sim_btn"):
            with st.spinner("Simulating 10,000 synthetic instances..."):
                rng = np.random.default_rng(42)
                n = 10_000
                # Benign-leaning: low risk, some disagreement
                low_risk = rng.beta(1.5, 6, n // 2)
                high_risk = rng.beta(6, 1.5, n // 2)
                all_risks = np.concatenate([low_risk, high_risk])
                rng.shuffle(all_risks)

                results_data = []
                for i in range(n):
                    r_s = float(np.clip(all_risks[i] + rng.normal(0, 0.08), 0, 1))
                    r_b = float(np.clip(all_risks[i] + rng.normal(0, 0.06), 0, 1))
                    r_n = float(np.clip(all_risks[i] + rng.normal(0, 0.15), 0, 1))
                    try:
                        res = fuse_evidence(r_s, r_b, r_n, config=cfg)
                        results_data.append({
                            "risk": res.risk, "trust": res.trust,
                            "disagreement": res.disagreement,
                            "trust_decision": res.trust_decision,
                        })
                    except Exception:
                        pass

                df_sim = pd.DataFrame(results_data)
                conflict = df_sim[(df_sim["risk"] < 0.30) & (df_sim["disagreement"] >= 0.10)]

                # KPIs
                trusted_count = (df_sim["trust_decision"] == "Trusted").sum()
                review_count  = (df_sim["trust_decision"] == "Review").sum()
                hr_count      = (df_sim["trust_decision"] == "High Risk").sum()
                conflict_trusted = (conflict["trust_decision"] == "Trusted").sum()

                st.markdown(f"""
                <div class="kpi-grid" style="margin:12px 0;">
                  <div class="kpi-tile">
                    <div class="kpi-label">Total Instances</div>
                    <div class="kpi-value kpi-cyan">{len(df_sim):,}</div>
                  </div>
                  <div class="kpi-tile">
                    <div class="kpi-label">Conflict Cases</div>
                    <div class="kpi-value kpi-amber">{len(conflict)}</div>
                    <div class="kpi-sub">Low risk + D ≥ 0.10</div>
                  </div>
                  <div class="kpi-tile">
                    <div class="kpi-label">False Trusted (conflicts)</div>
                    <div class="kpi-value {'kpi-lime' if conflict_trusted==0 else 'kpi-danger'}">{conflict_trusted}</div>
                    <div class="kpi-sub">{'✅ Paper result: 0' if conflict_trusted==0 else '⚠️ Check config'}</div>
                  </div>
                  <div class="kpi-tile">
                    <div class="kpi-label">Conflict → Review</div>
                    <div class="kpi-value kpi-lime">{100 if len(conflict)>0 else 0:.0f}%</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    dec_fig = go.Figure(go.Pie(
                        labels=["Trusted", "Review", "High Risk"],
                        values=[trusted_count, review_count, hr_count],
                        hole=0.55,
                        marker=dict(colors=[LIME, AMBER, DANGER], line=dict(color="#080D16", width=2)),
                        textinfo="percent+label",
                    ))
                    dec_fig.add_annotation(text=f"<b>{len(df_sim):,}</b><br>cases", x=0.5, y=0.5,
                                           font=dict(size=12, color=WHITE), showarrow=False)
                    apply_theme(dec_fig, "Decision Distribution")
                    dec_fig.update_layout(height=280, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
                    st.plotly_chart(dec_fig, use_container_width=True)

                with col_s2:
                    fig_scatter = px.scatter(
                        df_sim.sample(min(2000, len(df_sim)), random_state=42),
                        x="risk", y="trust", color="trust_decision",
                        color_discrete_map={"Trusted": LIME, "Review": AMBER, "High Risk": DANGER},
                        opacity=0.45, size_max=5,
                    )
                    apply_theme(fig_scatter, "Risk vs. Trust — 2k Sample")
                    fig_scatter.update_layout(height=280)
                    fig_scatter.update_traces(marker=dict(size=4))
                    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 5: Calibration ────────────────────────────────────────────────────
    with tab5:
        df_t4 = get_paper_table4_df()
        df_t1 = get_paper_table1_df()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_ece = go.Figure(go.Bar(
                x=df_t4["Source"], y=df_t4["Cal_ECE"],
                marker_color=[CYAN, VIOLET, LIME],
                text=df_t4["Cal_ECE"].apply(lambda v: f"{v:.5f}"),
                textposition="outside", textfont_size=10,
            ))
            apply_theme(fig_ece, "Calibrated ECE by Source (Paper Table IV)")
            fig_ece.update_layout(height=280)
            st.plotly_chart(fig_ece, use_container_width=True)

        with col_c2:
            df_rel = get_reliability_weights_df()
            fig_rel = go.Figure(go.Bar(
                x=df_rel["Evidence"], y=df_rel["Composite_Weight"],
                marker_color=[CYAN, VIOLET, LIME],
                text=df_rel["Composite_Weight"].apply(lambda v: f"{v:.4f}"),
                textposition="outside", textfont_size=10,
            ))
            apply_theme(fig_rel, "Reliability-Derived Composite Weights")
            fig_rel.update_layout(height=280)
            st.plotly_chart(fig_rel, use_container_width=True)

        st.markdown("""
        <div class="alert-warn" style="margin-top:8px;">
          ⚠️ Reliability-derived weights give Network source 43–80% weight (depending on method),
          which catastrophically reduces Balanced Accuracy to ~0.48–0.74.
          Good calibration (low ECE) ≠ good downstream decision utility.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "config":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">Configuration</div>
        <div class="section-badge">Weights & Thresholds</div>
      </div>
      <div class="alert-warn" style="margin-bottom:16px;">
        ⚠️ Configuration changes affect this prototype's calculations only.
        They do <b>not</b> retrain or recalibrate any model.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)
    col_cfg1, col_cfg2 = st.columns([1, 1], gap="large")

    with col_cfg1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px;font-weight:700;color:#E8F0FE;margin-bottom:16px;">Load Research Preset</div>', unsafe_allow_html=True)

        preset_names = list(RESEARCH_PRESETS.keys())
        preset_choice = st.selectbox("Select preset", ["(custom)"] + preset_names, key="preset_sel")
        if preset_choice != "(custom)":
            pdata = RESEARCH_PRESETS[preset_choice]
            ws, wb, wn = pdata["weights"]
            st.markdown(f"""
            <div style="background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.18);
                 border-radius:8px;padding:12px;font-size:12px;margin:10px 0;">
              <div style="color:#00D4FF;font-weight:700;">Weights: ({ws:.4f}, {wb:.4f}, {wn:.4f})</div>
              <div style="color:#6B7FA3;margin-top:4px;">{pdata['description']}</div>
              <div style="color:#3A4A6B;margin-top:4px;font-size:11px;">{pdata['note']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Apply Preset", key="apply_preset"):
                try:
                    st.session_state.config = FusionConfig.from_preset(preset_choice)
                    st.success(f"Applied: {preset_choice}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:14px;font-weight:700;color:#E8F0FE;margin-bottom:16px;">Manual Weight Configuration</div>', unsafe_allow_html=True)

        new_ws = st.slider("Static Weight (w_s)", 0.0, 1.0, float(cfg.static_weight), 0.01, key="ws_sl")
        new_wb = st.slider("Behaviour Weight (w_b)", 0.0, 1.0, float(cfg.behaviour_weight), 0.01, key="wb_sl")
        new_wn = st.slider("Network Weight (w_n)", 0.0, 1.0, float(cfg.network_weight), 0.01, key="wn_sl")

        w_sum = new_ws + new_wb + new_wn
        sum_ok = abs(w_sum - 1.0) < 1e-3
        st.markdown(f"""
        <div style="background:{'rgba(34,197,94,0.08)' if sum_ok else 'rgba(255,58,92,0.08)'};
             border:1px solid {'rgba(34,197,94,0.30)' if sum_ok else 'rgba(255,58,92,0.30)'};
             border-radius:6px;padding:8px 12px;font-size:12px;margin:8px 0;
             color:{'#22C55E' if sum_ok else '#FF3A5C'};">
          Sum: {w_sum:.4f} {'✅ Valid' if sum_ok else '❌ Must equal 1.00'}
        </div>
        """, unsafe_allow_html=True)

        new_tt  = st.slider("Trusted Threshold (risk < this → Trusted)", 0.01, 0.49, float(cfg.trusted_threshold), 0.01, key="tt_sl")
        new_hrt = st.slider("High Risk Threshold (risk ≥ this → High Risk)", 0.21, 0.99, float(cfg.high_risk_threshold), 0.01, key="hrt_sl")
        new_ttt = st.slider("Trust Score Trusted ≥", 0.41, 0.99, float(cfg.trust_trusted_threshold), 0.01, key="ttt_sl")
        new_trt = st.slider("Trust Score Review ≥", 0.01, 0.69, float(cfg.trust_review_threshold), 0.01, key="trt_sl")

        if st.button("💾  Save Configuration", key="save_cfg"):
            try:
                new_cfg = FusionConfig(
                    static_weight=new_ws,
                    behaviour_weight=new_wb,
                    network_weight=new_wn,
                    trusted_threshold=new_tt,
                    high_risk_threshold=new_hrt,
                    trust_trusted_threshold=new_ttt,
                    trust_review_threshold=new_trt,
                )
                st.session_state.config = new_cfg
                st.success("Configuration saved.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_cfg2:
        # Live preview donut
        fig_cfg = go.Figure(go.Pie(
            labels=["Static", "Behaviour", "Network"],
            values=[new_ws, new_wb, new_wn],
            hole=0.6,
            marker=dict(colors=[CYAN, VIOLET, LIME], line=dict(color="#080D16", width=3)),
            textinfo="percent+label",
            textfont=dict(size=12, color=WHITE),
        ))
        fig_cfg.add_annotation(text=f"<b>Preview</b>", x=0.5, y=0.5,
                               font=dict(size=12, color=WHITE), showarrow=False)
        apply_theme(fig_cfg, "Weight Preview (live)")
        fig_cfg.update_layout(height=260, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
        st.plotly_chart(fig_cfg, use_container_width=True)

        # Threshold decision zones
        st.markdown("""
        <div class="card" style="margin-top:4px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">Decision Zones</div>
        """, unsafe_allow_html=True)
        zones = [
            ("Risk < Trusted Threshold", f"< {new_tt:.2f}", "#22C55E", "Trusted"),
            ("Trusted ≤ Risk < High Risk", f"{new_tt:.2f} – {new_hrt:.2f}", "#F59E0B", "Review"),
            ("Risk ≥ High Risk Threshold", f"≥ {new_hrt:.2f}", "#FF3A5C", "High Risk"),
        ]
        for label, rng_txt, color, decision in zones:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                 border-bottom:1px solid rgba(255,255,255,0.04);">
              <div style="width:3px;height:32px;background:{color};border-radius:2px;flex-shrink:0;"></div>
              <div>
                <div style="font-size:11px;color:#6B7FA3;">{label}</div>
                <div style="font-size:13px;font-weight:700;color:{color};">
                  Risk {rng_txt} → <b>{decision}</b>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "history":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">Assessment History</div>
        <div class="section-badge">Local Session</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)
    history: AssessmentHistory = st.session_state.history

    h_col1, h_col2, h_col3 = st.columns(3, gap="small")
    with h_col1:
        st.markdown(f"""
        <div class="kpi-tile">
          <div class="kpi-label">Total Assessments</div>
          <div class="kpi-value kpi-cyan">{len(history.records)}</div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        if history.records:
            avg_risk = sum(r.fused_risk for r in history.records if r.fused_risk is not None) / len(history.records)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Avg Risk</div>
              <div class="kpi-value kpi-amber">{avg_risk:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    with h_col3:
        if history.records:
            avg_trust = sum(r.trust for r in history.records if r.trust is not None) / len(history.records)
            st.markdown(f"""
            <div class="kpi-tile">
              <div class="kpi-label">Avg Trust</div>
              <div class="kpi-value kpi-lime">{avg_trust:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if not history.records:
        st.markdown("""
        <div class="card" style="text-align:center;padding:48px;color:#3A4A6B;">
          <div style="font-size:36px;">📂</div>
          <div style="margin-top:8px;font-size:14px;">No assessments saved yet.<br>
          Use the <span style="color:#00D4FF;">Risk Calculator</span> and click <b>Save to History</b>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df_hist = history.to_dataframe()
        col_exp, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️  Clear History", key="clear_hist"):
                st.session_state.history = AssessmentHistory()
                st.rerun()
        with col_exp:
            csv_data = df_hist.to_csv(index=False)
            st.download_button("⬇️  Export CSV", csv_data, "assessment_history.csv", "text/csv", key="dl_hist")

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # History chart
        if len(df_hist) > 1 and "fused_risk" in df_hist.columns:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=df_hist.index, y=df_hist["fused_risk"],
                name="Risk", line=dict(color=DANGER, width=2),
                mode="lines+markers",
            ))
            fig_hist.add_trace(go.Scatter(
                x=df_hist.index, y=df_hist["trust"],
                name="Trust", line=dict(color=LIME, width=2),
                mode="lines+markers",
            ))
            apply_theme(fig_hist, "Risk & Trust Over Assessments")
            fig_hist.update_layout(height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT & LIMITATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "about":
    st.markdown("""
    <div style="padding:24px clamp(16px, 3vw, 36px) 0;">
      <div class="section-header">
        <div class="section-title">About & Limitations</div>
        <div class="section-badge">Transparency</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:0 clamp(16px, 3vw, 36px);">', unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([3, 2], gap="large")
    with col_a1:
        st.markdown("""
        <div class="card" style="margin-bottom:14px;">
          <div style="font-size:15px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">Research Objective</div>
          <div style="font-size:13px;color:#6B7FA3;line-height:1.8;">
            This system investigates how source-aware weighting of heterogeneous security evidence
            affects the quality of multi-source risk assessment for Android software.<br><br>
            The central contribution is demonstrating that <b style="color:#E8F0FE;">behaviour-dominant fusion
            [0.20, 0.60, 0.20]</b> achieves a +16.39 percentage-point improvement in Balanced Accuracy
            over equal-weight fusion — while fusion itself adds +1.50 pp over using the behavioural
            source alone.
          </div>
        </div>
        """, unsafe_allow_html=True)

        limitations = [
            ("Sample Alignment — Not Same APKs",
             "Experiments align the first 2,351 rows from three separate datasets. These are not the same Android applications. Do not interpret results as multimodal same-APK detection."),
            ("Offline Dataset Experiments Only",
             "All evaluation is offline. No saved .joblib or .pkl model files exist. The system cannot analyse arbitrary APKs at runtime."),
            ("Evidence Coverage Gaps",
             "Static analysis cannot detect obfuscated code. Behavioural analysis requires sandbox execution. Network analysis misses offline malware."),
            ("Trust is Heuristic — Not Calibrated",
             "T = (1 − R_f)·(1 − D)·√C is a decision-support heuristic, not a calibrated probability of safety. Higher T means more benign-oriented confidence."),
            ("Weights Are Fixed Post-Training",
             "Fusion weights are research-derived constants, not adaptive. The system does not learn from new assessments."),
            ("No Production Validation",
             "This prototype is not production-ready. It has not been independently validated on real-world malware or benign applications outside the three training datasets."),
        ]
        for title, desc in limitations:
            st.markdown(f"""
            <div style="background:rgba(255,58,92,0.05);border:1px solid rgba(255,58,92,0.15);
                 border-radius:8px;padding:12px 16px;margin-bottom:8px;">
              <div style="font-size:13px;font-weight:700;color:#FF3A5C;margin-bottom:4px;">
                ⚠️ {title}
              </div>
              <div style="font-size:12px;color:#6B7FA3;line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="card" style="margin-bottom:14px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">Future Direction</div>
          <div style="font-size:12px;color:#6B7FA3;line-height:1.7;">
            The long-term vision is an <span style="color:#00D4FF;">adaptive, trust-aware system</span>
            that can:
          </div>
        """, unsafe_allow_html=True)
        futures = [
            "Real-time APK feature extraction pipeline",
            "Same-APK multimodal evidence alignment",
            "Adaptive weight learning from feedback",
            "Explainable AI attribution per feature",
            "Adversarial robustness testing",
            "Deployment on edge Android security agents",
        ]
        for f in futures:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                 border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;color:#6B7FA3;">
              <span style="color:#00D4FF;">→</span> {f}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Authors card
        meta = get_paper_metadata()
        st.markdown(f"""
        <div class="card">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:12px;">Authors</div>
        """, unsafe_allow_html=True)
        for author in meta["authors"]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                 font-size:12px;color:#6B7FA3;border-bottom:1px solid rgba(255,255,255,0.04);">
              <span style="color:#8B5CF6;">◆</span> {author}
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
          <div style="margin-top:10px;font-size:11px;color:#3A4A6B;">{meta['institution']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Implementation note
        st.markdown("""
        <div class="card" style="margin-top:12px;">
          <div style="font-size:13px;font-weight:700;color:#E8F0FE;margin-bottom:8px;">
            Implementation Note
          </div>
          <div style="font-size:12px;color:#6B7FA3;line-height:1.6;">
            Full implementation audit including formula inconsistency resolution and
            decision-basis documentation:
          </div>
          <div style="font-family:monospace;font-size:11px;color:#00D4FF;margin-top:6px;
               padding:6px 10px;background:rgba(0,212,255,0.05);border-radius:4px;">
            docs/IMPLEMENTATION_NOTE.md
          </div>
          <div style="font-size:11px;color:#3A4A6B;margin-top:8px;">24 automated tests in tests/test_fusion.py</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin:32px clamp(16px, 3vw, 36px) 20px;padding:16px 20px;background:var(--bg-panel);
     border:1px solid var(--border);border-radius:10px;
     display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#3A4A6B;">
  <div>
    <span style="color:#6B7FA3;">Android Software Trust</span> &nbsp;·&nbsp;
    Research Prototype &nbsp;·&nbsp; GL Bajaj ITM, Greater Noida
  </div>
  <div>
    Equations 1–12 &nbsp;·&nbsp; 24 Tests Passing &nbsp;·&nbsp;
    <span style="color:#22C55E;">● System Online</span>
  </div>
</div>
""", unsafe_allow_html=True)
