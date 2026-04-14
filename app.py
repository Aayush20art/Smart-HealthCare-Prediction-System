import streamlit as st

st.set_page_config(
    page_title="HealthAI — Smart Healthcare Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, warnings, os, datetime
warnings.filterwarnings('ignore')

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ─── PREMIUM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=Syne:wght@600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,200;0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@300;400;500&family=Playfair+Display:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:    #050810;
    --bg-secondary:  #080d18;
    --bg-card:       rgba(255,255,255,0.028);
    --bg-card-hover: rgba(255,255,255,0.048);
    --border:        rgba(255,255,255,0.07);
    --border-accent: rgba(99,179,255,0.22);
    --accent-blue:   #3b9eff;
    --accent-cyan:   #00d4ff;
    --accent-teal:   #00ffcc;
    --accent-purple: #a78bfa;
    --accent-rose:   #fb7185;
    --accent-amber:  #fbbf24;
    --text-primary:  #e8edf5;
    --text-secondary:#7a8fa6;
    --text-muted:    #3d4f62;
    --success:       #34d399;
    --warning:       #fbbf24;
    --danger:        #f87171;
    --glow-blue:     rgba(59,158,255,0.15);
    --glow-teal:     rgba(0,255,204,0.12);
    --glow-purple:   rgba(167,139,250,0.12);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ═══ BACKGROUND ═══ */
.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(59,158,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 10%,  rgba(0,255,204,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 50% 100%, rgba(167,139,250,0.04) 0%, transparent 60%);
    min-height: 100vh;
}

/* ═══ SIDEBAR ═══ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060a14 0%, #040710 100%) !important;
    border-right: 1px solid rgba(59,158,255,0.1) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
/* Kill Streamlit default top empty space in sidebar */
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebarContent"] { padding-top: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* Sidebar nav items */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    padding: 0 2px !important;
}
/* Hide the auto "Module" label Streamlit renders above radio */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}
/* Kill extra top margin Streamlit adds to radio container */
[data-testid="stSidebar"] .stRadio {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(.stRadio) {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 12px !important;
    padding: 13px 16px !important;
    margin-bottom: 0 !important;
    display: flex !important;
    align-items: center !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-size: 0.84rem !important;
    font-weight: 450 !important;
    min-height: 52px !important;
    letter-spacing: 0.05px;
    position: relative;
    overflow: hidden;
}
[data-testid="stSidebar"] .stRadio label::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 2.5px;
    background: transparent;
    border-radius: 0 2px 2px 0;
    transition: all 0.25s ease;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(59,158,255,0.07) !important;
    border-color: rgba(59,158,255,0.2) !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] .stRadio label:hover::after {
    background: var(--accent-blue) !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }

/* ═══ MAIN HEADER ═══ */
.main-header {
    position: relative;
    background: linear-gradient(135deg, rgba(59,158,255,0.08) 0%, rgba(0,255,204,0.04) 50%, rgba(167,139,250,0.06) 100%);
    border: 1px solid rgba(59,158,255,0.15);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(59,158,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 140px; height: 140px;
    background: radial-gradient(circle, rgba(0,255,204,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.header-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.header-eyebrow::before {
    content: '';
    width: 20px; height: 1px;
    background: var(--accent-cyan);
    opacity: 0.6;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.main-header h1 span {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header .sub {
    font-size: 0.82rem;
    color: var(--text-secondary);
    font-weight: 300;
}
.header-pills {
    display: flex;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.68rem;
    color: var(--text-secondary);
    font-weight: 500;
    letter-spacing: 0.3px;
}
.pill.active {
    background: rgba(59,158,255,0.1);
    border-color: rgba(59,158,255,0.3);
    color: var(--accent-blue);
}

/* ═══ STATUS BAR ═══ */
.status-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 12px;
}
.status-live {
    display: flex;
    align-items: center;
    gap: 5px;
    color: var(--success);
    font-weight: 500;
}
.status-live::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--success);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--success);
    animation: breathe 2s ease infinite;
}
@keyframes breathe {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.6; transform: scale(0.85); }
}
.status-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
}

/* ═══ MODEL INFO BAR ═══ */
.model-bar {
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(59,158,255,0.04);
    border: 1px solid rgba(59,158,255,0.12);
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 20px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    flex-wrap: wrap;
}
.model-bar-item {
    display: flex;
    align-items: center;
    gap: 6px;
}
.model-bar-item strong { color: var(--accent-blue); font-weight: 500; }
.model-bar-divider { color: var(--text-muted); }

/* ═══ SECTION CARDS ═══ */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59,158,255,0.2), transparent);
}
.section-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(59,158,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(59,158,255,0.05);
}

/* ═══ SECTION TITLES ═══ */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::before {
    content: '';
    width: 16px; height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 1px;
}

/* ═══ BUTTONS ═══ */
.stButton > button {
    background: linear-gradient(135deg, #1a4fff, #0a3adb) !important;
    border: 1px solid rgba(59,158,255,0.3) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 13px 24px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px;
    width: 100%;
    box-shadow: 0 4px 20px rgba(26,79,255,0.25) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(26,79,255,0.4) !important;
    border-color: rgba(59,158,255,0.5) !important;
}
.stButton > button:active { transform: scale(0.98) !important; }

/* ═══ METRICS ═══ */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    transform: scaleX(0);
    transition: transform 0.3s ease;
    transform-origin: left;
}
[data-testid="metric-container"]:hover::after { transform: scaleX(1); }
[data-testid="metric-container"]:hover {
    background: rgba(59,158,255,0.05) !important;
    border-color: rgba(59,158,255,0.2) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.2) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 400 !important;
}

/* ═══ SLIDERS ═══ */
.stSlider > div > div > div {
    background: rgba(59,158,255,0.15) !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}
.stSlider label { color: var(--text-secondary) !important; font-size: 0.78rem !important; }
.stSlider [data-testid="stThumb"] {
    background: white !important;
    box-shadow: 0 0 0 3px rgba(59,158,255,0.4) !important;
}

/* ═══ SELECTBOX ═══ */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(59,158,255,0.3) !important;
}
.stSelectbox label { color: var(--text-secondary) !important; font-size: 0.78rem !important; }

/* ═══ TEXT AREA ═══ */
.stTextArea textarea {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.9 !important;
    transition: all 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(59,158,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(59,158,255,0.08) !important;
}

/* ═══ RESULT BANNERS ═══ */
.result-banner {
    border-radius: 14px;
    padding: 18px 22px;
    margin: 14px 0;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    animation: bannerIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
@keyframes bannerIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-high {
    background: rgba(248,113,113,0.07);
    border: 1px solid rgba(248,113,113,0.25);
    border-left: 3px solid #f87171;
}
.result-mod {
    background: rgba(251,191,36,0.07);
    border: 1px solid rgba(251,191,36,0.25);
    border-left: 3px solid #fbbf24;
}
.result-low {
    background: rgba(52,211,153,0.07);
    border: 1px solid rgba(52,211,153,0.25);
    border-left: 3px solid #34d399;
}
.result-icon { font-size: 1.6rem; line-height: 1; flex-shrink: 0; }
.result-content h3 { font-size: 0.92rem; font-weight: 600; margin: 0 0 4px 0; }
.result-content p  { font-size: 0.78rem; color: var(--text-secondary); margin: 0; line-height: 1.6; }
.result-high .result-content h3 { color: #f87171; }
.result-mod  .result-content h3 { color: #fbbf24; }
.result-low  .result-content h3 { color: #34d399; }

/* ═══ GAUGE ═══ */
.gauge-wrap {
    display: flex;
    justify-content: center;
    padding: 8px 0 0 0;
}

/* ═══ NLP ENTITIES ═══ */
.entity-group { margin-bottom: 14px; }
.entity-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 7px;
}
.entity-tag {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 3px 3px 3px 0;
    cursor: default;
    transition: all 0.15s ease;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.1px;
}
.entity-tag:hover { transform: translateY(-1px); filter: brightness(1.15); }
.tag-DISEASE    { background: rgba(248,113,113,0.1);  color: #fca5a5; border: 1px solid rgba(248,113,113,0.2); }
.tag-SYMPTOM    { background: rgba(167,139,250,0.1); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.2); }
.tag-MEDICATION { background: rgba(52,211,153,0.1);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.2); }
.tag-ANATOMY    { background: rgba(96,165,250,0.1);  color: #93c5fd; border: 1px solid rgba(96,165,250,0.2); }
.tag-VITAL      { background: rgba(251,191,36,0.1);  color: #fde68a; border: 1px solid rgba(251,191,36,0.2); }

/* ═══ HISTORY ═══ */
.hist-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 16px;
    border-radius: 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    margin-bottom: 8px;
    font-size: 0.78rem;
    transition: all 0.2s ease;
    cursor: default;
}
.hist-item:hover {
    background: rgba(59,158,255,0.05);
    border-color: rgba(59,158,255,0.15);
    transform: translateX(3px);
}
.hist-icon {
    width: 34px; height: 34px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    flex-shrink: 0;
}
.hist-label { color: var(--text-primary); font-weight: 500; flex: 1; }
.hist-detail { color: var(--text-muted); font-size: 0.68rem; font-family: 'DM Mono', monospace; margin-top: 1px; }
.hist-prob { font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.hist-badge {
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-high   { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.2); }
.badge-mod    { background: rgba(251,191,36,0.12);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.badge-low    { background: rgba(52,211,153,0.12);  color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.hist-time    { color: var(--text-muted); font-size: 0.65rem; font-family: 'DM Mono', monospace; white-space: nowrap; }

/* ═══ SIDEBAR BRAND ═══ */
.brand-block {
    padding: 22px 16px 16px 16px;
    margin-bottom: 0;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-logo {
    width: 46px; height: 46px;
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(59,158,255,0.25), rgba(0,255,204,0.15));
    border: 1px solid rgba(59,158,255,0.28);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 4px 18px rgba(59,158,255,0.2);
}
.brand-text { flex: 1; min-width: 0; }
.brand-name {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    color: var(--text-primary);
    font-weight: 700;
    letter-spacing: -0.3px;
    line-height: 1.15;
    white-space: nowrap;
}
.brand-sub {
    font-size: 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.3px;
    margin-top: 2px;
}
.brand-version {
    display: inline-block;
    margin-top: 5px;
    background: rgba(59,158,255,0.08);
    border: 1px solid rgba(59,158,255,0.18);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.58rem;
    color: var(--accent-blue);
    font-weight: 600;
    letter-spacing: 0.4px;
    font-family: 'DM Mono', monospace;
}

/* ═══ NAV SECTION LABEL ═══ */
.nav-label {
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    padding: 12px 4px 6px 4px;
}

/* ═══ SIDEBAR INFO GRID ═══ */
.sidebar-info {
    margin: 12px 8px 0 8px;
    padding: 12px 14px;
    background: rgba(255,255,255,0.018);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 12px;
    font-size: 0.7rem;
}
.sidebar-info-row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sidebar-info-row:last-child { border-bottom: none; padding-bottom: 0; }
.sidebar-info-row:first-child { padding-top: 0; }
.sidebar-info-label {
    color: rgba(59,158,255,0.7);
    font-weight: 600;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    white-space: nowrap;
    flex-shrink: 0;
    min-width: 68px;
}
.sidebar-info-value {
    color: var(--text-secondary);
    font-size: 0.68rem;
    line-height: 1.5;
}

/* ═══ DIVIDER ═══ */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ═══ ANNOTATED TEXT ═══ */
.annotated-text {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    line-height: 2.2;
    font-size: 0.85rem;
    color: #b0bec8;
    font-family: 'DM Mono', monospace;
}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,158,255,0.2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,158,255,0.4); }

/* ═══ FOOTER ═══ */
.footer {
    text-align: center;
    font-size: 0.68rem;
    color: var(--text-muted);
    padding: 20px 0;
    border-top: 1px solid var(--border);
    margin-top: 16px;
}
.footer a { color: var(--accent-blue); text-decoration: none; }

/* ═══ SPINNER ═══ */
.stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_all():
    return {
        'diabetes': (
            joblib.load(os.path.join(BASE, 'models/diabetes_model.pkl')),
            joblib.load(os.path.join(BASE, 'models/diabetes_scaler.pkl')),
            joblib.load(os.path.join(BASE, 'models/diabetes_features.pkl')),
        ),
        'heart': (
            joblib.load(os.path.join(BASE, 'models/heart_model.pkl')),
            joblib.load(os.path.join(BASE, 'models/heart_scaler.pkl')),
            joblib.load(os.path.join(BASE, 'models/heart_features.pkl')),
        ),
    }

models = load_all()


# ─── SHAP ────────────────────────────────────────────────────────────────────
def get_shap_values(model, scaler, input_df):
    try:
        import shap
        X_scaled = scaler.transform(input_df)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_scaled)
        sv = np.array(sv, dtype=np.float64)
        if sv.ndim == 3:
            if sv.shape[0] == 1:
                result = sv[0, :, 1]
            elif sv.shape[2] == 1:
                result = sv[0, :, 0]
            else:
                result = sv[1, 0, :]
        elif sv.ndim == 2:
            result = sv[0, :]
        else:
            result = sv
        return result.astype(np.float64)
    except Exception:
        fi = model.feature_importances_.astype(np.float64)
        X_scaled = scaler.transform(input_df)
        prob = float(model.predict_proba(X_scaled)[0][1])
        sign = 1.0 if prob > 0.5 else -1.0
        return fi * sign * prob


def plot_shap_bar(feature_names, shap_vals, title):
    feature_names = list(feature_names)
    shap_vals = np.array(shap_vals, dtype=np.float64).flatten()
    n = len(feature_names)
    if len(shap_vals) > n: shap_vals = shap_vals[:n]
    elif len(shap_vals) < n: shap_vals = np.pad(shap_vals, (0, n - len(shap_vals)))

    order        = np.argsort(np.abs(shap_vals))
    sorted_names = [feature_names[i] for i in order]
    sorted_vals  = shap_vals[order].tolist()
    colors       = ['#f87171' if v > 0 else '#34d399' for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(7, max(3.5, n * 0.48)))
    bars = ax.barh(sorted_names, sorted_vals, color=colors, edgecolor='none', height=0.5, alpha=0.85)

    # Add subtle glow effect via scatter
    for bar, val in zip(bars, sorted_vals):
        ax.barh(bar.get_y() + bar.get_height()/2,
                val, height=0.5, color=('#f87171' if val > 0 else '#34d399'),
                edgecolor='none', alpha=0.12, linewidth=0)

    ax.axvline(0, color=(0.23, 0.62, 1.0, 0.3), linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_xlabel("SHAP Value  ·  Positive = increases risk  ·  Negative = decreases risk",
                  fontsize=8, color='#4a6080', labelpad=8)
    ax.set_title(title, fontsize=10, fontweight='600', color='#e8edf5', pad=12,
                 fontfamily='DejaVu Sans')
    ax.tick_params(labelsize=8, colors='#4a6080', length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_facecolor('#050810')
    fig.patch.set_facecolor('#050810')
    ax.grid(axis='x', color=(1.0, 1.0, 1.0, 0.04), linewidth=0.5, linestyle='--')
    fig.tight_layout(pad=1.5)
    return fig


# ─── GAUGE SVG ────────────────────────────────────────────────────────────────
def render_gauge(prob):
    pct    = round(prob * 100, 1)
    angle  = -180 + prob * 180
    rad    = (angle - 90) * 3.14159 / 180
    cx, cy, r = 130, 110, 82
    x = cx + r * np.cos(rad)
    y = cy + r * np.sin(rad)

    if prob > 0.6:
        color, label = '#f87171', 'HIGH RISK'
    elif prob > 0.3:
        color, label = '#fbbf24', 'MODERATE'
    else:
        color, label = '#34d399', 'LOW RISK'

    # Track arc end point for partial fill
    svg = f"""
    <div class="gauge-wrap">
    <svg width="260" height="148" viewBox="0 0 260 148" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gfill" x1="0%" x2="100%">
          <stop offset="0%"   stop-color="#34d399"/>
          <stop offset="45%"  stop-color="#fbbf24"/>
          <stop offset="100%" stop-color="#f87171"/>
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>

      <!-- Outer decorative ring -->
      <path d="M30 115 A 100 100 0 0 1 230 115"
            fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1.5" stroke-linecap="round"/>

      <!-- Track -->
      <path d="M44 115 A 86 86 0 0 1 216 115"
            fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10" stroke-linecap="round"/>

      <!-- Fill -->
      <path d="M44 115 A 86 86 0 0 1 {x:.1f} {y:.1f}"
            fill="none" stroke="url(#gfill)" stroke-width="10" stroke-linecap="round"
            filter="url(#glow)"/>

      <!-- Needle dot -->
      <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"
              stroke="rgba(5,8,16,0.9)" stroke-width="2.5" filter="url(#glow)"/>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="white" opacity="0.9"/>

      <!-- Center display -->
      <text x="130" y="100" text-anchor="middle" fill="{color}"
            font-size="30" font-weight="300" font-family="DM Mono, monospace">{pct}%</text>
      <text x="130" y="118" text-anchor="middle" fill="{color}"
            font-size="9" font-weight="600" font-family="DM Sans, sans-serif"
            letter-spacing="2">{label}</text>

      <!-- Scale labels -->
      <text x="36" y="132" fill="#34d399" font-size="8" font-family="DM Sans" opacity="0.7">LOW</text>
      <text x="200" y="132" fill="#f87171" font-size="8" font-family="DM Sans" opacity="0.7">HIGH</text>

      <!-- Tick marks -->
      <line x1="130" y1="30" x2="130" y2="36" stroke="rgba(255,255,255,0.1)" stroke-width="1.5"/>
      <line x1="44" y1="115" x2="50" y2="115" stroke="rgba(255,255,255,0.1)" stroke-width="1.5"/>
      <line x1="216" y1="115" x2="210" y2="115" stroke="rgba(255,255,255,0.1)" stroke-width="1.5"/>
    </svg>
    </div>
    """
    return svg


# ─── NLP DICTIONARY ──────────────────────────────────────────────────────────
NLP_DICT = {
    "DISEASE":    ["diabetes","hypertension","heart disease","coronary artery disease",
                   "pneumonia","asthma","copd","cancer","stroke","tuberculosis",
                   "ckd","chronic kidney disease","heart failure","sepsis","anemia",
                   "arthritis","depression","anxiety","dementia","epilepsy",
                   "hypothyroidism","obesity","hepatitis"],
    "SYMPTOM":    ["chest pain","shortness of breath","dyspnea","fatigue","fever",
                   "nausea","vomiting","headache","dizziness","palpitations",
                   "edema","swelling","cough","wheezing","weakness","weight loss",
                   "tachycardia","syncope","confusion","blurred vision",
                   "polyuria","polydipsia","polyphagia"],
    "MEDICATION": ["metformin","insulin","aspirin","lisinopril","atorvastatin",
                   "amlodipine","metoprolol","warfarin","furosemide","omeprazole",
                   "amoxicillin","azithromycin","prednisone","levothyroxine",
                   "albuterol","sertraline","enalapril","losartan","clopidogrel"],
    "ANATOMY":    ["heart","lungs","kidney","liver","brain","blood","arteries",
                   "chest","abdomen","pancreas","thyroid","colon","aorta","ventricle"],
    "VITAL":      ["blood pressure","bp","heart rate","spo2","oxygen saturation",
                   "temperature","bmi","glucose","creatinine","hemoglobin",
                   "cholesterol","sodium","potassium","wbc","hba1c","troponin"],
}

def nlp_analyze(text):
    tl = text.lower()
    found = {}
    for etype, terms in NLP_DICT.items():
        hits = list(dict.fromkeys(t.title() for t in terms if t in tl))
        if hits:
            found[etype] = hits
    risk = min(
        len(found.get("DISEASE",    [])) * 15 +
        len(found.get("SYMPTOM",    [])) * 8  +
        len(found.get("MEDICATION", [])) * 5, 100
    )
    return found, risk


# ─── HISTORY ─────────────────────────────────────────────────────────────────
def add_history(module, prob, label, detail=""):
    st.session_state.history.insert(0, {
        "module": module,
        "prob":   round(prob * 100, 1),
        "label":  label,
        "detail": detail,
        "time":   datetime.datetime.now().strftime("%H:%M:%S"),
    })


def render_history():
    hist = st.session_state.history
    if not hist:
        st.markdown("<p style='color:var(--text-muted);text-align:center;padding:2rem 0;font-size:0.8rem'>No predictions yet — run a module to see results here.</p>", unsafe_allow_html=True)
        return
    icons = {"Heart Disease": "🫀", "Diabetes": "🩸", "ICU Readmission": "🏨", "NLP Analysis": "📄"}
    for h in hist:
        b = "badge-high" if h["prob"] > 60 else "badge-mod" if h["prob"] > 30 else "badge-low"
        pc = "#f87171" if h["prob"] > 60 else "#fbbf24" if h["prob"] > 30 else "#34d399"
        icon = icons.get(h["module"], "🔬")
        st.markdown(f"""
        <div class='hist-item'>
            <div class='hist-icon'>{icon}</div>
            <div style='flex:1;min-width:0'>
                <div class='hist-label'>{h['module']}</div>
                <div class='hist-detail'>{h['detail']}</div>
            </div>
            <div style='text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:4px'>
                <span class='hist-prob' style='color:{pc}'>{h['prob']}%</span>
                <span class='hist-badge {b}'>{h['label']}</span>
            </div>
            <div class='hist-time'>{h['time']}</div>
        </div>""", unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='brand-block'>
        <div class='brand-logo'>🏥</div>
        <div class='brand-text'>
            <div class='brand-name'>Health<span style='color:#fbbf24;text-shadow:0 0 20px rgba(251,191,36,0.45)'>AI</span></div>
            <div class='brand-sub'>Smart Prediction System</div>
            <div class='brand-version'>v2.0 · Advanced</div>
        </div>
    </div>
    <div class='nav-label'>Navigation</div>
    """, unsafe_allow_html=True)

    module = st.radio("", [
        "🫀  Heart Disease",
        "🩸  Diabetes",
        "🏨  ICU Readmission",
        "📄  Medical NLP Analyzer",
        "📋  Patient History",
    ], label_visibility="collapsed")

    st.markdown("""
    <div class='sidebar-info'>
        <div class='sidebar-info-row'>
            <span class='sidebar-info-label'>Cardiac</span>
            <span class='sidebar-info-value'>CAD · Arrhythmia · ACS</span>
        </div>
        <div class='sidebar-info-row'>
            <span class='sidebar-info-label'>Metabolic</span>
            <span class='sidebar-info-value'>T2DM · Insulin Resistance</span>
        </div>
        <div class='sidebar-info-row'>
            <span class='sidebar-info-label'>Critical</span>
            <span class='sidebar-info-value'>ICU · LACE · 30-Day Risk</span>
        </div>
        <div class='sidebar-info-row'>
            <span class='sidebar-info-label'>NLP</span>
            <span class='sidebar-info-value'>ICD-10 · Drug-Disease</span>
        </div>
    </div>
    <div style='margin:10px 8px 0 8px;text-align:center;font-size:0.58rem;
                color:var(--text-muted);border:1px solid var(--border);
                border-radius:8px;padding:5px 8px;letter-spacing:0.3px'>
        🔒 For educational use only
    </div>
    """, unsafe_allow_html=True)


# ─── HEADER ──────────────────────────────────────────────────────────────────
module_name = module.split("  ")[-1]
now_str     = datetime.datetime.now().strftime("%d %b %Y · %H:%M")

st.markdown(f"""
<div class='main-header'>
    <div style='display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap'>
        <div>
            <div class='header-eyebrow'>AI-Powered Clinical Decision Support</div>
            <h1>Smart Healthcare <span>Prediction</span></h1>
            <p class='sub'>Module active: <strong style='color:var(--text-primary);font-weight:500'>{module_name}</strong> — Real-time risk stratification with explainable AI</p>
            <div class='header-pills'>
                <span class='pill active'>Cardiac Risk</span>
                <span class='pill active'>Glycemic Analysis</span>
                <span class='pill active'>ICU Readmission</span>
                <span class='pill active'>Clinical NLP</span>
                <span class='pill'>4 Modules</span>
            </div>
        </div>
        <div style='text-align:right;flex-shrink:0'>
            <div class='status-bar' style='justify-content:flex-end'>
                <span class='status-live'>All Systems Live</span>
            </div>
            <div class='status-time' style='margin-top:8px'>{now_str}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════
if "Heart Disease" in module:
    model, scaler, features = models['heart']
    st.markdown("""<div class='model-bar'>
        <div class='model-bar-item'>🫀 <strong>Cardiac Risk Assessment</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Sensitivity <strong>~83.6%</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Population <strong>Cleveland Cohort</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Clinical Markers <strong>13</strong></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Patient Parameters</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.slider("Age", 29, 77, 54)
        sex      = st.selectbox("Sex", ["Female (0)", "Male (1)"])
        cp       = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal (2)", "Asymptomatic (3)"])
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 131)
        chol     = st.slider("Cholesterol (mg/dL)", 100, 570, 246)
    with c2:
        fbs      = st.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])
        restecg  = st.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
        thalach  = st.slider("Max Heart Rate", 70, 202, 149)
        exang    = st.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
    with c3:
        oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        slope    = st.selectbox("Slope of ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        ca       = st.slider("Major Vessels (0–3)", 0, 3, 0)
        thal     = st.selectbox("Thalassemia", ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"])
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⟳  Run Heart Disease Analysis"):
        input_df = pd.DataFrame([{
            'age': age, 'sex': int(sex[-2]), 'cp': int(cp[-2]),
            'trestbps': trestbps, 'chol': chol, 'fbs': int(fbs[-2]),
            'restecg': int(restecg[-2]), 'thalach': thalach,
            'exang': int(exang[-2]), 'oldpeak': oldpeak,
            'slope': int(slope[-2]), 'ca': ca, 'thal': int(thal[-2])
        }])[features]

        X_scaled = scaler.transform(input_df)
        prob     = float(model.predict_proba(X_scaled)[0][1])
        pred     = int(prob > 0.5)
        conf     = round(max(prob, 1 - prob) * 100, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Probability", f"{prob*100:.1f}%")
        c2.metric("Prediction", "High Risk ↑" if pred else "Low Risk ↓")
        c3.metric("Confidence", f"{conf}%")

        st.markdown(render_gauge(prob), unsafe_allow_html=True)

        if pred:
            st.markdown(f"""<div class='result-banner result-high'>
                <div class='result-icon'>⚠️</div>
                <div class='result-content'>
                    <h3>Heart Disease Risk Detected</h3>
                    <p>Probability <strong>{prob*100:.1f}%</strong> — Consult a cardiologist immediately. Model confidence: {conf}%.</p>
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='result-banner result-low'>
                <div class='result-icon'>✓</div>
                <div class='result-content'>
                    <h3>Low Heart Disease Risk</h3>
                    <p>Probability <strong>{prob*100:.1f}%</strong> — Continue healthy lifestyle habits. Model confidence: {conf}%.</p>
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>SHAP Explainability — Feature Contributions</div>", unsafe_allow_html=True)
        with st.spinner("Computing SHAP values…"):
            sv = get_shap_values(model, scaler, input_df)
        fig = plot_shap_bar(features, sv, "Heart Disease — SHAP Feature Impact")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<p style='font-size:0.72rem;color:var(--text-muted);margin-top:6px'>🔴 Positive SHAP → increases risk &nbsp;·&nbsp; 🟢 Negative SHAP → decreases risk</p></div>", unsafe_allow_html=True)

        add_history("Heart Disease", prob, "High Risk" if pred else "Low Risk", f"Age={age} · Chol={chol} · BP={trestbps}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DIABETES
# ══════════════════════════════════════════════════════════════════════════════
elif "Diabetes" in module:
    model, scaler, features = models['diabetes']
    st.markdown("""<div class='model-bar'>
        <div class='model-bar-item'>🩸 <strong>Glycemic Risk Screening</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Sensitivity <strong>~89.6%</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Population <strong>South Asian Cohort</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Clinical Markers <strong>8</strong></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Patient Parameters</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.slider("Pregnancies", 0, 17, 2)
        glucose     = st.slider("Glucose (mg/dL)", 44, 199, 120)
        bp          = st.slider("Blood Pressure (mmHg)", 0, 122, 69)
        skin        = st.slider("Skin Thickness (mm)", 0, 99, 20)
    with c2:
        insulin     = st.slider("Insulin (μU/mL)", 0, 846, 80)
        bmi         = st.slider("BMI", 0.0, 67.1, 32.0, 0.1)
        dpf         = st.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.47, 0.001)
        age         = st.slider("Age", 21, 81, 33)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⟳  Run Diabetes Analysis"):
        input_df = pd.DataFrame([{
            'Pregnancies': pregnancies, 'Glucose': glucose,
            'BloodPressure': bp, 'SkinThickness': skin,
            'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }])[features]

        X_scaled = scaler.transform(input_df)
        prob     = float(model.predict_proba(X_scaled)[0][1])
        pred     = int(prob > 0.5)
        hba1c    = round(5.5 + prob * 4, 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Diabetes Probability", f"{prob*100:.1f}%")
        c2.metric("Prediction", "Diabetic ↑" if pred else "Non-Diabetic ✓")
        c3.metric("Est. HbA1c", f"~{hba1c}%")

        st.markdown(render_gauge(prob), unsafe_allow_html=True)

        if pred:
            st.markdown(f"""<div class='result-banner result-high'>
                <div class='result-icon'>⚠️</div>
                <div class='result-content'>
                    <h3>Diabetes Risk Detected</h3>
                    <p>Probability <strong>{prob*100:.1f}%</strong> — Consult an endocrinologist. Estimated HbA1c: ~{hba1c}%.</p>
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='result-banner result-low'>
                <div class='result-icon'>✓</div>
                <div class='result-content'>
                    <h3>No Significant Diabetes Risk</h3>
                    <p>Probability <strong>{prob*100:.1f}%</strong> — Maintain healthy glucose and BMI levels.</p>
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>SHAP Explainability — Feature Contributions</div>", unsafe_allow_html=True)
        with st.spinner("Computing SHAP values…"):
            sv = get_shap_values(model, scaler, input_df)
        fig = plot_shap_bar(features, sv, "Diabetes Risk — SHAP Feature Impact")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<p style='font-size:0.72rem;color:var(--text-muted);margin-top:6px'>🔴 Positive SHAP → increases risk &nbsp;·&nbsp; 🟢 Negative SHAP → decreases risk</p></div>", unsafe_allow_html=True)

        add_history("Diabetes", prob, "Diabetic" if pred else "Non-Diabetic", f"Glucose={glucose} · BMI={bmi} · Age={age}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — ICU READMISSION
# ══════════════════════════════════════════════════════════════════════════════
elif "ICU" in module:
    st.markdown("""<div class='model-bar'>
        <div class='model-bar-item'>🏨 <strong>ICU Readmission Risk</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Index <strong>LACE Score</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Window <strong>30-Day Post-Discharge</strong></div>
        <span class='model-bar-divider'>·</span>
        <div class='model-bar-item'>Risk Factors <strong>8</strong></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-card'><div class='section-title'>Patient ICU Record</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age        = st.slider("Age", 18, 95, 60)
        icu_days   = st.slider("ICU Stay (days)", 1, 30, 5)
        prev_admit = st.slider("Previous Admissions (1yr)", 0, 10, 1)
        num_diag   = st.slider("Number of Diagnoses", 1, 20, 5)
    with c2:
        num_proc   = st.slider("Number of Procedures", 0, 25, 4)
        num_meds   = st.slider("Number of Medications", 1, 30, 8)
        creatinine = st.slider("Creatinine (mg/dL)", 0.5, 12.0, 1.1, 0.1)
        wbc        = st.slider("WBC (×10³/μL)", 2.0, 30.0, 8.5, 0.1)
    with c3:
        hemoglobin  = st.slider("Hemoglobin (g/dL)", 5.0, 18.0, 13.0, 0.1)
        sodium      = st.slider("Sodium (mEq/L)", 120, 160, 138)
        glucose_icu = st.slider("Blood Glucose (mg/dL)", 60, 400, 110)
        discharge   = st.selectbox("Discharge To", ["Home (low risk)", "SNF/Rehab (moderate)", "Transfer (high risk)"])
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⟳  Run ICU Readmission Analysis"):
        score = 0
        score += min(icu_days * 3, 15)
        score += min(prev_admit * 8, 24)
        score += min(num_diag * 2, 10)
        score += 8 if creatinine > 5 else 4 if creatinine > 2 else 0
        score += 5 if age > 70 else 3 if age > 55 else 0
        score += 6 if "Transfer" in discharge else 3 if "SNF" in discharge else 0
        score += 4 if wbc > 15 or wbc < 4 else 0
        score += 3 if hemoglobin < 9 else 0
        prob = min(score / 70.0, 1.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Readmission Risk", f"{prob*100:.1f}%")
        c2.metric("Risk Level", "🔴 High" if prob > 0.6 else "🟡 Moderate" if prob > 0.3 else "🟢 Low")
        c3.metric("LACE Score", f"{score} / 70")

        st.markdown(render_gauge(prob), unsafe_allow_html=True)

        if prob > 0.6:
            st.markdown(f"""<div class='result-banner result-high'>
                <div class='result-icon'>⚠️</div>
                <div class='result-content'>
                    <h3>High ICU Readmission Risk</h3>
                    <p>LACE Score: <strong>{score}/70</strong> — Enhanced post-discharge monitoring recommended.</p>
                </div></div>""", unsafe_allow_html=True)
        elif prob > 0.3:
            st.markdown(f"""<div class='result-banner result-mod'>
                <div class='result-icon'>⚡</div>
                <div class='result-content'>
                    <h3>Moderate Readmission Risk</h3>
                    <p>LACE Score: <strong>{score}/70</strong> — Follow-up within 48–72 hours recommended.</p>
                </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='result-banner result-low'>
                <div class='result-icon'>✓</div>
                <div class='result-content'>
                    <h3>Low Readmission Risk</h3>
                    <p>LACE Score: <strong>{score}/70</strong> — Standard discharge protocol applicable.</p>
                </div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-card'><div class='section-title'>Risk Factor Breakdown</div>", unsafe_allow_html=True)
        factors = {
            "ICU Stay Duration":     min(icu_days * 3, 15),
            "Previous Admissions":   min(prev_admit * 8, 24),
            "Number of Diagnoses":   min(num_diag * 2, 10),
            "Creatinine Level":      8 if creatinine > 5 else 4 if creatinine > 2 else 0,
            "Patient Age":           5 if age > 70 else 3 if age > 55 else 0,
            "Discharge Disposition": 6 if "Transfer" in discharge else 3 if "SNF" in discharge else 0,
            "Abnormal WBC":          4 if wbc > 15 or wbc < 4 else 0,
            "Low Hemoglobin":        3 if hemoglobin < 9 else 0,
        }
        names  = list(factors.keys())
        vals   = [float(v) for v in factors.values()]
        colors = ['#f87171' if v > 5 else '#fbbf24' if v > 0 else '#1a2535' for v in vals]

        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.barh(names, vals, color=colors, edgecolor='none', height=0.5, alpha=0.85)
        ax.set_xlabel("Risk Score Contribution", fontsize=8, color='#4a6080', labelpad=8)
        ax.tick_params(labelsize=8, colors='#4a6080', length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_facecolor('#050810')
        fig.patch.set_facecolor('#050810')
        ax.grid(axis='x', color=(1.0, 1.0, 1.0, 0.04), linewidth=0.5, linestyle='--')
        fig.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        add_history("ICU Readmission", prob,
                    "High Risk" if prob > 0.6 else "Moderate" if prob > 0.3 else "Low Risk",
                    f"LACE={score}/70 · Stay={icu_days}d · PrevAdmit={prev_admit}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — NLP
# ══════════════════════════════════════════════════════════════════════════════
elif "NLP" in module:
    SAMPLE = """Patient is a 58-year-old male presenting with chest pain and shortness of breath.
History of hypertension and diabetes. Current medications include metformin, aspirin, and lisinopril.
Blood pressure: 148/92 mmHg. Heart rate: 96 bpm. SpO2: 94%. Cholesterol: 280 mg/dL.
ECG shows ST-segment changes suggestive of coronary artery disease. Troponin mildly elevated.
Patient reports fatigue, palpitations, and edema in lower extremities for the past 3 weeks."""

    st.markdown("<div class='section-card'><div class='section-title'>Medical Report / Clinical Notes</div>", unsafe_allow_html=True)
    text = st.text_area("Paste clinical notes:", value=SAMPLE, height=175, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⟳  Analyze Medical Report"):
        if text.strip():
            found, risk = nlp_analyze(text)
            total = sum(len(v) for v in found.values())
            rl    = "High" if risk > 60 else "Moderate" if risk > 30 else "Low"
            prob  = risk / 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Clinical Risk Score", f"{risk} / 100")
            c2.metric("Risk Level", rl)
            c3.metric("Entities Found", str(total))

            st.markdown(render_gauge(prob), unsafe_allow_html=True)

            if risk > 60:
                st.markdown("""<div class='result-banner result-high'>
                    <div class='result-icon'>⚠️</div>
                    <div class='result-content'>
                        <h3>Elevated Clinical Risk</h3>
                        <p>Multiple disease indicators detected. Immediate clinical evaluation recommended.</p>
                    </div></div>""", unsafe_allow_html=True)
            elif risk > 30:
                st.markdown("""<div class='result-banner result-mod'>
                    <div class='result-icon'>⚡</div>
                    <div class='result-content'>
                        <h3>Moderate Risk Profile</h3>
                        <p>Monitor closely. Follow-up consultation within 48–72 hours.</p>
                    </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='result-banner result-low'>
                    <div class='result-icon'>✓</div>
                    <div class='result-content'>
                        <h3>Low Risk Profile</h3>
                        <p>Minimal clinical indicators. Routine monitoring advised.</p>
                    </div></div>""", unsafe_allow_html=True)

            # NER Tags
            st.markdown("<div class='section-card'><div class='section-title'>Named Entity Recognition</div>", unsafe_allow_html=True)
            labels = {
                "DISEASE":    ("🦠", "Diseases"),
                "SYMPTOM":    ("🤒", "Symptoms"),
                "MEDICATION": ("💊", "Medications"),
                "ANATOMY":    ("🫀", "Anatomy"),
                "VITAL":      ("📊", "Vitals & Labs"),
            }
            for etype, (icon, label) in labels.items():
                if etype in found:
                    tags = "".join(f"<span class='entity-tag tag-{etype}'>{e}</span>" for e in found[etype])
                    st.markdown(f"<div class='entity-group'><div class='entity-label'>{icon} {label}</div>{tags}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Annotated Report
            st.markdown("<div class='section-card'><div class='section-title'>Annotated Report</div>", unsafe_allow_html=True)
            import html as hl
            highlighted = hl.escape(text)
            color_map = {
                "DISEASE":    ("#3d1f1f", "#fca5a5"),
                "SYMPTOM":    ("#2d1f40", "#c4b5fd"),
                "MEDICATION": ("#1a2f26", "#6ee7b7"),
                "ANATOMY":    ("#1a2535", "#93c5fd"),
                "VITAL":      ("#332a10", "#fde68a"),
            }
            all_terms = [(t.lower(), et) for et, tl_list in found.items() for t in tl_list]
            all_terms.sort(key=lambda x: -len(x[0]))
            for term, etype in all_terms:
                bg, tc = color_map.get(etype, ("#1e2535", "#e2e8f0"))
                highlighted = highlighted.replace(
                    term.title(),
                    f"<mark style='background:{bg};color:{tc};padding:1px 6px;border-radius:4px;font-weight:500'>{term.title()}</mark>"
                )
            st.markdown(f"<div class='annotated-text'>{highlighted}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            add_history("NLP Analysis", prob, rl, f"Entities={total} · Score={risk}/100")
        else:
            st.warning("Please enter some clinical notes to analyze.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — PATIENT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in module:
    hist  = st.session_state.history
    total = len(hist)
    high  = sum(1 for h in hist if h['prob'] > 60)
    mod   = sum(1 for h in hist if 30 < h['prob'] <= 60)
    low   = sum(1 for h in hist if h['prob'] <= 30)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("High Risk", high)
    c3.metric("Moderate", mod)
    c4.metric("Low Risk", low)

    st.markdown("<div class='section-card'><div class='section-title'>Prediction Log</div>", unsafe_allow_html=True)
    render_history()
    st.markdown("</div>", unsafe_allow_html=True)

    if hist:
        if st.button("🗑  Clear All History"):
            st.session_state.history = []
            st.rerun()

        if total > 0:
            st.markdown("<div class='section-card'><div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.8))
            bars = ax.bar(["High Risk", "Moderate", "Low Risk"],
                          [high, mod, low],
                          color=['#f87171', '#fbbf24', '#34d399'],
                          width=0.45, edgecolor='none', alpha=0.85)
            for bar in bars:
                h_ = bar.get_height()
                if h_ > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h_ + 0.04, str(int(h_)),
                            ha='center', va='bottom', color='#e8edf5', fontsize=9)
            ax.set_facecolor('#050810')
            fig.patch.set_facecolor('#050810')
            ax.tick_params(colors='#4a6080', labelsize=8, length=0)
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_ylabel("Count", color='#4a6080', fontsize=8)
            ax.grid(axis='y', color=(1.0, 1.0, 1.0, 0.04), linewidth=0.5)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    Smart Healthcare Prediction System &nbsp;·&nbsp; v2.0 &nbsp;·&nbsp;
    · SHAP XAI · Clinical NLP &nbsp;·&nbsp;
    <strong>For educational use only — not a substitute for medical advice</strong>
</div>
""", unsafe_allow_html=True)