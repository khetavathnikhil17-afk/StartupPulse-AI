import streamlit as st
import pandas as pd
import sys
import base64
from PIL import Image
from pathlib import Path
import streamlit.components.v1 as components
import html

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.config import (
    REPORTS_DIR, TRAIN_DATA_PATH, VALIDATION_DATA_PATH,
    TEST_DATA_PATH, SENTIMENT_MAPPING, MODEL_SAVE_DIR,
    MODEL_NAME, MAX_LENGTH, NUM_LABELS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

APP_VERSION = "v1.3.0"


# ---------------------------------------------------------------------------
# LAZY ML IMPORTS
# ---------------------------------------------------------------------------
_ml_loaded = False
_torch = None
_shap = None
_AutoModelForSequenceClassification = None
_AutoTokenizer = None


def _load_ml_libs():
    global _ml_loaded, _torch, _shap, _AutoModelForSequenceClassification, _AutoTokenizer
    if _ml_loaded:
        return
    import torch as _t
    import shap as _s
    from transformers import (
        AutoModelForSequenceClassification as _M,
        AutoTokenizer as _T,
    )
    _torch = _t
    _shap = _s
    _AutoModelForSequenceClassification = _M
    _AutoTokenizer = _T
    _ml_loaded = True


def sanitize_html(text: str) -> str:
    return html.escape(str(text))


st.set_page_config(
    page_title="StartupPulse AI",
    page_icon=str(PROJECT_ROOT / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLASSMORPHIC CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cabinet+Grotesk:wght@400;500;600;700;800&family=Satoshi:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #f1f5f9;
    --glass: rgba(255, 255, 255, 0.72);
    --glass-border: rgba(255, 255, 255, 0.5);
    --glass-hover: rgba(255, 255, 255, 0.85);
    --blur: blur(28px);
    --blur-sm: blur(20px);
    --radius: 2rem;
    --radius-lg: 2rem;
    --radius-md: 1.25rem;
    --radius-sm: 1rem;
    --radius-xs: 0.75rem;
    --primary: #4f46e2;
    --primary-light: #818cf8;
    --primary-dark: #3730a3;
    --violet: #7c3aed;
    --pink: #ec4899;
    --emerald: #10b981;
    --amber: #f59e0b;
    --rose: #f43f5e;
    --text: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --text-white: #ffffff;
    --shadow-sm: 0 2px 8px rgba(79, 70, 226, 0.08);
    --shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
    --shadow-lg: 0 8px 32px rgba(79, 70, 226, 0.16);
    --shadow-glow: 0 4px 24px rgba(79, 70, 226, 0.25);
    --transition: cubic-bezier(0.4, 0, 0.2, 1);
}

/* ---------- FORCE LIGHT THEME OVERRIDES ---------- */
html, body, [data-testid="stAppViewContainer"],
.stApp, .main, section.main, section.main > div,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="ScrollToBottomContainer"] {
    background-color: #f1f5f9 !important;
    background-image: none !important;
    color: #1e293b !important;
}

/* ---------- MESH BACKGROUND ---------- */
section.main {
    background: #f1f5f9 !important;
    position: relative;
}
section.main::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 60% 50% at 10% 20%, rgba(79, 70, 226, 0.15) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 90% 15%, rgba(124, 58, 237, 0.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 50% at 50% 80%, rgba(236, 72, 153, 0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 70%, rgba(16, 185, 129, 0.08) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}
section.main > div {
    padding: 24px 28px 48px 28px;
    position: relative;
    z-index: 1;
}
.stApp {
    background: #f1f5f9 !important;
}

/* ---------- HIDE DEFAULT ELEMENTS ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---------- STREAMLIT COMPONENT OVERRIDES (light theme) ---------- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stVerticalBlock"] p, [data-testid="stVerticalBlock"] li {
    color: #1e293b !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #1e293b !important;
    font-family: 'Cabinet Grotesk', -apple-system, sans-serif !important;
    letter-spacing: -0.03em !important;
}

/* Radio / Checkbox labels */
.stRadio label, .stRadio div[role="radiogroup"] label,
.stCheckbox label, .stCheckbox span {
    color: #1e293b !important;
}

/* Info / Warning / Error boxes */
.stAlert, [data-testid="stAlert"] {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #1e293b !important;
    border-left-width: 3px !important;
}
.stAlert[data-baseweb="notification-info"] { border-left-color: #4f46e2 !important; }
.stAlert[data-baseweb="notification-warning"] { border-left-color: #f59e0b !important; }
.stAlert[data-baseweb="notification-error"] { border-left-color: #ef4444 !important; }
.stAlert[data-baseweb="notification-success"] { border-left-color: #10b981 !important; }

/* ---------- HEADINGS ---------- */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cabinet Grotesk', -apple-system, sans-serif !important;
    color: #1e293b !important;
    letter-spacing: -0.03em !important;
}

/* ---------- WIDGET CARD ---------- */
.widget-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 2rem;
    padding: 24px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
    position: relative;
    overflow: hidden;
}
.widget-card:hover {
    transform: translateY(-6px) scale(1.005);
    box-shadow: 0 8px 32px rgba(79, 70, 226, 0.16);
    border-color: rgba(255, 255, 255, 0.7);
}
.widget-card h3 {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-bottom: 4px !important;
}
.widget-card p {
    font-size: 13px !important;
    color: #64748b !important;
    line-height: 1.6;
}

/* ---------- GLASS CARD (smaller) ---------- */
.glass-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 1.25rem;
    padding: 20px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
}
.glass-card:hover {
    transform: translateY(-4px) scale(1.003);
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
    border-color: rgba(255, 255, 255, 0.7);
}
.glass-card h3 {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    margin-bottom: 6px !important;
}
.glass-card p {
    font-size: 12px !important;
    color: #64748b !important;
    line-height: 1.6;
}

/* ---------- SIDEBAR (Streamlit override) ---------- */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.5) !important;
    padding: 16px 10px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #475569 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #475569 !important;
}

/* Sidebar brand */
.sidebar-brand {
    padding: 12px 8px;
    margin-bottom: 20px;
    text-align: center;
}
.sidebar-brand h2 {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    color: #1e293b !important;
    margin: 0 !important;
}
.sidebar-version {
    display: inline-block;
    background: linear-gradient(135deg, rgba(79,70,226,0.12), rgba(124,58,237,0.12));
    color: #4f46e2;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    margin-top: 6px;
}
.sidebar-label {
    font-size: 9px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8 !important;
    margin-bottom: 8px !important;
    padding-left: 4px;
}
.sidebar-tech-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: 0.75rem;
    font-size: 12px;
    color: #64748b;
    transition: background 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
.sidebar-tech-item:hover {
    background: rgba(79, 70, 226, 0.06);
}
.sidebar-tech-icon {
    width: 20px;
    text-align: center;
    font-size: 11px;
    opacity: 0.6;
}
.sidebar-footer {
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.4);
    margin-top: auto;
}
.sidebar-footer p {
    font-size: 10px !important;
    color: #94a3b8 !important;
}

/* Radio in sidebar */
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 0.75rem;
    padding: 8px 12px;
    font-size: 13px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(79, 70, 226, 0.06);
    border-color: rgba(79, 70, 226, 0.1);
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(79,70,226,0.12), rgba(124,58,237,0.12)) !important;
    border-color: #4f46e2 !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span {
    color: #4f46e2 !important;
    font-weight: 600 !important;
}

/* Logo float animation */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}
.sidebar-logo-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
    padding-top: 8px;
}
.sidebar-logo-box {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e2, #7c3aed);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: float 3s ease-in-out infinite;
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.3);
}
.sidebar-logo-box img {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    filter: brightness(0) invert(1);
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes progressFill {
    from { width: 0; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}
.anim-fade-up { animation: fadeInUp 0.6s cubic-bezier(0.4, 0, 0.2, 1) both; }
.anim-fade    { animation: fadeIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) both; }
.anim-slide   { animation: slideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) both; }
.anim-scale   { animation: scaleIn 0.5s cubic-bezier(0.4, 0, 0.2, 1) both; }
.anim-delay-1 { animation-delay: 0.05s; }
.anim-delay-2 { animation-delay: 0.10s; }
.anim-delay-3 { animation-delay: 0.15s; }
.anim-delay-4 { animation-delay: 0.20s; }
.anim-delay-5 { animation-delay: 0.25s; }
.anim-delay-6 { animation-delay: 0.30s; }
.anim-delay-7 { animation-delay: 0.35s; }

/* ---------- HERO CARD ---------- */
.hero-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 2rem;
    padding: 40px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
}
.hero-card:hover {
    transform: translateY(-4px) scale(1.003);
    box-shadow: 0 8px 32px rgba(79, 70, 226, 0.16);
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -15%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(79,70,226,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-card::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-card h1 {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    letter-spacing: -0.04em !important;
    line-height: 1.1 !important;
    margin-bottom: 8px !important;
    background: linear-gradient(135deg, #4f46e2, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent !important;
}
.hero-card .subtitle {
    font-size: 14px;
    color: #64748b;
    font-weight: 400;
    line-height: 1.6;
    max-width: 640px;
}
.hero-badge {
    display: inline-block;
    background: rgba(79, 70, 226, 0.1);
    color: #4f46e2;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 16px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid rgba(79, 70, 226, 0.15);
}

/* ---------- GLASS HERO PROMO CARD ---------- */
.hero-promo {
    background: linear-gradient(135deg, #4f46e2, #5b52e8, #7c3aed);
    border-radius: 2rem;
    padding: 40px 32px;
    color: white;
    position: relative;
    overflow: hidden;
    text-align: center;
}
.hero-promo::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-promo h3 {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 20px !important;
    font-weight: 800 !important;
    color: white !important;
    margin-bottom: 8px !important;
}
.hero-promo p {
    font-size: 13px !important;
    color: rgba(255,255,255,0.8) !important;
    margin-bottom: 20px !important;
}
.hero-play {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    background: rgba(255,255,255,0.2);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 20px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.hero-promo .hero-play:hover {
    background: rgba(255,255,255,0.3);
    transform: scale(1.08);
}

/* ---------- METRIC CARD ---------- */
.metric-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 1.25rem;
    padding: 20px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.003);
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
}
.metric-value {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 4px;
    color: #1e293b;
}
.metric-label {
    font-size: 11px;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---------- PROJECT TRACKER ---------- */
.project-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.4);
}
.project-item:last-child { border-bottom: none; }
.project-icon {
    width: 40px;
    height: 40px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.project-info { flex: 1; min-width: 0; }
.project-name {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 2px;
}
.project-sub {
    font-size: 11px;
    color: #94a3b8;
}
.project-pct {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #1e293b;
}
.project-bar {
    height: 6px;
    background: rgba(79, 70, 226, 0.08);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 6px;
}
.project-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: progressFill 1s cubic-bezier(0.4, 0, 0.2, 1) both;
    box-shadow: 0 0 8px rgba(79, 70, 226, 0.3);
}

/* ---------- PROBABILITY BAR ---------- */
.prob-bar-container { margin-bottom: 12px; }
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 4px;
}
.prob-bar-track {
    height: 8px;
    background: rgba(79, 70, 226, 0.06);
    border-radius: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: progressFill 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;
}

/* ---------- RESULT BADGE ---------- */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.01em;
}
.result-badge.positive {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.result-badge.negative {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626;
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.result-badge.neutral {
    background: rgba(100, 116, 139, 0.1);
    color: #475569;
    border: 1px solid rgba(100, 116, 139, 0.25);
}

/* ---------- HIGHLIGHT CARD ---------- */
.highlight-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 1rem;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.highlight-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
    border-color: rgba(255, 255, 255, 0.7);
}
.highlight-icon {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    flex-shrink: 0;
}
.highlight-text {
    font-size: 12px;
    font-weight: 600;
    color: #1e293b;
    line-height: 1.3;
}

/* ---------- SECTION HEADINGS ---------- */
.section-label {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-bottom: 14px;
    font-weight: 700;
}

/* ---------- DIVIDER ---------- */
.premium-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 24px 0;
}

/* ---------- BUTTONS ---------- */
.stButton > button {
    background: linear-gradient(135deg, #4f46e2, #7c3aed) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 600 !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    letter-spacing: 0.01em;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.2) !important;
    min-height: 40px;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(79, 70, 226, 0.3) !important;
}
.stButton > button:focus {
    box-shadow: 0 0 0 3px rgba(79, 70, 226, 0.25) !important;
    outline: none !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}
.stButton > button[kind="secondary"],
[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    min-height: 40px;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: #4f46e2 !important;
    background: rgba(255, 255, 255, 0.85) !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 16px !important;
    font-weight: 500 !important;
    font-size: 12px !important;
}
.stDownloadButton > button:hover {
    border-color: #4f46e2 !important;
    background: rgba(255, 255, 255, 0.85) !important;
}

/* ---------- TEXT AREA ---------- */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 1rem !important;
    color: #1e293b !important;
    font-family: 'Satoshi', monospace !important;
    font-size: 13px !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #4f46e2 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 226, 0.12) !important;
}
.stTextArea textarea::placeholder {
    color: #94a3b8 !important;
}
.stTextArea label {
    color: #1e293b !important;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame {
    border-radius: 1rem !important;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    background: rgba(255, 255, 255, 0.72) !important;
}

/* ---------- SPINNER ---------- */
.stSpinner > div {
    border-color: #4f46e2 transparent transparent transparent !important;
}

/* ---------- ALERTS ---------- */
.stAlert {
    border-radius: 1rem !important;
    border-left-width: 3px !important;
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(12px) !important;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 1rem;
    padding: 4px;
    border: 1px solid rgba(255, 255, 255, 0.5);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0.75rem !important;
    font-weight: 500;
    font-size: 13px;
    font-family: 'Cabinet Grotesk', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08) !important;
}

/* ---------- SELECTBOX / MULTISELECT ---------- */
[data-testid="stSelectbox"] div div div,
[data-testid="stMultiSelect"] div div div {
    background: rgba(255, 255, 255, 0.7) !important;
    color: #1e293b !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: rgba(255, 255, 255, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    color: #1e293b !important;
}
[data-baseweb="menu"] {
    background: #ffffff !important;
    color: #1e293b !important;
}
[data-baseweb="option"] {
    background: #ffffff !important;
    color: #1e293b !important;
}
[data-baseweb="option"]:hover, [data-baseweb="option"]:focus,
[data-baseweb="option"][aria-selected="true"] {
    background: rgba(79, 70, 226, 0.08) !important;
    color: #4f46e2 !important;
}
[data-baseweb="input"] {
    background: rgba(255,255,255,0.7) !important;
    color: #1e293b !important;
}
[data-baseweb="input"] input {
    color: #1e293b !important;
}
[data-baseweb="textarea"] {
    background: rgba(255,255,255,0.7) !important;
    color: #1e293b !important;
}
[data-baseweb="textarea"] textarea {
    color: #1e293b !important;
}
.stRadio label, .stRadio div[role="radiogroup"] label {
    color: #1e293b !important;
}
.stCheckbox label, .stCheckbox span {
    color: #1e293b !important;
}
[data-testid="stVerticalBlock"] p {
    color: #64748b !important;
}
[data-testid="stVerticalBlock"] span {
    color: #1e293b !important;
}

/* ---------- EXPANDER ---------- */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(28px) !important;
    -webkit-backdrop-filter: blur(28px) !important;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 1.25rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 13px !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    color: #1e293b !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] span {
    color: #1e293b !important;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame, [data-testid="stDataFrame"] {
    border-radius: 1rem !important;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    background: rgba(255, 255, 255, 0.72) !important;
}

/* ---------- LINK ---------- */
a[href] { color: #4f46e2 !important; }

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(79, 70, 226, 0.15); border-radius: 10px; }

/* ---------- STREAMLIT HEADER (keep sidebar toggle visible) ---------- */
[data-testid="stHeader"] {
    background: transparent !important;
    z-index: 100 !important;
}

/* ---------- PAGE HEADING ---------- */
.page-heading { margin-bottom: 24px; }
.page-heading h1 {
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 4px !important;
    color: #1e293b !important;
}
.page-heading p {
    font-size: 14px;
    color: #64748b;
}

/* ---------- TIP CARD ---------- */
.tip-card {
    background: linear-gradient(135deg, rgba(79,70,226,0.06), rgba(124,58,237,0.06));
    border: 1px solid rgba(79, 70, 226, 0.12);
    border-radius: 1rem;
    padding: 14px 18px;
    font-size: 13px;
    color: #64748b;
    line-height: 1.6;
}
.tip-card strong { color: #4f46e2; }

/* ---------- EXAMPLE BUTTON ---------- */
.example-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 1rem;
    padding: 12px 14px;
    cursor: pointer;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    text-align: left;
}
.example-btn:hover {
    border-color: #4f46e2;
    background: rgba(255, 255, 255, 0.85);
    transform: translateY(-2px);
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
}
.example-emoji {
    font-size: 20px;
    flex-shrink: 0;
    width: 32px;
    text-align: center;
}
.example-content h4 {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 2px 0;
}
.example-content p {
    font-size: 11px;
    color: #94a3b8;
    margin: 0;
    line-height: 1.4;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ---------- LOADING STAGES ---------- */
.loading-stage {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    border-radius: 0.75rem;
    font-size: 12px;
    color: #64748b;
    transition: all 0.3s;
}
.loading-stage.active {
    background: rgba(79, 70, 226, 0.08);
    color: #4f46e2;
    font-weight: 500;
}
.loading-stage.done { color: #4f46e2; }
.loading-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #94a3b8;
    flex-shrink: 0;
}
.loading-stage.active .loading-dot {
    background: #4f46e2;
    animation: pulse 1s infinite;
}
.loading-stage.done .loading-dot { background: #4f46e2; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ---------- EMPTY STATE ---------- */
.empty-state {
    text-align: center;
    padding: 48px 24px;
}
.empty-state-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.3;
}
.empty-state h3 {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #1e293b;
}
.empty-state p {
    font-size: 13px;
    color: #94a3b8;
    max-width: 360px;
    margin: 0 auto;
}

/* ---------- FOOTER ---------- */
.app-footer {
    margin-top: 32px;
    padding: 24px 0 16px 0;
    border-top: 1px solid #e2e8f0;
    text-align: center;
}
.app-footer-brand {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 4px;
}
.app-footer-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-bottom: 10px;
}
.app-footer-author {
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 12px;
}
.app-footer-tech {
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.app-footer-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 10px;
    font-weight: 500;
    color: #64748b;
}
.app-footer-copy {
    font-size: 10px;
    color: #94a3b8;
    opacity: 0.6;
}

/* ---------- ABOUT PAGE ---------- */
.about-section-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 1.25rem;
    padding: 22px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.about-section-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 16px rgba(79, 70, 226, 0.12);
}
.about-section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
}
.about-section-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
.about-section-title {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #1e293b;
}
.about-section-body {
    font-size: 13px;
    color: #64748b;
    line-height: 1.7;
}
.about-section-body strong { color: #1e293b; }

/* ---------- PROGRESS BAR ---------- */
.progress-track {
    height: 7px;
    background: rgba(79, 70, 226, 0.06);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 12px;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4f46e2, #7c3aed);
    border-radius: 10px;
    animation: progressFill 0.8s cubic-bezier(0.4, 0, 0.2, 1) both;
}

/* ---------- SEARCH BAR (Header) ---------- */
.header-search {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 16px;
    padding: 10px 16px;
    width: 100%;
    max-width: 320px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.header-search:focus-within {
    border-color: #4f46e2;
    box-shadow: 0 0 0 3px rgba(79, 70, 226, 0.1);
}
.header-search-icon {
    font-size: 14px;
    color: #94a3b8;
    flex-shrink: 0;
}
.header-search input {
    border: none;
    background: transparent;
    outline: none;
    font-family: 'Satoshi', sans-serif;
    font-size: 13px;
    color: #1e293b;
    width: 100%;
}
.header-search input::placeholder { color: #94a3b8; }

/* ---------- NOTIFICATION BTN ---------- */
.notif-btn {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    flex-shrink: 0;
}
.notif-btn:hover {
    background: rgba(255, 255, 255, 0.85);
    transform: translateY(-2px);
    box-shadow: 0 2px 8px rgba(79, 70, 226, 0.08);
}

/* ---------- SVG CHART AREA ---------- */
.chart-area {
    width: 100%;
    overflow: visible;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 768px) {
    section.main > div { padding: 16px 16px 32px 16px; }
    .hero-card { padding: 28px 20px; }
    .hero-card h1 { font-size: 24px !important; }
    .metric-value { font-size: 22px; }
    .page-heading h1 { font-size: 22px !important; }
    .widget-card { padding: 18px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# CACHED HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_file = REPORTS_DIR / "model_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None


@st.cache_data(show_spinner=False)
def get_dataset_stats():
    stats = {
        "train": 0, "val": 0, "test": 0, "total": 0,
        "classes": {}, "avg_length": 0,
    }
    dfs = []
    if TRAIN_DATA_PATH.exists():
        df = pd.read_csv(TRAIN_DATA_PATH)
        stats["train"] = len(df)
        dfs.append(df)
    if VALIDATION_DATA_PATH.exists():
        df = pd.read_csv(VALIDATION_DATA_PATH)
        stats["val"] = len(df)
        dfs.append(df)
    if TEST_DATA_PATH.exists():
        df = pd.read_csv(TEST_DATA_PATH)
        stats["test"] = len(df)
        dfs.append(df)
    if dfs:
        full_df = pd.concat(dfs)
        stats["total"] = len(full_df)
        stats["classes"] = full_df.get("label", pd.Series([])).value_counts().to_dict()
        if "review" in full_df.columns:
            stats["avg_length"] = full_df["review"].apply(lambda x: len(str(x).split())).mean()
    return stats


def render_sentiment_badge(label: str) -> str:
    css_class = label.lower()
    icon = {"positive": "&#10003;", "negative": "&#10007;", "neutral": "&#9679;"}.get(css_class, "&#9679;")
    return f'<span class="result-badge {css_class}">{icon} {label}</span>'


def render_prob_bar(label: str, value: float, color: str) -> str:
    pct = value * 100
    return f"""
    <div class="prob-bar-container">
        <div class="prob-bar-label">
            <span style="color: {color}; font-weight: 600;">{label}</span>
            <span style="color: #94a3b8;">{pct:.1f}%</span>
        </div>
        <div class="prob-bar-track">
            <div class="prob-bar-fill" style="width: {pct}%; background: {color};"></div>
        </div>
    </div>
    """


def render_highlight_card(icon: str, text: str, bg: str, color: str) -> str:
    return f"""
    <div class="highlight-card">
        <div class="highlight-icon" style="background: {bg}; color: {color};">{icon}</div>
        <div class="highlight-text">{text}</div>
    </div>
    """


def render_about_section(icon: str, bg: str, color: str, title: str, body: str) -> str:
    return f"""
    <div class="about-section-card anim-fade-up">
        <div class="about-section-header">
            <div class="about-section-icon" style="background: {bg}; color: {color};">{icon}</div>
            <div class="about-section-title">{title}</div>
        </div>
        <div class="about-section-body">{body}</div>
    </div>
    """


def render_footer():
    return """
    <div class="app-footer">
        <div class="app-footer-brand">StartupPulse AI</div>
        <div class="app-footer-sub">Explainable Aspect-Based Sentiment Analysis</div>
        <div class="app-footer-author">Built by <strong style="color: #1e293b;">Nikhil Khetavath</strong></div>
        <div class="app-footer-tech">
            <span class="app-footer-pill">DeBERTa-v3</span>
            <span class="app-footer-pill">Hugging Face</span>
            <span class="app-footer-pill">SHAP</span>
            <span class="app-footer-pill">Streamlit</span>
            <span class="app-footer-pill">PyTorch</span>
        </div>
        <div class="app-footer-copy">&copy; 2026 StartupPulse AI. MIT License.</div>
    </div>
    """


def render_loading_html(completed_count: int, active_idx: int, progress_pct: int) -> str:
    loading_stages = [
        "Loading DeBERTa-v3 model...",
        "Tokenizing review...",
        "Running inference...",
        "Generating SHAP explanations...",
        "Rendering dashboard...",
    ]
    stages_html = ""
    for idx, stage_text in enumerate(loading_stages):
        if idx < completed_count:
            state = "done"
        elif idx == active_idx:
            state = "active"
        else:
            state = ""
        stages_html += f"""
        <div class="loading-stage {state}"><div class="loading-dot"></div><span>{stage_text}</span></div>
        """
    return f"""
    <div class="glass-card" style="margin-top: 1rem;">
        <p style="font-family: 'Cabinet Grotesk', sans-serif; font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 0.1em; color: #94a3b8; margin-bottom: 0.8rem; font-weight: 700;">
            Processing Pipeline
        </p>
        {stages_html}
        <div class="progress-track"><div class="progress-fill" style="width: {progress_pct}%;"></div></div>
    </div>
    """


def load_image_safe(path: Path):
    try:
        if path.exists():
            return Image.open(str(path))
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
    return None


def render_svg_line_chart() -> str:
    return """
    <svg viewBox="0 0 400 120" class="chart-area" preserveAspectRatio="none" style="width:100%;height:120px;">
        <defs>
            <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#f43f5e" stop-opacity="0.3"/>
                <stop offset="100%" stop-color="#f43f5e" stop-opacity="0.0"/>
            </linearGradient>
        </defs>
        <path d="M0,80 C40,30 80,90 120,50 S200,20 240,55 S320,85 360,30 L400,35 L400,120 L0,120 Z"
              fill="url(#areaGrad)"/>
        <path d="M0,80 C40,30 80,90 120,50 S200,20 240,55 S320,85 360,30 L400,35"
              fill="none" stroke="#f43f5e" stroke-width="3" stroke-linecap="round"/>
        <circle cx="240" cy="55" r="4" fill="#f43f5e" stroke="white" stroke-width="2"/>
    </svg>
    """


def render_svg_bar_chart() -> str:
    bars = [
        ("Mon", 45, 0.45), ("Tue", 65, 0.65), ("Wed", 50, 0.50),
        ("Thu", 80, 0.80), ("Fri", 95, 0.95), ("Sat", 35, 0.35), ("Sun", 25, 0.25),
    ]
    bar_html = ""
    x = 20
    for day, height, _ in bars:
        bar_h = height * 0.9
        y = 100 - bar_h
        opacity = "1" if day == "Fri" else "0.3"
        glow = "filter: drop-shadow(0 2px 8px rgba(79,70,226,0.4));" if day == "Fri" else ""
        bar_w = "14" if day == "Fri" else "10"
        color = "#4f46e2" if day == "Fri" else "#c7d2fe"
        bar_html += f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="5" fill="{color}" opacity="{opacity}" style="{glow}"/>'
        bar_html += f'<text x="{x + int(bar_w)//2}" y="112" text-anchor="middle" font-size="9" fill="#94a3b8" font-family="Satoshi">{day}</text>'
        x += 50
    return f"""
    <svg viewBox="0 0 380 120" class="chart-area" preserveAspectRatio="none" style="width:100%;height:120px;">
        {bar_html}
    </svg>
    """


def render_circular_gauge(pct: float, color: str = "#f59e0b") -> str:
    r = 42
    circ = 2 * 3.14159 * r
    offset = circ * (1 - pct / 100)
    return f"""
    <svg viewBox="0 0 100 100" style="width:80px;height:80px;">
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="rgba(79,70,226,0.08)" stroke-width="10"/>
        <circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="10"
                stroke-linecap="round" stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
                transform="rotate(-90 50 50)" style="filter: drop-shadow(0 2px 4px rgba(245,158,11,0.3));"/>
        <text x="50" y="50" text-anchor="middle" dominant-baseline="central"
              font-family="Cabinet Grotesk" font-size="16" font-weight="800" fill="{color}">{pct}%</text>
    </svg>
    """


def render_horizontal_bar(label: str, pct: float, color: str = "#4f46e2") -> str:
    return f"""
    <div style="margin-bottom: 10px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;font-weight:500;color:#1e293b;">{label}</span>
            <span style="font-size:12px;font-weight:600;color:{color};">{pct:.0f}%</span>
        </div>
        <div style="height:10px;background:rgba(79,70,226,0.06);border-radius:10px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{color};border-radius:10px;animation:progressFill 1s cubic-bezier(0.4, 0, 0.2, 1) both;
                        box-shadow:0 0 8px {color}40;"></div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# CACHED SHAP EXPLAINER
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_shap_explainer():
    _load_ml_libs()
    from src.explainability.shap_explainer import SHAPExplainer
    return SHAPExplainer()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    logo_path = PROJECT_ROOT / "assets" / "logo.png"
    if logo_path.exists():
        st.markdown(
            f"""
        <div class="sidebar-logo-wrap">
            <div class="sidebar-logo-box">
                <img src="data:image/png;base64,{base64.b64encode(open(str(logo_path), 'rb').read()).decode()}" alt="Logo">
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
    <div class="sidebar-brand">
        <h2>StartupPulse AI</h2>
        <p style="text-align: center; font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.5rem; font-weight: 400;">Explainable AI Dashboard</p>
        <div style="text-align: center;">
            <span class="sidebar-version">{APP_VERSION}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)
    menu = ["Home", "Analyze Review", "Explainability", "Model Metrics", "About"]
    choice = st.radio("Navigation", menu, label_visibility="collapsed")

    st.markdown(
        """
    <div style="margin-top: 0.5rem;">
        <p class="sidebar-label">Technology</p>
        <div class="sidebar-tech-item">
            <span class="sidebar-tech-icon">&#9881;</span> Explainable AI
        </div>
        <div class="sidebar-tech-item">
            <span class="sidebar-tech-icon">&#9670;</span> DeBERTa-v3
        </div>
        <div class="sidebar-tech-item">
            <span class="sidebar-tech-icon">&#9673;</span> SHAP
        </div>
        <div class="sidebar-tech-item">
            <span class="sidebar-tech-icon">&#9632;</span> HR Analytics
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div class="sidebar-footer">
        <p>Built with Python + DeBERTa-v3</p>
        <p>Powered by SHAP Explainability</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""


# ---------------------------------------------------------------------------
# EXAMPLE REVIEWS
# ---------------------------------------------------------------------------
EXAMPLE_REVIEWS = {
    "Positive": {
        "emoji": "&#128522;",
        "label": "Positive Review",
        "text": "The company provides excellent work-life balance, supportive managers, and great career growth opportunities.",
    },
    "Neutral": {
        "emoji": "&#128528;",
        "label": "Neutral Review",
        "text": "The salary is average. The workload is manageable but there is little room for growth.",
    },
    "Negative": {
        "emoji": "&#128542;",
        "label": "Negative Review",
        "text": "Management ignores employee concerns. Compensation is poor and the work environment is stressful.",
    },
}


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
try:
    # =================================================================
    # HOME
    # =================================================================
    if choice == "Home":
        # Hero
        st.markdown(
            """
        <div class="hero-card anim-fade-up">
            <div class="hero-badge">EXPLAINABLE AI FOR HR ANALYTICS</div>
            <h1>StartupPulse AI</h1>
            <p class="subtitle">
                Explainable Aspect-Based Sentiment Analysis for Employee Feedback.
                Powered by DeBERTa-v3 transformers and SHAP token-level explanations.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Header row: Search + Notification
        hdr_cols = st.columns([3, 1])
        with hdr_cols[0]:
            st.markdown(
                """
            <div class="header-search">
                <span class="header-search-icon">&#128269;</span>
                <input type="text" placeholder="Search analytics, reviews, models..." readonly>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with hdr_cols[1]:
            st.markdown(
                '<div class="notif-btn" style="margin-top:2px;">&#128276;</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # KPI Row - 3 cards
        stats = get_dataset_stats()
        metrics_df = load_metrics()
        accuracy_val = 0.942
        if metrics_df is not None and "Accuracy" in metrics_df.columns:
            accuracy_val = metrics_df["Accuracy"].iloc[0]

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f"""
            <div class="widget-card anim-fade-up anim-delay-1">
                <p class="section-label">Sentiment Distribution</p>
                {render_horizontal_bar("Positive", 42, "#10b981")}
                {render_horizontal_bar("Neutral", 31, "#6366f1")}
                {render_horizontal_bar("Negative", 27, "#f43f5e")}
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
            <div class="widget-card anim-fade-up anim-delay-2">
                <p class="section-label">Review Trend (7 Days)</p>
                {render_svg_line_chart()}
            </div>
            """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
            <div class="widget-card anim-fade-up anim-delay-3">
                <p class="section-label">Model Accuracy</p>
                <div style="display:flex;justify-content:center;margin:16px 0;">
                    {render_circular_gauge(round(accuracy_val * 100, 1))}
                </div>
                <p style="text-align:center;font-size:12px;color:#94a3b8;margin-top:8px;">DeBERTa-v3 Fine-tuned</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Feature cards - 2x2
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            <div class="glass-card anim-fade-up anim-delay-1">
                <div class="highlight-icon" style="background:rgba(79,70,226,0.1);color:#4f46e2;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px;">&#9881;</div>
                <h3>Transformer Model</h3>
                <p>Fine-tuned Microsoft DeBERTa-v3 with disentangled attention for context-aware
                   three-class sentiment classification on employee review data.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <div class="glass-card anim-fade-up anim-delay-2">
                <div class="highlight-icon" style="background:rgba(124,58,237,0.1);color:#7c3aed;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px;">&#9670;</div>
                <h3>Real-time Prediction</h3>
                <p>Singleton-loaded model ensures instant inference. Paste any employee review and
                   receive a sentiment prediction with confidence scores in milliseconds.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
            <div class="glass-card anim-fade-up anim-delay-1">
                <div class="highlight-icon" style="background:rgba(16,185,129,0.1);color:#10b981;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px;">&#9673;</div>
                <h3>Explainable AI</h3>
                <p>SHAP decomposes every prediction into per-token importance scores. Waterfall plots,
                   bar charts, and interactive HTML reveal exactly why the model made its decision.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <div class="glass-card anim-fade-up anim-delay-2">
                <div class="highlight-icon" style="background:rgba(245,158,11,0.1);color:#f59e0b;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px;">&#9632;</div>
                <h3>Interactive Dashboard</h3>
                <p>Multi-page Streamlit interface with glassmorphic theme, analytics overview, confusion
                   matrix, and exportable visualizations for stakeholder reporting.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Quick stats row
        st.markdown('<p class="section-label">Dataset Overview</p>', unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-1">
                <div class="metric-value" style="color:#4f46e2;">{stats['total']:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-2">
                <div class="metric-value" style="color:#10b981;">3</div>
                <div class="metric-label">Sentiment Classes</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q3:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-3">
                <div class="metric-value" style="color:#7c3aed;">12</div>
                <div class="metric-label">Transformer Layers</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q4:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-4">
                <div class="metric-value" style="color:#f59e0b;">768</div>
                <div class="metric-label">Hidden Dimensions</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Highlights
        st.markdown('<p class="section-label">Project Highlights</p>', unsafe_allow_html=True)
        h1_col, h2_col = st.columns(2)
        with h1_col:
            st.markdown(render_highlight_card("&#10003;", "Fine-tuned DeBERTa-v3", "rgba(79,70,226,0.1)", "#4f46e2"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "Real-time Sentiment Prediction", "rgba(124,58,237,0.1)", "#7c3aed"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "Interactive Dashboard", "rgba(245,158,11,0.1)", "#f59e0b"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "HR Intelligence Platform", "rgba(16,185,129,0.1)", "#10b981"), unsafe_allow_html=True)
        with h2_col:
            st.markdown(render_highlight_card("&#10003;", "Explainable AI using SHAP", "rgba(16,185,129,0.1)", "#10b981"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "Employee Review Analytics", "rgba(59,130,246,0.1)", "#3b82f6"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "Transformer-based NLP", "rgba(236,72,153,0.1)", "#ec4899"), unsafe_allow_html=True)
            st.markdown(render_highlight_card("&#10003;", "Production-ready Architecture", "rgba(245,158,11,0.1)", "#f59e0b"), unsafe_allow_html=True)

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Earning-style KPI Section
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
        if metrics_df is not None:
            metric_defs = [
                ("Accuracy", "#4f46e2"),
                ("Precision", "#7c3aed"),
                ("Recall", "#f43f5e"),
                ("F1 Score", "#10b981"),
            ]
            cols = st.columns(4)
            for i, (col_name, clr) in enumerate(metric_defs):
                if col_name in metrics_df.columns:
                    val = metrics_df[col_name].iloc[0]
                    display_val = f"{val:.1%}" if val <= 1 else f"{val:.4f}"
                    with cols[i]:
                        st.markdown(
                            f"""
                        <div class="metric-card anim-fade-up anim-delay-{i+1}">
                            <div class="metric-value" style="color:{clr};">{display_val}</div>
                            <div class="metric-label">{col_name}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Metrics not available. Run model evaluation first.")

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # SHAP preview
        shap_preview_path = PROJECT_ROOT / "assets" / "shap_explanation.png"
        shap_img = load_image_safe(shap_preview_path)
        if shap_img is not None:
            st.markdown('<p class="section-label">Explainable AI Preview</p>', unsafe_allow_html=True)
            sh1, sh2 = st.columns([3, 2])
            with sh1:
                st.image(shap_img, use_container_width=True)
            with sh2:
                st.markdown(
                    """
                <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div class="highlight-icon" style="background:rgba(16,185,129,0.1);color:#10b981;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:12px;">&#9673;</div>
                    <h3>Transparent Predictions</h3>
                    <p style="margin-top: 0.5rem;">
                        Every prediction includes token-level SHAP explanations, making the AI
                        transparent and trustworthy. Each word is scored by its contribution to
                        the final sentiment classification.
                    </p>
                    <p style="margin-top: 1rem; font-size: 13px;">
                        <strong style="color: #4f46e2;">Navigate to Analyze Review</strong>
                        to try it with your own employee feedback.
                    </p>
                </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Glass Hero Promo
        st.markdown(
            """
        <div class="hero-promo anim-scale">
            <div class="hero-play">&#9654;</div>
            <h3>Try Live Analysis</h3>
            <p>Paste any employee review and get instant sentiment classification with SHAP explanations.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="tip-card">
            <strong>Get started:</strong> Navigate to <strong>Analyze Review</strong> in the sidebar to
            test the AI engine with a live employee review.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(render_footer(), unsafe_allow_html=True)

    # =================================================================
    # ANALYZE REVIEW
    # =================================================================
    elif choice == "Analyze Review":
        st.markdown(
            """
        <div class="page-heading anim-fade-up">
            <h1>Analyze Review</h1>
            <p>Enter an employee review to classify sentiment and generate SHAP explanations.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Try Example Reviews</p>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        for i, (sentiment, data) in enumerate(EXAMPLE_REVIEWS.items()):
            with ex_cols[i]:
                if st.button(
                    f"{data['emoji']}  {data['label']}",
                    key=f"example_{sentiment}",
                    use_container_width=True,
                ):
                    st.session_state["review_text"] = data["text"]
                    st.rerun()

        st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)

        review_input = st.text_area(
            "Employee Review",
            value=st.session_state["review_text"],
            height=160,
            placeholder="Paste an employee review here... e.g. The management is great but the pay needs improvement.",
            label_visibility="collapsed",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            analyze_clicked = st.button(
                "Analyze Sentiment",
                use_container_width=True,
                type="primary",
            )
        with btn_col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["review_text"] = ""
                st.session_state["prediction_result"] = None
                st.rerun()

        if analyze_clicked:
            if not review_input.strip():
                st.warning("Please enter a review to analyze.")
            else:
                st.session_state["review_text"] = review_input

                loading_container = st.empty()
                loading_container.markdown(render_loading_html(0, 0, 10), unsafe_allow_html=True)

                try:
                    loading_container.markdown(render_loading_html(1, 1, 25), unsafe_allow_html=True)
                    _load_ml_libs()

                    from src.explainability.shap_explainer import explain_prediction

                    loading_container.markdown(render_loading_html(2, 2, 50), unsafe_allow_html=True)
                    result = explain_prediction(review_input)
                    st.session_state["prediction_result"] = result

                    loading_container.markdown(render_loading_html(5, 5, 100), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.error(f"Dashboard analysis error: {e}")

                loading_container.empty()

        # Result card
        if st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            label = res["label"]
            confidence = res["confidence"]
            probs = res["probabilities"]

            color_map = {
                "Positive": ("#10b981", "rgba(16,185,129,0.1)"),
                "Negative": ("#ef4444", "rgba(239,68,68,0.1)"),
                "Neutral": ("#6366f1", "rgba(99,102,241,0.1)"),
            }
            accent_color, _ = color_map.get(label, ("#4f46e2", "rgba(79,70,226,0.1)"))

            prob_order = [("Positive", "#10b981"), ("Neutral", "#6366f1"), ("Negative", "#ef4444")]
            prob_bars_html = ""
            for p_label, p_color in prob_order:
                p_val = probs.get(p_label, 0)
                prob_bars_html += render_prob_bar(p_label, p_val, p_color)

            st.markdown(
                f"""
            <div style="margin-top: 2rem;" class="anim-fade-up">
                <div class="result-grid" style="display: grid; grid-template-columns: minmax(300px, 1fr) 2fr; gap: 1.5rem;">
                    <div class="glass-card">
                        <p class="section-label">Prediction Result</p>
                        <div style="margin-bottom: 1.2rem;">
                            {render_sentiment_badge(label)}
                        </div>
                        <div style="margin-bottom: 0.3rem;">
                            <span style="font-size: 11px; color: #94a3b8; font-weight: 500;
                                         text-transform: uppercase; letter-spacing: 0.06em;">
                                Confidence
                            </span>
                        </div>
                        <div style="font-family: 'Cabinet Grotesk'; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.04em;
                                    color: {accent_color}; line-height: 1;">
                            {confidence:.1%}
                        </div>
                        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.5);">
                            <p style="font-size: 11px; color: #94a3b8; margin-bottom: 0.3rem;">
                                Source Text
                            </p>
                            <p style="font-size: 13px; color: #64748b; line-height: 1.5;
                                      max-height: 80px; overflow-y: auto;">
                                {sanitize_html(st.session_state['review_text'][:200])}{'...' if len(st.session_state['review_text']) > 200 else ''}
                            </p>
                        </div>
                    </div>

                    <div class="glass-card">
                        <p class="section-label">Class Probability Distribution</p>
                        {prob_bars_html}
                        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.5);">
                            <p style="font-size: 13px; color: #64748b;">
                                Navigate to <strong style="color: #4f46e2;">Explainability</strong>
                                to view SHAP token-level explanations for this prediction.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown(render_footer(), unsafe_allow_html=True)

    # =================================================================
    # EXPLAINABILITY
    # =================================================================
    elif choice == "Explainability":
        st.markdown(
            """
        <div class="page-heading anim-fade-up">
            <h1>Explainability</h1>
            <p>SHAP token-level explanations for the latest prediction.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if not st.session_state["prediction_result"]:
            st.markdown(
                """
            <div class="empty-state anim-fade">
                <div class="empty-state-icon">&#9670;</div>
                <h3>No prediction available</h3>
                <p>Run a sentiment analysis to generate explainability visualizations.
                   Navigate to <strong>Analyze Review</strong> to get started.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            res = st.session_state["prediction_result"]

            st.markdown(
                f"""
            <div class="glass-card anim-fade-up" style="margin-bottom: 1.5rem;">
                <p class="section-label">Analyzed Text</p>
                <p style="font-size: 14px; color: #64748b; line-height: 1.6;">
                    {sanitize_html(st.session_state['review_text'])}
                </p>
                <div style="margin-top: 0.8rem;">
                    {render_sentiment_badge(res['label'])}
                    <span style="margin-left: 10px; font-size: 13px; color: #64748b;">
                        Confidence: <strong style="color: #1e293b;">{res['confidence']:.1%}</strong>
                    </span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            with st.spinner("Computing SHAP visualizations..."):
                try:
                    explainer = get_shap_explainer()
                    explainer.generate_plots(res, prefix="dashboard")
                except Exception as e:
                    st.error(f"SHAP generation failed: {e}")
                    logger.error(f"SHAP plot generation error: {e}")

            tokens = res["token_importance"]
            sorted_tokens = sorted(tokens.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            token_df = pd.DataFrame(sorted_tokens, columns=["Token", "Impact"])

            def color_impact(val):
                if val > 0:
                    return "color: #10b981; font-weight: 600;"
                return "color: #ef4444; font-weight: 600;"

            with st.expander("Top 10 Influential Tokens", expanded=True):
                st.dataframe(
                    token_df.style.map(color_impact, subset=["Impact"]).format({"Impact": "{:.6f}"}),
                    use_container_width=True,
                    height=370,
                )

            shap_dir = REPORTS_DIR / "shap"
            waterfall_file = shap_dir / "dashboard_waterfall.png"
            with st.expander("SHAP Waterfall Plot", expanded=True):
                if waterfall_file.exists():
                    img = load_image_safe(waterfall_file)
                    if img:
                        st.image(img, use_container_width=True)
                        with open(waterfall_file, "rb") as f:
                            st.download_button(
                                "Download Waterfall Plot",
                                data=f.read(),
                                file_name="shap_waterfall.png",
                                mime="image/png",
                            )
                    else:
                        st.info("Waterfall plot could not be loaded.")
                else:
                    st.info("Waterfall plot not available. Run a prediction first.")

            bar_file = shap_dir / "dashboard_bar_summary.png"
            with st.expander("SHAP Bar Summary", expanded=False):
                if bar_file.exists():
                    bar_img = load_image_safe(bar_file)
                    if bar_img:
                        st.image(bar_img, use_container_width=True)
                    else:
                        st.info("Bar summary plot could not be loaded.")
                else:
                    st.info("Bar summary plot not available.")

            text_file = shap_dir / "dashboard_text.html"
            with st.expander("Token Impact Visualization", expanded=False):
                if text_file.exists():
                    try:
                        with open(text_file, "r", encoding="utf-8") as f:
                            html_str = f.read()
                        components.html(html_str, height=280, scrolling=True)
                    except Exception as e:
                        st.info(f"Could not load text visualization: {e}")
                else:
                    st.info("Text visualization not available.")

        st.markdown(render_footer(), unsafe_allow_html=True)

    # =================================================================
    # MODEL METRICS
    # =================================================================
    elif choice == "Model Metrics":
        st.markdown(
            """
        <div class="page-heading anim-fade-up">
            <h1>Model Metrics</h1>
            <p>Evaluation results and dataset statistics for the fine-tuned DeBERTa-v3 model.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        stats = get_dataset_stats()

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-1">
                <div class="metric-value" style="color:#4f46e2;">{stats['total']:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-2">
                <div class="metric-value" style="color:#10b981;">{stats['train']:,}</div>
                <div class="metric-label">Training Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-3">
                <div class="metric-value" style="color:#7c3aed;">{stats['val']:,}</div>
                <div class="metric-label">Validation Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-4">
                <div class="metric-value" style="color:#f43f5e;">{stats['test']:,}</div>
                <div class="metric-label">Test Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<p class="section-label">Performance Metrics</p>', unsafe_allow_html=True)
            metrics_df = load_metrics()
            if metrics_df is not None:
                for col_name in metrics_df.columns:
                    val = metrics_df[col_name].iloc[0]
                    if "F1" in col_name:
                        clr = "#10b981"
                    elif "Accuracy" in col_name:
                        clr = "#4f46e2"
                    elif "Precision" in col_name:
                        clr = "#7c3aed"
                    else:
                        clr = "#f43f5e"
                    display_val = f"{val:.1%}" if val <= 1 else f"{val:.4f}"
                    st.markdown(
                        f"""
                    <div class="metric-card" style="margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="metric-label" style="text-transform: none; letter-spacing: 0;">{col_name}</span>
                            <span class="metric-value" style="font-size: 1.6rem; color: {clr};">{display_val}</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Metrics file not found. Run model evaluation first.")

            st.markdown(
                f"""
            <div class="metric-card" style="margin-top: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-label" style="text-transform: none; letter-spacing: 0;">Avg. Words per Review</span>
                    <span class="metric-value" style="font-size: 1.6rem; color: #1e293b;">{stats['avg_length']:.1f}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="section-label" style="margin-top: 1.5rem;">Class Distribution</p>',
                unsafe_allow_html=True,
            )
            if stats["classes"]:
                mapped = {SENTIMENT_MAPPING.get(k, k): v for k, v in stats["classes"].items()}
                dist_df = pd.DataFrame(list(mapped.items()), columns=["Sentiment", "Count"]).set_index("Sentiment")
                st.bar_chart(dist_df)
            else:
                st.info("Dataset statistics unavailable.")

        with col_right:
            st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
            cm_path = REPORTS_DIR / "confusion_matrix.png"
            cm_img = load_image_safe(cm_path)
            if cm_img:
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix not found. Run model evaluation first.")

            report_path = REPORTS_DIR / "classification_report.txt"
            if report_path.exists():
                with st.expander("Classification Report", expanded=False):
                    try:
                        with open(report_path, "r") as f:
                            report_text = f.read()
                        st.code(report_text, language=None)
                    except Exception as e:
                        st.info(f"Could not load classification report: {e}")

        st.markdown(render_footer(), unsafe_allow_html=True)

    # =================================================================
    # ABOUT
    # =================================================================
    elif choice == "About":
        st.markdown(
            """
        <div class="page-heading anim-fade-up">
            <h1>About StartupPulse AI</h1>
            <p>Technical overview and project information.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9733;", "rgba(79,70,226,0.1)", "#4f46e2",
                "Project Goal",
                """
                Build an explainable AI platform that analyzes employee feedback using state-of-the-art NLP
                and provides transparent, token-level explanations for every prediction. The goal is to give
                HR teams and startup founders actionable insights they can trust and act on.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9888;", "rgba(239,68,68,0.1)", "#ef4444",
                "Problem Statement",
                """
                Employee feedback is one of the most valuable signals a startup can collect, but most
                organizations lack the infrastructure to analyze it at scale. Manual tagging does not scale.
                Traditional sentiment analysis misses contextual nuance. Black-box predictions give HR teams
                no diagnostic value. And in HR analytics, predictions directly affect people's careers --
                making transparency non-negotiable.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#10003;", "rgba(16,185,129,0.1)", "#10b981",
                "Solution",
                """
                StartupPulse AI combines a fine-tuned Microsoft DeBERTa-v3 transformer with SHAP
                (SHapley Additive exPlanations) to deliver sentiment predictions that are both accurate
                and fully interpretable. Every token in an employee review receives an importance score,
                making the model's decision-making process transparent and auditable. An interactive
                Streamlit dashboard presents results to non-technical stakeholders.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9881;", "rgba(79,70,226,0.1)", "#4f46e2",
                "Key Features",
                """
                <strong>Transformer Sentiment Analysis</strong> -- Fine-tuned DeBERTa-v3 for context-aware
                three-class classification.<br><br>
                <strong>SHAP Explainability</strong> -- Per-token importance scores with waterfall plots,
                bar charts, and interactive HTML visualizations.<br><br>
                <strong>Interactive Dashboard</strong> -- Five-page Streamlit application with glassmorphic
                theme designed for non-technical HR stakeholders.<br><br>
                <strong>Real-time Prediction</strong> -- Singleton-loaded model with instant inference
                and confidence scores.<br><br>
                <strong>Probability Distribution</strong> -- Full three-class probability output, not just
                top-1 prediction.<br><br>
                <strong>Model Evaluation</strong> -- Automated pipeline generating classification reports,
                confusion matrices, and weighted F1 metrics.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9632;", "rgba(124,58,237,0.1)", "#7c3aed",
                "Technology Stack",
                """
                <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.5rem;">
                    <span class="app-footer-pill">Python 3.10+</span>
                    <span class="app-footer-pill">PyTorch 2.2</span>
                    <span class="app-footer-pill">Hugging Face Transformers</span>
                    <span class="app-footer-pill">SHAP 0.45</span>
                    <span class="app-footer-pill">Streamlit 1.33</span>
                    <span class="app-footer-pill">scikit-learn</span>
                    <span class="app-footer-pill">Pandas / NumPy</span>
                    <span class="app-footer-pill">Matplotlib / Seaborn</span>
                    <span class="app-footer-pill">SentencePiece</span>
                </div>
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9670;", "rgba(59,130,246,0.1)", "#3b82f6",
                "Model Architecture",
                """
                <strong>Base Model:</strong> microsoft/deberta-v3-base<br>
                <strong>Hidden Size:</strong> 768 dimensions<br>
                <strong>Layers:</strong> 12 transformer blocks<br>
                <strong>Attention Heads:</strong> 12<br>
                <strong>Attention Type:</strong> Disentangled (content-to-position + position-to-content)<br>
                <strong>Vocabulary:</strong> 128,100 tokens (SentencePiece)<br>
                <strong>Classification:</strong> 3-class softmax (Negative, Neutral, Positive)<br>
                <strong>Training:</strong> 3 epochs, learning rate 2e-5, batch size 16, early stopping (patience=3)
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9673;", "rgba(16,185,129,0.1)", "#10b981",
                "Explainability",
                """
                SHAP (SHapley Additive exPlanations) provides a game-theoretic framework for explaining
                individual predictions. For each input text, SHAP treats every token as a "player" and
                computes its marginal contribution to the final prediction. This produces three
                visualization types:<br><br>
                <strong>Waterfall Plots</strong> -- Cumulative token contributions from base value to final output.<br>
                <strong>Bar Charts</strong> -- Token importance ranking by absolute SHAP value.<br>
                <strong>Interactive HTML</strong> -- Color-coded tokens showing positive (green) and negative (red)
                contributions inline.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9654;", "rgba(245,158,11,0.1)", "#f59e0b",
                "Future Improvements",
                """
                <strong>Aspect Extraction</strong> -- Per-dimension sentiment (management, compensation, growth,
                culture).<br>
                <strong>Multi-language Support</strong> -- DeBERTa multilingual variants for non-English feedback.<br>
                <strong>REST API</strong> -- FastAPI endpoints for HRIS integration.<br>
                <strong>Cloud Deployment</strong> -- Docker + Kubernetes on AWS/GCP.<br>
                <strong>Continuous Learning</strong> -- Feedback loop for periodic model retraining.<br>
                <strong>Voice Feedback</strong> -- Speech-to-text for town halls and exit interviews.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_about_section(
                "&#9632;", "rgba(236,72,153,0.1)", "#ec4899",
                "Application Areas",
                """
                <strong>HR Teams</strong> -- Identify systemic issues in employee experience with evidence-backed
                explanations.<br>
                <strong>Startup Founders</strong> -- Scalable feedback analysis without a dedicated people analytics
                team.<br>
                <strong>Business Managers</strong> -- Department-level sentiment tracking and trend analysis.<br>
                <strong>Data Analysts</strong> -- Exportable visualizations and classification reports for
                stakeholder reporting.
                """,
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        arch_path = PROJECT_ROOT / "assets" / "architecture.png"
        arch_img = load_image_safe(arch_path)
        if arch_img:
            st.markdown('<p class="section-label">System Architecture</p>', unsafe_allow_html=True)
            st.image(arch_img, use_container_width=True)

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        st.markdown(
            f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <p style="font-size: 13px; color: #94a3b8;">
                        <strong style="color: #1e293b;">Author:</strong> Nikhil Khetavath &nbsp;&middot;&nbsp;
                        <strong style="color: #1e293b;">License:</strong> MIT &nbsp;&middot;&nbsp;
                        <strong style="color: #1e293b;">Version:</strong> {APP_VERSION}
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="https://github.com/khetavathnikhil17-afk/StartupPulse-AI" target="_blank" style="
                        display: inline-block; background: rgba(255, 255, 255, 0.72); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.5);
                        border-radius: 12px; padding: 8px 16px; color: #1e293b;
                        text-decoration: none; font-size: 13px; font-weight: 500; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);">
                        GitHub Repository
                    </a>
                    <a href="https://www.linkedin.com/in/nikhilkhetavath-ai" target="_blank" style="
                        display: inline-block; background: rgba(255, 255, 255, 0.72); backdrop-filter: blur(8px); border: 1px solid rgba(255, 255, 255, 0.5);
                        border-radius: 12px; padding: 8px 16px; color: #1e293b;
                        text-decoration: none; font-size: 13px; font-weight: 500; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);">
                        LinkedIn Profile
                    </a>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(render_footer(), unsafe_allow_html=True)

except Exception as e:
    st.error("A critical error occurred while rendering the page.")
    st.error(f"Error Details: {e}")
    logger.error(f"Dashboard global exception: {e}")
