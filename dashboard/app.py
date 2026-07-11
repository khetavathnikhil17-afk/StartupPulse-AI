import streamlit as st
import pandas as pd
import sys
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
# LAZY ML IMPORTS — loaded only when needed (Analyze Review / Explainability)
# ---------------------------------------------------------------------------
_ml_loaded = False
_torch = None
_shap = None
_AutoModelForSequenceClassification = None
_AutoTokenizer = None


def _load_ml_libs():
    """Lazily import heavy ML libraries only when a prediction is requested."""
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
    """Sanitize text for safe HTML injection."""
    return html.escape(str(text))


st.set_page_config(
    page_title="StartupPulse AI",
    page_icon=str(PROJECT_ROOT / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL PREMIUM CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #09090b;
    --bg-secondary: #18181b;
    --bg-card: rgba(24, 24, 27, 0.7);
    --bg-card-hover: rgba(39, 39, 42, 0.8);
    --border: rgba(63, 63, 70, 0.5);
    --border-subtle: rgba(63, 63, 70, 0.3);
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --accent-glow: rgba(99, 102, 241, 0.25);
    --positive: #22c55e;
    --positive-bg: rgba(34, 197, 94, 0.1);
    --negative: #ef4444;
    --negative-bg: rgba(239, 68, 68, 0.1);
    --neutral: #3b82f6;
    --neutral-bg: rgba(59, 130, 246, 0.1);
    --radius: 16px;
    --radius-sm: 10px;
    --radius-xs: 6px;
    --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    --shadow-lg: 0 8px 40px rgba(0, 0, 0, 0.4);
}

/* ---------- BASE RESET ---------- */
.stApp, .main, [data-testid="stAppViewContainer"],
[data-testid="stHeader"], section.main {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

section.main > div { padding: 2rem 2.5rem 4rem 2.5rem; }

/* ---------- HIDE DEFAULT STREAMLIT ELEMENTS ---------- */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---------- STREAMLIT DARK THEME OVERRIDES ---------- */
[data-testid="stSelectbox"] div div div,
[data-testid="stMultiSelect"] div div div {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--bg-secondary) !important;
    border-color: var(--border) !important;
}
[data-baseweb="menu"] {
    background: var(--bg-secondary) !important;
}
[data-baseweb="option"] {
    background: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="option"]:hover, [data-baseweb="option"]:focus,
[data-baseweb="option"][aria-selected="true"] {
    background: var(--bg-card-hover) !important;
    color: var(--accent-light) !important;
}
.stRadio label, .stRadio div[role="radiogroup"] label {
    color: var(--text-primary) !important;
}
.stCheckbox label, .stCheckbox span {
    color: var(--text-primary) !important;
}
[data-testid="stVerticalBlock"] p {
    color: var(--text-secondary) !important;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f12 0%, #131316 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
    padding: 1.5rem 1rem;
}
[data-testid="stSidebar"] [data-testid="stImageContainer"] {
    display: flex;
    justify-content: center;
    margin-bottom: 0;
}
[data-testid="stSidebar"] [data-testid="stImageContainer"] img {
    border-radius: 50%;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
.sidebar-brand {
    padding: 1.2rem 0.8rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
}
.sidebar-brand h2 {
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}
.sidebar-version {
    display: inline-block;
    background: var(--accent-glow);
    color: var(--accent-light);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 6px;
    letter-spacing: 0.03em;
}
.sidebar-label {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    margin-bottom: 0.3rem !important;
}
.sidebar-tech-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    border-radius: var(--radius-xs);
    font-size: 0.78rem;
    color: var(--text-secondary);
    transition: background 0.2s;
}
.sidebar-tech-item:hover {
    background: var(--bg-card);
}
.sidebar-tech-icon {
    width: 20px;
    text-align: center;
    font-size: 0.75rem;
    opacity: 0.6;
}
.sidebar-footer {
    padding-top: 1rem;
    border-top: 1px solid var(--border-subtle);
    margin-top: auto;
}
.sidebar-footer p {
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
}

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.8rem;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--bg-card);
    border-color: var(--border);
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: var(--accent-glow) !important;
    border-color: var(--accent) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] span {
    color: var(--accent-light) !important;
    font-weight: 600 !important;
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.anim-fade-up { animation: fadeInUp 0.5s ease both; }
.anim-fade    { animation: fadeIn 0.4s ease both; }
.anim-slide   { animation: slideIn 0.4s ease both; }
.anim-delay-1 { animation-delay: 0.05s; }
.anim-delay-2 { animation-delay: 0.10s; }
.anim-delay-3 { animation-delay: 0.15s; }
.anim-delay-4 { animation-delay: 0.20s; }

/* ---------- HERO CARD ---------- */
.hero-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(168,85,247,0.08) 50%, rgba(99,102,241,0.04) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: var(--radius);
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-card h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.6rem !important;
}
.hero-card .subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    line-height: 1.6;
    max-width: 640px;
}
.hero-badge {
    display: inline-block;
    background: var(--accent-glow);
    color: var(--accent-light);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
}

/* ---------- GLASS CARD ---------- */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    transition: all 0.25s ease;
    box-shadow: var(--shadow);
}
.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: var(--shadow-lg), 0 0 30px var(--accent-glow);
    transform: translateY(-2px);
}
.glass-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 1rem;
}
.glass-card h3 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.5rem !important;
}
.glass-card p {
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.6;
}

/* ---------- MINI HIGHLIGHT CARD ---------- */
.highlight-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.25s ease;
}
.highlight-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: var(--bg-card-hover);
}
.highlight-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.highlight-text {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.3;
}

/* ---------- METRIC CARD ---------- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-lg), 0 0 20px var(--accent-glow);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ---------- RESULT BADGE ---------- */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.01em;
}
.result-badge.positive {
    background: var(--positive-bg);
    color: var(--positive);
    border: 1px solid rgba(34, 197, 94, 0.25);
}
.result-badge.negative {
    background: var(--negative-bg);
    color: var(--negative);
    border: 1px solid rgba(239, 68, 68, 0.25);
}
.result-badge.neutral {
    background: var(--neutral-bg);
    color: var(--neutral);
    border: 1px solid rgba(59, 130, 246, 0.25);
}

/* ---------- SECTION HEADINGS ---------- */
.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.section-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin-bottom: 1.8rem;
}

/* ---------- DIVIDER ---------- */
.premium-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

/* ---------- PROBABILITY BAR ---------- */
.prob-bar-container { margin-bottom: 0.8rem; }
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 4px;
}
.prob-bar-track {
    height: 8px;
    background: rgba(63, 63, 70, 0.4);
    border-radius: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.6s ease;
}

/* ---------- EXPANDER ---------- */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; }

/* ---------- BUTTONS ---------- */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em;
    transition: all 0.2s !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}
.stButton > button:hover {
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.4) !important;
    transform: translateY(-1px);
}
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    background: var(--bg-card-hover) !important;
}

/* ---------- TEXT AREA ---------- */
.stTextArea textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.92rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

/* ---------- DATAFRAME ---------- */
.stDataFrame {
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ---------- SPINNER ---------- */
.stSpinner > div {
    border-color: var(--accent) transparent transparent transparent !important;
}

/* ---------- ALERT ---------- */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border-left-width: 3px !important;
}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-xs) !important;
    font-weight: 500;
    font-size: 0.88rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    border-bottom: none !important;
}

/* ---------- LINK BUTTON ---------- */
a[href] { color: var(--accent-light) !important; }

/* ---------- SCROLLBAR ---------- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* ---------- PAGE HEADING ---------- */
.page-heading { margin-bottom: 2rem; }
.page-heading h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.3rem !important;
}
.page-heading p {
    font-size: 0.95rem;
    color: var(--text-secondary);
}

/* ---------- TIP CARD ---------- */
.tip-card {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.tip-card strong { color: var(--accent-light); }

/* ---------- EXAMPLE BUTTON ---------- */
.example-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 0.75rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
}
.example-btn:hover {
    border-color: var(--accent);
    background: var(--bg-card-hover);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.example-emoji {
    font-size: 1.5rem;
    flex-shrink: 0;
    width: 36px;
    text-align: center;
}
.example-content h4 {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 2px 0;
}
.example-content p {
    font-size: 0.78rem;
    color: var(--text-muted);
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
    border-radius: var(--radius-xs);
    font-size: 0.85rem;
    color: var(--text-secondary);
    transition: all 0.3s;
}
.loading-stage.active {
    background: var(--accent-glow);
    color: var(--accent-light);
    font-weight: 500;
}
.loading-stage.done {
    color: var(--positive);
}
.loading-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
}
.loading-stage.active .loading-dot {
    background: var(--accent-light);
    animation: pulse 1s infinite;
}
.loading-stage.done .loading-dot {
    background: var(--positive);
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.4); }
}

/* ---------- EMPTY STATE ---------- */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.4;
}
.empty-state h3 {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.empty-state p {
    font-size: 0.9rem;
    color: var(--text-muted);
    max-width: 360px;
    margin: 0 auto;
}

/* ---------- FOOTER ---------- */
.app-footer {
    margin-top: 3rem;
    padding: 2.5rem 0 1.5rem 0;
    border-top: 1px solid var(--border);
    text-align: center;
}
.app-footer-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.app-footer-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    margin-bottom: 0.8rem;
}
.app-footer-author {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}
.app-footer-tech {
    display: flex;
    justify-content: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.app-footer-pill {
    display: inline-block;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.app-footer-copy {
    font-size: 0.72rem;
    color: var(--text-muted);
    opacity: 0.6;
}

/* ---------- ABOUT PAGE ---------- */
.about-section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
}
.about-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.about-section-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.about-section-title {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.about-section-body {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.7;
}
.about-section-body strong { color: var(--text-primary); }

/* ---------- PROGRESS BAR (loading) ---------- */
.progress-track {
    height: 3px;
    background: rgba(63, 63, 70, 0.3);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 12px;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), #7c3aed);
    border-radius: 10px;
    transition: width 0.5s ease;
}

/* ---------- SECTION LABEL ---------- */
.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 1rem;
    font-weight: 600;
}

/* ---------- RESPONSIVE ---------- */
@media (max-width: 768px) {
    section.main > div { padding: 1rem 1rem 3rem 1rem; }
    .hero-card { padding: 2rem 1.5rem; }
    .hero-card h1 { font-size: 1.8rem !important; }
    .hero-card .subtitle { font-size: 1rem; }
    .glass-card { padding: 1.2rem; }
    .metric-value { font-size: 1.6rem; }
    .page-heading h1 { font-size: 1.5rem !important; }
    .result-grid { grid-template-columns: 1fr !important; }
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
            <span style="color: var(--text-secondary);">{pct:.1f}%</span>
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
        <div class="app-footer-author">Built by <strong style="color: var(--text-primary);">Nikhil Khetavath</strong></div>
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
    """Render loading stages HTML (defined at module level, not per-click)."""
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
        <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                  color: var(--text-muted); margin-bottom: 0.8rem; font-weight: 600;">
            Processing Pipeline
        </p>
        {stages_html}
        <div class="progress-track"><div class="progress-fill" style="width: {progress_pct}%;"></div></div>
    </div>
    """


def load_image_safe(path: Path):
    """Load an image with error handling, returns None on failure."""
    try:
        if path.exists():
            return Image.open(str(path))
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
    return None


# ---------------------------------------------------------------------------
# CACHED SHAP EXPLAINER (loaded once, reused across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_shap_explainer():
    """Return a cached SHAPExplainer singleton."""
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
            """
        <div style="text-align: center; padding: 0.5rem 0 0.2rem 0; margin-bottom: 0.3rem;">
        """,
            unsafe_allow_html=True,
        )
        st.image(str(logo_path), width=140)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
    <div class="sidebar-brand">
        <h2 style="text-align: center; margin-bottom: 0.15rem;">StartupPulse AI</h2>
        <p style="text-align: center; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.5rem; font-weight: 400;">Explainable AI Dashboard</p>
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

    # Tech stack badges
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

        # Feature cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
            <div class="glass-card anim-fade-up anim-delay-1">
                <div class="glass-card-icon" style="background: rgba(99,102,241,0.12); color: #818cf8;">&#9881;</div>
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
                <div class="glass-card-icon" style="background: rgba(168,85,247,0.12); color: #c084fc;">&#9670;</div>
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
                <div class="glass-card-icon" style="background: rgba(34,197,94,0.12); color: #4ade80;">&#9673;</div>
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
                <div class="glass-card-icon" style="background: rgba(251,146,60,0.12); color: #fb923c;">&#9632;</div>
                <h3>Interactive Dashboard</h3>
                <p>Multi-page Streamlit interface with dark theme, analytics overview, confusion
                   matrix, and exportable visualizations for stakeholder reporting.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Quick stats
        stats = get_dataset_stats()
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-1">
                <div class="metric-value" style="color: var(--accent-light);">{stats['total']:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-2">
                <div class="metric-value" style="color: var(--positive);">3</div>
                <div class="metric-label">Sentiment Classes</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q3:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-3">
                <div class="metric-value" style="color: #c084fc;">12</div>
                <div class="metric-label">Transformer Layers</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with q4:
            st.markdown(
                """
            <div class="metric-card anim-fade-up anim-delay-4">
                <div class="metric-value" style="color: #fb923c;">768</div>
                <div class="metric-label">Hidden Dimensions</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Project highlights
        st.markdown('<p class="section-label">Project Highlights</p>', unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        with h1:
            st.markdown(
                render_highlight_card("&#10003;", "Fine-tuned DeBERTa-v3", "rgba(99,102,241,0.12)", "#818cf8"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "Real-time Sentiment Prediction", "rgba(168,85,247,0.12)", "#c084fc"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "Interactive Dashboard", "rgba(251,146,60,0.12)", "#fb923c"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "HR Intelligence Platform", "rgba(34,197,94,0.12)", "#4ade80"),
                unsafe_allow_html=True,
            )
        with h2:
            st.markdown(
                render_highlight_card("&#10003;", "Explainable AI using SHAP", "rgba(34,197,94,0.12)", "#4ade80"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "Employee Review Analytics", "rgba(59,130,246,0.12)", "#60a5fa"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "Transformer-based NLP", "rgba(244,114,182,0.12)", "#f472b6"),
                unsafe_allow_html=True,
            )
            st.markdown(
                render_highlight_card("&#10003;", "Production-ready Architecture", "rgba(251,191,36,0.12)", "#fbbf24"),
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Live Model Performance
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)

        metrics_df = load_metrics()
        if metrics_df is not None:
            metric_defs = [
                ("Accuracy", "var(--accent-light)"),
                ("Precision", "#c084fc"),
                ("Recall", "#fb923c"),
                ("F1 Score", "var(--positive)"),
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
                            <div class="metric-value" style="color: {clr};">{display_val}</div>
                            <div class="metric-label">{col_name}</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
        else:
            st.info("Metrics not available. Run model evaluation first.")

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # SHAP preview on home
        shap_preview_path = PROJECT_ROOT / "assets" / "shap_explanation.png"
        shap_img = load_image_safe(shap_preview_path)
        if shap_img is not None:
            st.markdown('<p class="section-label">Explainable AI</p>', unsafe_allow_html=True)
            sh1, sh2 = st.columns([3, 2])
            with sh1:
                st.image(shap_img, use_container_width=True)
            with sh2:
                st.markdown(
                    """
                <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div class="glass-card-icon" style="background: rgba(34,197,94,0.12); color: #4ade80;">&#9673;</div>
                    <h3>Transparent Predictions</h3>
                    <p style="margin-top: 0.5rem;">
                        Every prediction includes token-level SHAP explanations, making the AI
                        transparent and trustworthy. Each word is scored by its contribution to
                        the final sentiment classification.
                    </p>
                    <p style="margin-top: 1rem; font-size: 0.82rem;">
                        <strong style="color: var(--accent-light);">Navigate to Analyze Review</strong>
                        to try it with your own employee feedback.
                    </p>
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

        # Footer
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

        # Example reviews
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

        # Text area
        review_input = st.text_area(
            "Employee Review",
            value=st.session_state["review_text"],
            height=160,
            placeholder="Paste an employee review here... e.g. The management is great but the pay needs improvement.",
            label_visibility="collapsed",
        )

        btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 4])
        with btn_col1:
            analyze_clicked = st.button("Analyze Sentiment", use_container_width=True)
        with btn_col2:
            if st.button("Clear", use_container_width=True):
                st.session_state["review_text"] = ""
                st.session_state["prediction_result"] = None
                st.rerun()

        # Analysis pipeline
        if analyze_clicked:
            if not review_input.strip():
                st.warning("Please enter a review to analyze.")
            else:
                st.session_state["review_text"] = review_input

                loading_container = st.empty()
                loading_container.markdown(render_loading_html(0, 0, 0), unsafe_allow_html=True)

                try:
                    from src.explainability.shap_explainer import explain_prediction

                    loading_container.markdown(render_loading_html(1, 1, 30), unsafe_allow_html=True)
                    result = explain_prediction(review_input)
                    st.session_state["prediction_result"] = result

                    loading_container.markdown(render_loading_html(5, 5, 100), unsafe_allow_html=True)
                    loading_container.empty()

                except Exception as e:
                    loading_container.empty()
                    st.error(f"Analysis failed: {e}")
                    logger.error(f"Dashboard analysis error: {e}")

        # ---- Result card ----
        if st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            label = res["label"]
            confidence = res["confidence"]
            probs = res["probabilities"]

            color_map = {
                "Positive": ("var(--positive)", "var(--positive-bg)"),
                "Negative": ("var(--negative)", "var(--negative-bg)"),
                "Neutral": ("var(--neutral)", "var(--neutral-bg)"),
            }
            accent_color, _ = color_map.get(label, ("var(--accent)", "var(--accent-glow)"))

            prob_order = [("Positive", "#22c55e"), ("Neutral", "#3b82f6"), ("Negative", "#ef4444")]
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
                            <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 500;
                                         text-transform: uppercase; letter-spacing: 0.06em;">
                                Confidence
                            </span>
                        </div>
                        <div style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em;
                                    color: {accent_color}; line-height: 1;">
                            {confidence:.1%}
                        </div>
                        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                            <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.3rem;">
                                Source Text
                            </p>
                            <p style="font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;
                                      max-height: 80px; overflow-y: auto;">
                                {sanitize_html(st.session_state['review_text'][:200])}{'...' if len(st.session_state['review_text']) > 200 else ''}
                            </p>
                        </div>
                    </div>

                    <div class="glass-card">
                        <p class="section-label">Class Probability Distribution</p>
                        {prob_bars_html}
                        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--border);">
                            <p style="font-size: 0.78rem; color: var(--text-muted);">
                                Navigate to <strong style="color: var(--accent-light);">Explainability</strong>
                                to view SHAP token-level explanations for this prediction.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Footer
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

            # Source text banner
            st.markdown(
                f"""
            <div class="glass-card anim-fade-up" style="margin-bottom: 1.5rem;">
                <p class="section-label">Analyzed Text</p>
                <p style="font-size: 0.92rem; color: var(--text-secondary); line-height: 1.6;">
                    {sanitize_html(st.session_state['review_text'])}
                </p>
                <div style="margin-top: 0.8rem;">
                    {render_sentiment_badge(res['label'])}
                    <span style="margin-left: 10px; font-size: 0.85rem; color: var(--text-secondary);">
                        Confidence: <strong style="color: var(--text-primary);">{res['confidence']:.1%}</strong>
                    </span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Generate SHAP plots
            with st.spinner("Computing SHAP visualizations..."):
                try:
                    explainer = get_shap_explainer()
                    explainer.generate_plots(res, prefix="dashboard")
                except Exception as e:
                    st.error(f"SHAP generation failed: {e}")
                    logger.error(f"SHAP plot generation error: {e}")

            # Top tokens table
            tokens = res["token_importance"]
            sorted_tokens = sorted(tokens.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            token_df = pd.DataFrame(sorted_tokens, columns=["Token", "Impact"])

            def color_impact(val):
                if val > 0:
                    return "color: #22c55e; font-weight: 600;"
                return "color: #ef4444; font-weight: 600;"

            with st.expander("Top 10 Influential Tokens", expanded=True):
                st.dataframe(
                    token_df.style.map(color_impact, subset=["Impact"]).format({"Impact": "{:.6f}"}),
                    use_container_width=True,
                    height=370,
                )

            # Waterfall plot
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

            # Bar plot
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

            # Text HTML
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

        # Footer
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

        # Stat cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-1">
                <div class="metric-value" style="color: var(--accent-light);">{stats['total']:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s2:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-2">
                <div class="metric-value" style="color: var(--positive);">{stats['train']:,}</div>
                <div class="metric-label">Training Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s3:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-3">
                <div class="metric-value" style="color: #c084fc;">{stats['val']:,}</div>
                <div class="metric-label">Validation Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with s4:
            st.markdown(
                f"""
            <div class="metric-card anim-fade-up anim-delay-4">
                <div class="metric-value" style="color: #fb923c;">{stats['test']:,}</div>
                <div class="metric-label">Test Set</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            # Performance metrics
            st.markdown('<p class="section-label">Performance Metrics</p>', unsafe_allow_html=True)
            metrics_df = load_metrics()
            if metrics_df is not None:
                for col_name in metrics_df.columns:
                    val = metrics_df[col_name].iloc[0]
                    if "F1" in col_name:
                        clr = "var(--positive)"
                    elif "Accuracy" in col_name:
                        clr = "var(--accent-light)"
                    elif "Precision" in col_name:
                        clr = "#c084fc"
                    else:
                        clr = "#fb923c"
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

            # Avg words
            st.markdown(
                f"""
            <div class="metric-card" style="margin-top: 0.8rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="metric-label" style="text-transform: none; letter-spacing: 0;">Avg. Words per Review</span>
                    <span class="metric-value" style="font-size: 1.6rem; color: var(--text-primary);">{stats['avg_length']:.1f}</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Class distribution
            st.markdown(
                """
            <p style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
                      color: var(--text-muted); margin-top: 1.5rem; margin-bottom: 1rem; font-weight: 600;">
                Class Distribution
            </p>
            """,
                unsafe_allow_html=True,
            )
            if stats["classes"]:
                mapped = {SENTIMENT_MAPPING.get(k, k): v for k, v in stats["classes"].items()}
                dist_df = pd.DataFrame(list(mapped.items()), columns=["Sentiment", "Count"]).set_index("Sentiment")
                st.bar_chart(dist_df)
            else:
                st.info("Dataset statistics unavailable.")

        with col_right:
            # Confusion matrix
            st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
            cm_path = REPORTS_DIR / "confusion_matrix.png"
            cm_img = load_image_safe(cm_path)
            if cm_img:
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix not found. Run model evaluation first.")

            # Classification report
            report_path = REPORTS_DIR / "classification_report.txt"
            if report_path.exists():
                with st.expander("Classification Report", expanded=False):
                    try:
                        with open(report_path, "r") as f:
                            report_text = f.read()
                        st.code(report_text, language=None)
                    except Exception as e:
                        st.info(f"Could not load classification report: {e}")

        # Footer
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

        # Project Goal
        st.markdown(
            render_about_section(
                "&#9733;", "rgba(99,102,241,0.12)", "#818cf8",
                "Project Goal",
                """
                Build an explainable AI platform that analyzes employee feedback using state-of-the-art NLP
                and provides transparent, token-level explanations for every prediction. The goal is to give
                HR teams and startup founders actionable insights they can trust and act on.
                """,
            ),
            unsafe_allow_html=True,
        )

        # Problem Statement
        st.markdown(
            render_about_section(
                "&#9888;", "rgba(239,68,68,0.12)", "#f87171",
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

        # Solution
        st.markdown(
            render_about_section(
                "&#10003;", "rgba(34,197,94,0.12)", "#4ade80",
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

        # Key Features
        st.markdown(
            render_about_section(
                "&#9881;", "rgba(99,102,241,0.12)", "#818cf8",
                "Key Features",
                """
                <strong>Transformer Sentiment Analysis</strong> -- Fine-tuned DeBERTa-v3 for context-aware
                three-class classification.<br><br>
                <strong>SHAP Explainability</strong> -- Per-token importance scores with waterfall plots,
                bar charts, and interactive HTML visualizations.<br><br>
                <strong>Interactive Dashboard</strong> -- Five-page Streamlit application with dark theme
                designed for non-technical HR stakeholders.<br><br>
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

        # Technology Stack
        st.markdown(
            render_about_section(
                "&#9632;", "rgba(168,85,247,0.12)", "#c084fc",
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

        # Model Architecture
        st.markdown(
            render_about_section(
                "&#9670;", "rgba(59,130,246,0.12)", "#60a5fa",
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

        # Explainability
        st.markdown(
            render_about_section(
                "&#9673;", "rgba(34,197,94,0.12)", "#4ade80",
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

        # Future Improvements
        st.markdown(
            render_about_section(
                "&#9654;", "rgba(251,191,36,0.12)", "#fbbf24",
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

        # Application Areas
        st.markdown(
            render_about_section(
                "&#9632;", "rgba(244,114,182,0.12)", "#f472b6",
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

        # Architecture image
        arch_path = PROJECT_ROOT / "assets" / "architecture.png"
        arch_img = load_image_safe(arch_path)
        if arch_img:
            st.markdown('<p class="section-label">System Architecture</p>', unsafe_allow_html=True)
            st.image(arch_img, use_container_width=True)

        st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)

        # Links and license
        st.markdown(
            f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <p style="font-size: 0.82rem; color: var(--text-muted);">
                        <strong style="color: var(--text-primary);">Author:</strong> Nikhil Khetavath &nbsp;&middot;&nbsp;
                        <strong style="color: var(--text-primary);">License:</strong> MIT &nbsp;&middot;&nbsp;
                        <strong style="color: var(--text-primary);">Version:</strong> {APP_VERSION}
                    </p>
                </div>
                <div style="display: flex; gap: 8px;">
                    <a href="https://github.com/khetavathnikhil17-afk/StartupPulse-AI" target="_blank" style="
                        display: inline-block; background: var(--bg-card-hover); border: 1px solid var(--border);
                        border-radius: var(--radius-xs); padding: 8px 16px; color: var(--text-primary);
                        text-decoration: none; font-size: 0.82rem; font-weight: 500;">
                        GitHub Repository
                    </a>
                    <a href="https://www.linkedin.com/in/nikhilkhetavath-ai" target="_blank" style="
                        display: inline-block; background: var(--bg-card-hover); border: 1px solid var(--border);
                        border-radius: var(--radius-xs); padding: 8px 16px; color: var(--text-primary);
                        text-decoration: none; font-size: 0.82rem; font-weight: 500;">
                        LinkedIn Profile
                    </a>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Footer
        st.markdown(render_footer(), unsafe_allow_html=True)

except Exception as e:
    st.error("A critical error occurred while rendering the page.")
    st.error(f"Error Details: {e}")
    logger.error(f"Dashboard global exception: {e}")
