"""StartupPulse AI Dashboard v2.1 - Enterprise Dark Mode."""

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.components.header import render_page_header
from dashboard.components.section import render_section_label
from dashboard.components.cards import (
    render_metric_card,
    render_sentiment_badge,
    render_prob_bar,
    render_about_section,
    render_loading_html,
    render_footer,
)
from dashboard.components.charts import (
    render_horizontal_bar,
    render_circular_gauge,
    render_svg_line_chart,
)
from dashboard.utils.helpers import (
    sanitize_html,
    load_metrics,
    get_dataset_stats,
    load_image_safe,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StartupPulse AI",
    page_icon=str(PROJECT_ROOT / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).resolve().parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "review_text" not in st.session_state:
    st.session_state["review_text"] = ""

EXAMPLE_REVIEWS = {
    "Positive": {
        "label": "Positive Review",
        "text": "The company provides excellent work-life balance, supportive managers, and great career growth opportunities.",
    },
    "Neutral": {
        "label": "Neutral Review",
        "text": "The salary is average. The workload is manageable but there is little room for growth.",
    },
    "Negative": {
        "label": "Negative Review",
        "text": "Management ignores employee concerns. Compensation is poor and the work environment is stressful.",
    },
}

ABOUT_SECTIONS = [
    {
        "icon": "GO",
        "bg": "rgba(59,130,246,0.12)",
        "color": "#3b82f6",
        "title": "Project Goal",
        "body": (
            "Build an explainable AI platform that analyzes employee feedback using state-of-the-art NLP "
            "and provides transparent, token-level explanations for every prediction. The goal is to give "
            "HR teams and startup founders actionable insights they can trust and act on."
        ),
    },
    {
        "icon": "PS",
        "bg": "rgba(239,68,68,0.12)",
        "color": "#ef4444",
        "title": "Problem Statement",
        "body": (
            "Employee feedback is one of the most valuable signals a startup can collect, but most "
            "organizations lack the infrastructure to analyze it at scale. Manual tagging does not scale. "
            "Traditional sentiment analysis misses contextual nuance. Black-box predictions give HR teams "
            "no diagnostic value. And in HR analytics, predictions directly affect people's careers -- "
            "making transparency non-negotiable."
        ),
    },
    {
        "icon": "SO",
        "bg": "rgba(34,197,94,0.12)",
        "color": "#22c55e",
        "title": "Solution",
        "body": (
            "StartupPulse AI combines a fine-tuned Microsoft DeBERTa-v3 transformer with SHAP "
            "(SHapley Additive exPlanations) to deliver sentiment predictions that are both accurate "
            "and fully interpretable. Every token in an employee review receives an importance score, "
            "making the model's decision-making process transparent and auditable. An interactive "
            "Streamlit dashboard presents results to non-technical stakeholders."
        ),
    },
    {
        "icon": "KF",
        "bg": "rgba(59,130,246,0.12)",
        "color": "#3b82f6",
        "title": "Key Features",
        "body": (
            "<strong>Transformer Sentiment Analysis</strong> -- Fine-tuned DeBERTa-v3 for context-aware "
            "three-class classification.<br><br>"
            "<strong>SHAP Explainability</strong> -- Per-token importance scores with waterfall plots, "
            "bar charts, and interactive HTML visualizations.<br><br>"
            "<strong>Interactive Dashboard</strong> -- Five-page Streamlit application with clean enterprise "
            "design for non-technical HR stakeholders.<br><br>"
            "<strong>Real-time Prediction</strong> -- Singleton-loaded model with instant inference "
            "and confidence scores.<br><br>"
            "<strong>Probability Distribution</strong> -- Full three-class probability output, not just "
            "top-1 prediction.<br><br>"
            "<strong>Model Evaluation</strong> -- Automated pipeline generating classification reports, "
            "confusion matrices, and weighted F1 metrics."
        ),
    },
    {
        "icon": "TS",
        "bg": "rgba(167,139,250,0.12)",
        "color": "#a78bfa",
        "title": "Technology Stack",
        "body": (
            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px;">'
            '<span class="tech-pill">Python 3.10+</span>'
            '<span class="tech-pill">PyTorch 2.2</span>'
            '<span class="tech-pill">Hugging Face Transformers</span>'
            '<span class="tech-pill">SHAP 0.45</span>'
            '<span class="tech-pill">Streamlit 1.33</span>'
            '<span class="tech-pill">scikit-learn</span>'
            '<span class="tech-pill">Pandas / NumPy</span>'
            '<span class="tech-pill">Matplotlib / Seaborn</span>'
            '<span class="tech-pill">SentencePiece</span>'
            '</div>'
        ),
    },
    {
        "icon": "MA",
        "bg": "rgba(59,130,246,0.12)",
        "color": "#3b82f6",
        "title": "Model Architecture",
        "body": (
            "<strong>Base Model:</strong> microsoft/deberta-v3-base<br>"
            "<strong>Hidden Size:</strong> 768 dimensions<br>"
            "<strong>Layers:</strong> 12 transformer blocks<br>"
            "<strong>Attention Heads:</strong> 12<br>"
            "<strong>Attention Type:</strong> Disentangled (content-to-position + position-to-content)<br>"
            "<strong>Vocabulary:</strong> 128,100 tokens (SentencePiece)<br>"
            "<strong>Classification:</strong> 3-class softmax (Negative, Neutral, Positive)<br>"
            "<strong>Training:</strong> 3 epochs, learning rate 2e-5, batch size 16, early stopping (patience=3)"
        ),
    },
    {
        "icon": "EX",
        "bg": "rgba(34,197,94,0.12)",
        "color": "#22c55e",
        "title": "Explainability",
        "body": (
            "SHAP (SHapley Additive exPlanations) provides a game-theoretic framework for explaining "
            "individual predictions. For each input text, SHAP treats every token as a \"player\" and "
            "computes its marginal contribution to the final prediction. This produces three "
            "visualization types:<br><br>"
            "<strong>Waterfall Plots</strong> -- Cumulative token contributions from base value to final output.<br>"
            "<strong>Bar Charts</strong> -- Token importance ranking by absolute SHAP value.<br>"
            "<strong>Interactive HTML</strong> -- Color-coded tokens showing positive and negative "
            "contributions inline."
        ),
    },
    {
        "icon": "FI",
        "bg": "rgba(245,158,11,0.12)",
        "color": "#f59e0b",
        "title": "Future Improvements",
        "body": (
            "<strong>Aspect Extraction</strong> -- Per-dimension sentiment (management, compensation, growth, "
            "culture).<br>"
            "<strong>Multi-language Support</strong> -- DeBERTa multilingual variants for non-English feedback.<br>"
            "<strong>REST API</strong> -- FastAPI endpoints for HRIS integration.<br>"
            "<strong>Cloud Deployment</strong> -- Docker + Kubernetes on AWS/GCP.<br>"
            "<strong>Continuous Learning</strong> -- Feedback loop for periodic model retraining.<br>"
            "<strong>Voice Feedback</strong> -- Speech-to-text for town halls and exit interviews."
        ),
    },
    {
        "icon": "AA",
        "bg": "rgba(236,72,153,0.12)",
        "color": "#ec4899",
        "title": "Application Areas",
        "body": (
            "<strong>HR Teams</strong> -- Identify systemic issues in employee experience with evidence-backed "
            "explanations.<br>"
            "<strong>Startup Founders</strong> -- Scalable feedback analysis without a dedicated people analytics "
            "team.<br>"
            "<strong>Business Managers</strong> -- Department-level sentiment tracking and trend analysis.<br>"
            "<strong>Data Analysts</strong> -- Exportable visualizations and classification reports for "
            "stakeholder reporting."
        ),
    },
]

choice = render_sidebar(PROJECT_ROOT)

try:
    # ════════════════════════════════════════════════════════════════════════
    # HOME
    # ════════════════════════════════════════════════════════════════════════
    if choice == "Home":
        st.markdown(render_page_header(
            "StartupPulse AI",
            "Explainable AI for Employee Feedback Analytics"
        ), unsafe_allow_html=True)

        stats = get_dataset_stats()
        metrics_df = load_metrics()
        accuracy_val = 0.942
        if metrics_df is not None and "Accuracy" in metrics_df.columns:
            accuracy_val = metrics_df["Accuracy"].iloc[0]

        st.markdown("""
        <div class="hero-section">
            <h1>StartupPulse AI</h1>
            <p>Explainable Aspect-Based Sentiment Analysis for Employee Feedback. Powered by DeBERTa-v3 transformers and SHAP token-level explanations.</p>
        </div>
        """, unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">Sentiment Distribution</div>'
                f'{render_horizontal_bar("Positive", 42, "#22c55e")}'
                f'{render_horizontal_bar("Neutral", 31, "#6366f1")}'
                f'{render_horizontal_bar("Negative", 27, "#ef4444")}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">Review Trend (7 Days)</div>'
                f'{render_svg_line_chart()}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="card">'
                f'<div class="card-header">Model Accuracy</div>'
                f'<div style="display:flex;justify-content:center;margin:12px 0;">'
                f'{render_circular_gauge(round(accuracy_val * 100, 1))}'
                f'</div>'
                f'<p style="text-align:center;font-size:12px;color:#6b7280;margin:0;">DeBERTa-v3 Fine-tuned</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(render_section_label("Processing Pipeline"), unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-flow">
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(59,130,246,0.12);color:#3b82f6;">DS</div>
                Dataset
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(139,92,246,0.12);color:#a78bfa;">PP</div>
                Preprocessing
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(34,197,94,0.12);color:#22c55e;">DB</div>
                DeBERTa-v3
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(245,158,11,0.12);color:#f59e0b;">PR</div>
                Prediction
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(239,68,68,0.12);color:#ef4444;">SH</div>
                SHAP
            </div>
            <div class="pipeline-arrow">&rarr;</div>
            <div class="pipeline-step">
                <div class="pipeline-step-icon" style="background:rgba(59,130,246,0.12);color:#3b82f6;">IN</div>
                Insights
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(render_section_label("Capabilities"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="info-card">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                '<div style="width:32px;height:32px;border-radius:8px;background:rgba(59,130,246,0.12);color:#3b82f6;'
                'display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">TM</div>'
                '<h3 style="margin:0;">Transformer Model</h3>'
                '</div>'
                '<p>Fine-tuned Microsoft DeBERTa-v3 with disentangled attention for context-aware '
                'three-class sentiment classification on employee review data.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="info-card">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                '<div style="width:32px;height:32px;border-radius:8px;background:rgba(34,197,94,0.12);color:#22c55e;'
                'display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">RT</div>'
                '<h3 style="margin:0;">Real-time Prediction</h3>'
                '</div>'
                '<p>Singleton-loaded model ensures instant inference. Paste any employee review and '
                'receive a sentiment prediction with confidence scores in milliseconds.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="info-card">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                '<div style="width:32px;height:32px;border-radius:8px;background:rgba(245,158,11,0.12);color:#f59e0b;'
                'display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">XA</div>'
                '<h3 style="margin:0;">Explainable AI</h3>'
                '</div>'
                '<p>SHAP decomposes every prediction into per-token importance scores. Waterfall plots, '
                'bar charts, and interactive HTML reveal exactly why the model made its decision.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="info-card">'
                '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                '<div style="width:32px;height:32px;border-radius:8px;background:rgba(167,139,250,0.12);color:#a78bfa;'
                'display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">EA</div>'
                '<h3 style="margin:0;">Enterprise Analytics</h3>'
                '</div>'
                '<p>Multi-page Streamlit interface with clean enterprise design, analytics overview, confusion '
                'matrix, and exportable visualizations for stakeholder reporting.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(render_section_label("Dataset Overview"), unsafe_allow_html=True)
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.markdown(render_metric_card(f"{stats['total']:,}", "Total Reviews", "#3b82f6"), unsafe_allow_html=True)
        with q2:
            st.markdown(render_metric_card("3", "Sentiment Classes", "#22c55e"), unsafe_allow_html=True)
        with q3:
            st.markdown(render_metric_card("12", "Transformer Layers", "#a78bfa"), unsafe_allow_html=True)
        with q4:
            st.markdown(render_metric_card("768", "Hidden Dimensions", "#f59e0b"), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(render_section_label("Model Performance"), unsafe_allow_html=True)
        if metrics_df is not None:
            metric_defs = [
                ("Accuracy", "#3b82f6"),
                ("Precision", "#a78bfa"),
                ("Recall", "#ef4444"),
                ("F1 Score", "#22c55e"),
            ]
            cols = st.columns(4)
            for i, (col_name, clr) in enumerate(metric_defs):
                if col_name in metrics_df.columns:
                    val = metrics_df[col_name].iloc[0]
                    display_val = f"{val:.1%}" if val <= 1 else f"{val:.4f}"
                    with cols[i]:
                        st.markdown(render_metric_card(display_val, col_name, clr), unsafe_allow_html=True)
        else:
            st.info("Metrics not available. Run model evaluation first.")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        shap_preview_path = PROJECT_ROOT / "assets" / "shap_explanation.png"
        shap_img = load_image_safe(shap_preview_path)
        if shap_img is not None:
            st.markdown(render_section_label("Explainable AI Preview"), unsafe_allow_html=True)
            sh1, sh2 = st.columns([3, 2])
            with sh1:
                st.image(shap_img, use_container_width=True)
            with sh2:
                st.markdown(
                    '<div class="info-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">'
                    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                    '<div style="width:32px;height:32px;border-radius:8px;background:rgba(34,197,94,0.12);color:#22c55e;'
                    'display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;">TP</div>'
                    '<h3 style="margin:0;">Transparent Predictions</h3>'
                    '</div>'
                    '<p style="margin-top:8px;">Every prediction includes token-level SHAP explanations, making the AI '
                    'transparent and trustworthy. Each word is scored by its contribution to '
                    'the final sentiment classification.</p>'
                    '<p style="margin-top:12px;font-size:13px;">'
                    '<strong style="color:#3b82f6;">Navigate to Analyze Review</strong> '
                    'to try it with your own employee feedback.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(
            '<div class="tip-card">'
            '<strong>Get started:</strong> Navigate to <strong>Analyze Review</strong> in the sidebar to '
            'test the AI engine with a live employee review.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(render_footer(), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # ANALYZE REVIEW
    # ════════════════════════════════════════════════════════════════════════
    elif choice == "Analyze Review":
        st.markdown(render_page_header(
            "Analyze Review",
            "Enter an employee review to classify sentiment and generate SHAP explanations."
        ), unsafe_allow_html=True)

        st.markdown(render_section_label("Example Reviews"), unsafe_allow_html=True)
        ex_cols = st.columns(3)
        for i, (sentiment, data) in enumerate(EXAMPLE_REVIEWS.items()):
            with ex_cols[i]:
                if st.button(data["label"], key=f"example_{sentiment}", use_container_width=True):
                    st.session_state["review_text"] = data["text"]
                    st.rerun()

        st.markdown(render_section_label("Review Input"), unsafe_allow_html=True)
        review_input = st.text_area(
            "Employee Review",
            value=st.session_state["review_text"],
            height=160,
            placeholder="Paste an employee review here...",
            label_visibility="collapsed",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            analyze_clicked = st.button("Analyze Sentiment", use_container_width=True, type="primary")
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
                    from src.explainability.shap_explainer import explain_prediction
                    loading_container.markdown(render_loading_html(2, 2, 50), unsafe_allow_html=True)
                    result = explain_prediction(review_input)
                    st.session_state["prediction_result"] = result
                    loading_container.markdown(render_loading_html(5, 5, 100), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                loading_container.empty()

        if st.session_state["prediction_result"]:
            res = st.session_state["prediction_result"]
            label = res["label"]
            confidence = res["confidence"]
            probs = res["probabilities"]
            accent_color = {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#6366f1"}.get(label, "#3b82f6")
            prob_order = [("Positive", "#22c55e"), ("Neutral", "#6366f1"), ("Negative", "#ef4444")]
            prob_bars_html = ""
            for p_label, p_color in prob_order:
                prob_bars_html += render_prob_bar(p_label, probs.get(p_label, 0), p_color)

            st.markdown(
                f'<div style="margin-top:24px;">'
                f'<div class="result-grid">'
                f'<div class="card">'
                f'<div class="card-header">Prediction Result</div>'
                f'<div style="margin-bottom:12px;">{render_sentiment_badge(label)}</div>'
                f'<div style="margin-bottom:4px;">'
                f'<span style="font-size:11px;color:#6b7280;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;">Confidence</span>'
                f'</div>'
                f'<div style="font-size:28px;font-weight:700;letter-spacing:-0.02em;color:{accent_color};line-height:1;">{confidence:.1%}</div>'
                f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #1e1e30;">'
                f'<p style="font-size:11px;color:#6b7280;margin-bottom:4px;">Source Text</p>'
                f'<p style="font-size:13px;color:#9ca3af;line-height:1.5;max-height:80px;overflow-y:auto;">'
                f'{sanitize_html(st.session_state["review_text"][:200])}'
                f'{"..." if len(st.session_state["review_text"]) > 200 else ""}'
                f'</p></div></div>'
                f'<div class="card">'
                f'<div class="card-header">Class Probability Distribution</div>'
                f'{prob_bars_html}'
                f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid #1e1e30;">'
                f'<p style="font-size:13px;color:#9ca3af;">'
                f'Navigate to <strong style="color:#3b82f6;">Explainability</strong> '
                f'to view SHAP token-level explanations for this prediction.</p>'
                f'</div></div></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(render_footer(), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPLAINABILITY
    # ════════════════════════════════════════════════════════════════════════
    elif choice == "Explainability":
        st.markdown(render_page_header(
            "Explainability",
            "SHAP token-level explanations for the latest prediction."
        ), unsafe_allow_html=True)

        if not st.session_state["prediction_result"]:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">---</div>'
                '<h3>No prediction available</h3>'
                '<p>Run a sentiment analysis to generate explainability visualizations. '
                'Navigate to <strong>Analyze Review</strong> to get started.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            res = st.session_state["prediction_result"]
            st.markdown(
                f'<div class="card" style="margin-bottom:16px;">'
                f'<div class="card-header">Analyzed Text</div>'
                f'<p style="font-size:13px;color:#9ca3af;line-height:1.6;margin:0;">'
                f'{sanitize_html(st.session_state["review_text"])}</p>'
                f'<div style="margin-top:10px;">'
                f'{render_sentiment_badge(res["label"])}'
                f'<span style="margin-left:10px;font-size:13px;color:#9ca3af;">'
                f'Confidence: <strong style="color:#e5e7eb;">{res["confidence"]:.1%}</strong></span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            with st.spinner("Computing SHAP visualizations..."):
                try:
                    from src.config.config import REPORTS_DIR

                    @st.cache_resource(show_spinner=False)
                    def get_shap_explainer_cached():
                        from src.explainability.shap_explainer import SHAPExplainer
                        return SHAPExplainer()

                    explainer = get_shap_explainer_cached()
                    explainer.generate_plots(res, prefix="dashboard")
                except Exception as e:
                    st.error(f"SHAP generation failed: {e}")

            tokens = res["token_importance"]
            sorted_tokens = sorted(tokens.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            token_df = pd.DataFrame(sorted_tokens, columns=["Token", "Impact"])

            def color_impact(val):
                return "color: #22c55e; font-weight: 600;" if val > 0 else "color: #ef4444; font-weight: 600;"

            with st.expander("Top 10 Influential Tokens", expanded=True):
                st.dataframe(
                    token_df.style.map(color_impact, subset=["Impact"]).format({"Impact": "{:.6f}"}),
                    use_container_width=True,
                    height=370,
                )

            from src.config.config import REPORTS_DIR
            shap_dir = REPORTS_DIR / "shap"

            waterfall_file = shap_dir / "dashboard_waterfall.png"
            with st.expander("SHAP Waterfall Plot", expanded=True):
                if waterfall_file.exists():
                    img = load_image_safe(waterfall_file)
                    if img:
                        st.image(img, use_container_width=True)
                        with open(waterfall_file, "rb") as f:
                            st.download_button("Download Waterfall Plot", data=f.read(), file_name="shap_waterfall.png", mime="image/png")
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
                            components.html(f.read(), height=280, scrolling=True)
                    except Exception as e:
                        st.info(f"Could not load text visualization: {e}")
                else:
                    st.info("Text visualization not available.")

        st.markdown(render_footer(), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # MODEL METRICS
    # ════════════════════════════════════════════════════════════════════════
    elif choice == "Model Metrics":
        st.markdown(render_page_header(
            "Model Metrics",
            "Evaluation results and dataset statistics for the fine-tuned DeBERTa-v3 model."
        ), unsafe_allow_html=True)

        stats = get_dataset_stats()
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(render_metric_card(f"{stats['total']:,}", "Total Reviews", "#3b82f6"), unsafe_allow_html=True)
        with s2:
            st.markdown(render_metric_card(f"{stats['train']:,}", "Training Set", "#22c55e"), unsafe_allow_html=True)
        with s3:
            st.markdown(render_metric_card(f"{stats['val']:,}", "Validation Set", "#a78bfa"), unsafe_allow_html=True)
        with s4:
            st.markdown(render_metric_card(f"{stats['test']:,}", "Test Set", "#ef4444"), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(render_section_label("Performance Metrics"), unsafe_allow_html=True)
            metrics_df = load_metrics()
            if metrics_df is not None:
                color_map = {"F1": "#22c55e", "Accuracy": "#3b82f6", "Precision": "#a78bfa", "Recall": "#ef4444"}
                for col_name in metrics_df.columns:
                    val = metrics_df[col_name].iloc[0]
                    clr = "#9ca3af"
                    for key, c in color_map.items():
                        if key in col_name:
                            clr = c
                            break
                    display_val = f"{val:.1%}" if val <= 1 else f"{val:.4f}"
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span style="font-size:12px;color:#9ca3af;">{col_name}</span>'
                        f'<span style="font-size:18px;font-weight:700;color:{clr};">{display_val}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Metrics file not found. Run model evaluation first.")

            st.markdown(
                f'<div class="metric-card" style="margin-top:8px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-size:12px;color:#9ca3af;">Avg. Words per Review</span>'
                f'<span style="font-size:18px;font-weight:700;color:#e5e7eb;">{stats["avg_length"]:.1f}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown(render_section_label("Class Distribution"), unsafe_allow_html=True)
            if stats["classes"]:
                from src.config.config import SENTIMENT_MAPPING
                mapped = {SENTIMENT_MAPPING.get(k, k): v for k, v in stats["classes"].items()}
                dist_df = pd.DataFrame(list(mapped.items()), columns=["Sentiment", "Count"]).set_index("Sentiment")
                st.bar_chart(dist_df)
            else:
                st.info("Dataset statistics unavailable.")

        with col_right:
            st.markdown(render_section_label("Confusion Matrix"), unsafe_allow_html=True)
            cm_path = PROJECT_ROOT / "reports" / "confusion_matrix.png"
            cm_img = load_image_safe(cm_path)
            if cm_img:
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix not found. Run model evaluation first.")

            report_path = PROJECT_ROOT / "reports" / "classification_report.txt"
            if report_path.exists():
                with st.expander("Classification Report", expanded=False):
                    try:
                        with open(report_path, "r") as f:
                            st.code(f.read(), language=None)
                    except Exception as e:
                        st.info(f"Could not load classification report: {e}")

        st.markdown(render_footer(), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # ABOUT
    # ════════════════════════════════════════════════════════════════════════
    elif choice == "About":
        st.markdown(render_page_header(
            "About StartupPulse AI",
            "Technical overview and project information."
        ), unsafe_allow_html=True)

        for section in ABOUT_SECTIONS:
            st.markdown(
                render_about_section(section["icon"], section["bg"], section["color"], section["title"], section["body"]),
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        arch_path = PROJECT_ROOT / "assets" / "architecture.png"
        arch_img = load_image_safe(arch_path)
        if arch_img:
            st.markdown(render_section_label("System Architecture"), unsafe_allow_html=True)
            st.image(arch_img, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown(
            '<div class="card">'
            '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'
            '<div>'
            '<p style="font-size:13px;color:#9ca3af;margin:0;">'
            '<strong style="color:#e5e7eb;">Author:</strong> Nikhil Khetavath &nbsp;&middot;&nbsp; '
            '<strong style="color:#e5e7eb;">License:</strong> MIT &nbsp;&middot;&nbsp; '
            '<strong style="color:#e5e7eb;">Version:</strong> v2.1.0'
            '</p></div>'
            '<div style="display:flex;gap:8px;">'
            '<a href="https://github.com/khetavathnikhil17-afk/StartupPulse-AI" target="_blank" style="'
            'display:inline-block;background:#13131f;border:1px solid #2a2a3d;'
            'border-radius:8px;padding:6px 14px;color:#e5e7eb;'
            'text-decoration:none;font-size:13px;font-weight:500;">GitHub</a>'
            '<a href="https://www.linkedin.com/in/nikhilkhetavath-ai" target="_blank" style="'
            'display:inline-block;background:#13131f;border:1px solid #2a2a3d;'
            'border-radius:8px;padding:6px 14px;color:#e5e7eb;'
            'text-decoration:none;font-size:13px;font-weight:500;">LinkedIn</a>'
            '</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(render_footer(), unsafe_allow_html=True)

except Exception as e:
    st.error("A critical error occurred while rendering the page.")
    st.error(f"Error Details: {e}")
