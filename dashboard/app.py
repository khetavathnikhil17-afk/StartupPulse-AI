import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image
from pathlib import Path
import streamlit.components.v1 as components

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.explainability.shap_explainer import explain_prediction, SHAPExplainer, SHAP_DIR
from src.config.config import REPORTS_DIR, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Config
APP_VERSION = "v1.2.1-production"
st.set_page_config(
    page_title="StartupPulse AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .badge-positive {
        background-color: #198754;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-negative {
        background-color: #DC3545;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-neutral {
        background-color: #FFC107;
        color: black;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def load_metrics():
    metrics_file = REPORTS_DIR / "model_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None

def get_dataset_stats():
    """Loads metrics dynamically from datasets."""
    stats = {
        "train": 0, "val": 0, "test": 0, "total": 0, 
        "classes": {}, "avg_length": 0
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

def render_badge(sentiment: str):
    if sentiment.lower() == "positive":
        return f'<span class="badge-positive">Positive</span>'
    elif sentiment.lower() == "negative":
        return f'<span class="badge-negative">Negative</span>'
    else:
        return f'<span class="badge-neutral">Neutral</span>'

def main():
    st.sidebar.title("🚀 StartupPulse AI")
    st.sidebar.markdown(f"**Version**: `{APP_VERSION}`")
    st.sidebar.markdown("---")
    
    menu = ["Home", "Sentiment Analysis", "Explainability", "Analytics", "About"]
    choice = st.sidebar.radio("Navigation", menu)

    # State management
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = None
    if 'review_text' not in st.session_state:
        st.session_state['review_text'] = ""

    try:
        if choice == "Home":
            st.title("Welcome to StartupPulse AI 🚀")
            st.markdown("### Explainable Employee Sentiment Analysis")
            
            st.markdown("""
            StartupPulse AI bridges the gap between deep learning precision and human interpretability. 
            Powered by a fine-tuned **DeBERTa-v3** transformer model and **SHAP** (SHapley Additive exPlanations), 
            this platform delivers highly accurate sentiment predictions alongside fully transparent reasoning.
            """)
            
            st.markdown("---")
            
            st.markdown("### ✨ Key Features")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                <h4>🧠 Advanced NLP</h4>
                <p>Utilizes Microsoft's DeBERTa-v3 for context-aware 3-class sentiment tracking.</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric-card">
                <h4>🔍 XAI Integration</h4>
                <p>Real-time token impact visibility via natively integrated SHAP Explainer pipelines.</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="metric-card">
                <h4>📊 Dynamic Analytics</h4>
                <p>Real-time parsing of test sets, confusion matrices, and global model metrics.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("### 🚦 Quick Start")
            st.info("Navigate to **Sentiment Analysis** in the sidebar to test the AI engine with a live employee review.")

        elif choice == "Sentiment Analysis":
            st.title("📝 Sentiment Analysis")
            st.markdown("Enter an employee review below to evaluate sentiment and initiate explainable metrics.")
            
            review_input = st.text_area(
                "Review Text:", 
                value=st.session_state['review_text'], 
                height=150,
                placeholder="E.g., The management is fantastic and supports employee growth!"
            )
            
            # Use default button rendering instead of deprecated parameters
            if st.button("Analyze Sentiment"):
                if not review_input.strip():
                    st.warning("Please enter a valid review to analyze.")
                else:
                    st.session_state['review_text'] = review_input
                    with st.spinner("Processing through DeBERTa-v3 & initializing SHAP vectors..."):
                        try:
                            result = explain_prediction(review_input)
                            st.session_state['prediction_result'] = result
                            st.success("✅ Analysis Complete!")
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            logger.error(f"Dashboard analysis error: {e}")

            if st.session_state['prediction_result']:
                res = st.session_state['prediction_result']
                st.markdown("---")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### Result")
                    badge_html = render_badge(res['label'])
                    st.markdown(f"**Sentiment**: {badge_html}", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric("Model Confidence", f"{res['confidence']:.2%}")
                    
                with col2:
                    st.markdown("### Probability Distribution")
                    prob_df = pd.DataFrame(list(res['probabilities'].items()), columns=["Sentiment", "Probability"])
                    prob_df.set_index("Sentiment", inplace=True)
                    st.bar_chart(prob_df)
                    
                st.info("💡 **Tip**: Navigate to the **Explainability** tab to see exactly *why* the model made this decision.")

        elif choice == "Explainability":
            st.title("🔍 Explainable AI (SHAP)")
            st.markdown("Deep dive into the neural mechanisms of the latest prediction.")
            
            if not st.session_state['prediction_result']:
                st.warning("⚠️ Please run Sentiment Analysis first to generate explanations.")
            else:
                res = st.session_state['prediction_result']
                st.markdown(f"**Source Text:** *{st.session_state['review_text']}*")
                st.markdown("---")
                
                with st.spinner("Compiling SHAP visualizations..."):
                    explainer = SHAPExplainer()
                    explainer.generate_plots(res, prefix="dashboard")
                
                # Display Top Tokens
                st.subheader("💡 Top Influential Tokens")
                tokens = res['token_importance']
                sorted_tokens = sorted(tokens.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                token_df = pd.DataFrame(sorted_tokens, columns=["Token", "SHAP Impact Value"])
                
                def color_shap(val):
                    color = '#198754' if val > 0 else '#DC3545'
                    return f'color: {color}; font-weight: bold;'
                
                # Render dataframe safely
                st.dataframe(token_df.style.map(color_shap, subset=['SHAP Impact Value']))

                # Display Waterfall safely by loading into memory completely to avoid MediaFileHandler issues
                st.subheader("🌊 SHAP Waterfall Plot")
                waterfall_path = SHAP_DIR / "dashboard_waterfall.png"
                if waterfall_path.exists():
                    img = Image.open(str(waterfall_path))
                    st.image(img)
                    
                    # Safe download handler by reading byte content first
                    with open(waterfall_path, "rb") as file:
                        img_bytes = file.read()
                        
                    st.download_button("Download Waterfall Plot", data=img_bytes, file_name="shap_waterfall.png", mime="image/png")
                else:
                    st.warning("Waterfall plot could not be generated. Please ensure the backend is evaluating correctly.")
                    
                st.markdown("---")
                    
                # Display Text Plot (HTML) safely
                st.subheader("📄 SHAP Text Impact Analysis")
                text_path = SHAP_DIR / "dashboard_text.html"
                if text_path.exists():
                    with open(text_path, "r", encoding="utf-8") as f:
                        html_str = f.read()
                    components.html(html_str, height=250, scrolling=True)
                else:
                    st.warning("Text explanation could not be generated.")

        elif choice == "Analytics":
            st.title("📊 Global Analytics")
            st.markdown("Macro-level statistics covering model evaluation metrics and underlying data distribution.")
            
            stats = get_dataset_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviews", stats["total"])
            col2.metric("Training Set", stats["train"])
            col3.metric("Validation Set", stats["val"])
            col4.metric("Test Set", stats["test"])
            
            st.markdown("---")
            
            colA, colB = st.columns(2)
            
            with colA:
                st.subheader("Performance Metrics")
                metrics_df = load_metrics()
                if metrics_df is not None:
                    st.dataframe(metrics_df)
                else:
                    st.warning("Metrics not found. Run model evaluation first.")
                    
                st.subheader("Class Distribution (Full Dataset)")
                if stats["classes"]:
                    class_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
                    mapped_classes = {class_mapping.get(k, k): v for k, v in stats["classes"].items()}
                    dist_df = pd.DataFrame(list(mapped_classes.items()), columns=["Sentiment", "Count"]).set_index("Sentiment")
                    st.bar_chart(dist_df)
                else:
                    st.warning("Dataset statistics unavailable.")
                
            with colB:
                st.subheader("Confusion Matrix")
                cm_path = REPORTS_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    # Load securely into memory via PIL
                    cm_img = Image.open(str(cm_path))
                    st.image(cm_img)
                else:
                    st.warning("Confusion matrix not found. Run model evaluation first.")
                    
                st.metric("Avg. Words per Review", f"{stats['avg_length']:.1f}")

        elif choice == "About":
            st.title("ℹ️ About StartupPulse AI")
            st.markdown(f"**Version**: {APP_VERSION}")
            
            st.markdown("""
            ### 🎯 Project Overview
            **StartupPulse AI** bridges the gap between deep machine learning NLP capabilities and stakeholder transparency. 
            By providing explicit SHAP reasoning behind every prediction, the application transitions from a "black box" model to a trusted analytical tool for HR metrics and employee sentiment tracking.
            
            ### 🧠 Architecture
            - **Model Framework**: HuggingFace Transformers (`microsoft/deberta-v3-base`)
            - **Explainability**: SHAP (SHapley Additive exPlanations) Token & Partition Explainers
            - **Frontend Interface**: Streamlit
            - **Pipeline Framework**: PyTorch
            
            ### 📂 Dataset
            The model parses structured qualitative employee feedback mapped into a strict trichotomy:
            - `0`: Negative
            - `1`: Neutral
            - `2`: Positive
            
            ### 🔮 Future Work
            - Deployment scaling via Docker & AWS/GCP pipelines.
            - Unsupervised topic modeling alongside sentiment extraction.
            - Active Learning feedback loops to continually adapt to dynamic organizational jargon.
            
            ---
            **Author**: *StartupPulse AI Engineering Team*  
            **License**: MIT
            """)

    except Exception as e:
        st.error("A critical error occurred while rendering the page. The application has safely caught the exception.")
        st.error(f"Error Details: {str(e)}")
        logger.error(f"Dashboard global exception: {e}")

if __name__ == "__main__":
    main()
