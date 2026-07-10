import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from src.config.config import MODEL_SAVE_DIR, SENTIMENT_MAPPING, REPORTS_DIR, MODEL_NAME
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Output directory for SHAP reports
SHAP_DIR = REPORTS_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

class SHAPExplainer:
    """
    Explainable AI (XAI) module using SHAP for DeBERTa-v3 sentiment classification.
    Automatically handles CPU/GPU execution and generates visualization plots.
    """
    def __init__(self):
        """Initializes the tokenizer, model, and SHAP explainer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SHAPExplainer on device: {self.device}")
        
        # Verify model directory exists
        if not MODEL_SAVE_DIR.exists() or not (MODEL_SAVE_DIR / "config.json").exists():
            logger.warning(f"Model not found at {MODEL_SAVE_DIR}. Downloading base model for testing...")
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
            self.model.save_pretrained(str(MODEL_SAVE_DIR))
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SAVE_DIR))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_SAVE_DIR))
            
        self.model.to(self.device)
        self.model.eval()

        # Build SHAP explainer for transformers text models
        self.explainer = shap.Explainer(self._predict_func, self.tokenizer)

    def _predict_func(self, texts):
        """
        Internal inference function for SHAP that outputs probabilities.
        
        Args:
            texts (list): List of input texts.
            
        Returns:
            np.ndarray: Matrix of class probabilities.
        """
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        return probs.cpu().numpy()

    def explain_prediction(self, text: str):
        """
        Explains a single prediction using SHAP.
        
        Args:
            text (str): The input text to explain.
            
        Returns:
            dict: Dictionary containing predicted label, confidence, probability distribution, 
                  SHAP values, and token importance.
        """
        logger.info("Generating explanation for text...")
        
        try:
            # Generate probabilities
            probs = self._predict_func([text])[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            label = SENTIMENT_MAPPING.get(pred_idx, "Unknown")
            
            prob_dist = {SENTIMENT_MAPPING.get(i, f"Class {i}"): float(p) for i, p in enumerate(probs)}

            # Generate SHAP values
            shap_values = self.explainer([text])
            
            # Extract token importance (for the predicted class)
            tokens = shap_values.data[0]
            values = shap_values.values[0, :, pred_idx]
            token_importance = {str(t): float(v) for t, v in zip(tokens, values)}
            
            return {
                "label": label,
                "confidence": confidence,
                "probabilities": prob_dist,
                "shap_values": shap_values,
                "token_importance": token_importance,
                "predicted_class_index": pred_idx
            }
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            raise RuntimeError(f"Failed to generate SHAP explanation: {e}")

    def generate_plots(self, explanation_result, prefix="plot"):
        """
        Generates and saves Waterfall, Text, and Bar (Summary) plots.
        
        Args:
            explanation_result (dict): The output from explain_prediction().
            prefix (str): Prefix for the saved plot filenames.
        """
        shap_values = explanation_result["shap_values"]
        pred_idx = explanation_result["predicted_class_index"]
        
        # 1. Waterfall Plot
        try:
            plt.figure(figsize=(10, 6))
            exp = shap.Explanation(
                values=shap_values.values[0, :, pred_idx],
                base_values=shap_values.base_values[0, pred_idx],
                data=shap_values.data[0]
            )
            shap.plots.waterfall(exp, show=False)
            plt.title(f"SHAP Waterfall Plot ({SENTIMENT_MAPPING[pred_idx]})")
            waterfall_path = SHAP_DIR / f"{prefix}_waterfall.png"
            plt.savefig(waterfall_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Waterfall plot to {waterfall_path}")
        except Exception as e:
            logger.error(f"Failed to generate Waterfall plot: {e}")

        # 2. Bar Plot (Summary of Token Importance)
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.bar(exp, show=False)
            plt.title(f"SHAP Token Importance Bar Plot ({SENTIMENT_MAPPING[pred_idx]})")
            bar_path = SHAP_DIR / f"{prefix}_bar_summary.png"
            plt.savefig(bar_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved Bar Summary plot to {bar_path}")
        except Exception as e:
            logger.error(f"Failed to generate Bar plot: {e}")
            
        # 3. Text Plot
        try:
            # text plot generates HTML representation
            html_content = shap.plots.text(shap_values[:, :, pred_idx], display=False)
            text_path = SHAP_DIR / f"{prefix}_text.html"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Saved Text HTML plot to {text_path}")
        except Exception as e:
            logger.error(f"Failed to generate Text plot: {e}")

# Reusable function as requested
_global_explainer = None

def explain_prediction(text: str):
    """
    Main reusable interface for generating SHAP predictions.
    
    Args:
        text (str): Input text to explain.
        
    Returns:
        dict: Contains label, confidence, SHAP values, token importance, and probability distribution.
    """
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = SHAPExplainer()
    return _global_explainer.explain_prediction(text)


if __name__ == "__main__":
    logger.info("Running SHAP Explainability verification...")
    
    sample_text = "The new management is fantastic and truly cares about employees."
    
    try:
        explainer = SHAPExplainer()
        result = explainer.explain_prediction(sample_text)
        
        logger.info(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
        logger.info(f"Token Importance (Top 3): {sorted(result['token_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]}")
        
        explainer.generate_plots(result, prefix="verification")
        
        logger.info(f"Verification successful. Plots saved to {SHAP_DIR}")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
