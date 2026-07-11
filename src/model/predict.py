"""
Sentiment prediction module for StartupPulse AI.

This module provides a singleton-based sentiment predictor using the
fine-tuned DeBERTa-v3 model.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from src.config.config import MODEL_SAVE_DIR, SENTIMENT_MAPPING, MAX_LENGTH, MODEL_NAME, NUM_LABELS
from src.model.tokenizer import get_tokenizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPredictor:
    """
    Singleton sentiment predictor using DeBERTa-v3.
    
    Automatically downloads the base model if fine-tuned model is not available.
    """
    
    def __init__(self):
        """Initialize the predictor with model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer()
        
        # Automatic Checks
        if not MODEL_SAVE_DIR.exists():
            MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            
        if not (MODEL_SAVE_DIR / "config.json").exists():
            logger.warning(f"Model not found at {MODEL_SAVE_DIR}. Downloading base model...")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS
            )
            model.save_pretrained(str(MODEL_SAVE_DIR))
            
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_SAVE_DIR))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            dict: Contains label, confidence, and probability distribution
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).squeeze().tolist()
            
        pred_idx = logits.argmax(dim=1).item()
        
        return {
            "label": SENTIMENT_MAPPING[pred_idx],
            "confidence": probs[pred_idx],
            "probabilities": {SENTIMENT_MAPPING[i]: p for i, p in enumerate(probs)}
        }


# Global predictor instance for reuse
_predictor = None


def predict_sentiment(text: str) -> dict:
    """
    Wrapper function for sentiment prediction.
    
    Args:
        text: Input text to classify
        
    Returns:
        dict: Contains label, confidence, and probability distribution
    """
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor.predict(text)



