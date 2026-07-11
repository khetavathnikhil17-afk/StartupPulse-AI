"""
Model training module for StartupPulse AI.

This module handles the training of the DeBERTa-v3 sentiment classifier.
"""
import pandas as pd
from transformers import AutoModelForSequenceClassification, EarlyStoppingCallback
from src.config.config import (
    MODEL_NAME, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, MODEL_SAVE_DIR, 
    MAX_LENGTH, NUM_LABELS, EARLY_STOPPING_PATIENCE
)
from src.pipeline.dataset import SentimentDataset
from src.pipeline.prepare_dataset import prepare_data
from src.model.tokenizer import get_tokenizer
from src.model.trainer import get_trainer
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def train_model() -> None:
    """
    Train the sentiment analysis model.
    
    Prepares data, initializes model and trainer, and saves the best model.
    """
    set_seed()
    
    if not TRAIN_DATA_PATH.exists() or not VALIDATION_DATA_PATH.exists():
        logger.info("Data not found, preparing data...")
        prepare_data()

    logger.info("Loading tokenizer and model...")
    tokenizer = get_tokenizer()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    logger.info("Loading datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VALIDATION_DATA_PATH)

    train_dataset = SentimentDataset(
        train_df["review"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH
    )
    val_dataset = SentimentDataset(
        val_df["review"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH
    )

    logger.info("Initializing trainer...")
    trainer = get_trainer(model, train_dataset, val_dataset, tokenizer)
    
    # Add early stopping callback
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    ))

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving best model to {MODEL_SAVE_DIR}...")
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    
    logger.info("Training complete.")


if __name__ == "__main__":
    train_model()
