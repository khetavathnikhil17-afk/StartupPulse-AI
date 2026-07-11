"""
CPU-optimized training script for StartupPulse AI.
Uses a subset of data and fewer epochs for reasonable CPU training time.
"""
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from src.config.config import (
    MODEL_NAME, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, MODEL_SAVE_DIR,
    MAX_LENGTH, NUM_LABELS, EARLY_STOPPING_PATIENCE, LOGS_DIR, SEED
)
from src.pipeline.dataset import SentimentDataset
from src.utils.metrics import compute_metrics
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

# CPU optimization: use subset and fewer epochs
TRAIN_SUBSET_SIZE = 10000  # Use 10K samples instead of 53K for CPU
VAL_SUBSET_SIZE = 2000
NUM_EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 3e-5


def train_model_cpu():
    """Train sentiment model optimized for CPU."""
    set_seed()

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    logger.info("Loading datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VALIDATION_DATA_PATH)

    # Subsample for CPU training
    if len(train_df) > TRAIN_SUBSET_SIZE:
        train_df = train_df.sample(n=TRAIN_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
        logger.info(f"Using {TRAIN_SUBSET_SIZE} training samples")

    if len(val_df) > VAL_SUBSET_SIZE:
        val_df = val_df.sample(n=VAL_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
        logger.info(f"Using {VAL_SUBSET_SIZE} validation samples")

    train_dataset = SentimentDataset(
        train_df["review"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH
    )
    val_dataset = SentimentDataset(
        val_df["review"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH
    )

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(LOGS_DIR),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,  # CPU - no mixed precision
        save_total_limit=2,
        report_to="none",  # No wandb/tensorboard
        dataloader_num_workers=0,  # CPU single process
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    ))

    logger.info(f"Starting training on CPU ({NUM_EPOCHS} epochs, {TRAIN_SUBSET_SIZE} samples)...")
    trainer.train()

    logger.info(f"Saving model to {MODEL_SAVE_DIR}...")
    trainer.save_model(str(MODEL_SAVE_DIR))
    tokenizer.save_pretrained(str(MODEL_SAVE_DIR))

    logger.info("Training complete.")
    return trainer


if __name__ == "__main__":
    train_model_cpu()
