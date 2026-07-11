"""
Model trainer module for StartupPulse AI.

This module configures and provides the Hugging Face Trainer for model training.
"""
from transformers import Trainer, TrainingArguments
from src.config.config import MODEL_SAVE_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, LOGS_DIR
from src.utils.metrics import compute_metrics
import torch

def get_trainer(model, train_dataset, val_dataset, processing_class):
    """Configures and returns the Transformers Trainer."""
    training_args = TrainingArguments(
        output_dir=str(MODEL_SAVE_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=str(LOGS_DIR),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(), # Mixed Precision (GPU)
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        processing_class=processing_class
    )
    return trainer