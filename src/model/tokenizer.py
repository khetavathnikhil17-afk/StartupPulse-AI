from transformers import AutoTokenizer
from src.config.config import MODEL_NAME, MODEL_SAVE_DIR
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

def get_tokenizer():
    """Loads tokenizer from local save directory if exists, else downloads it."""
    if MODEL_SAVE_DIR.exists() and (MODEL_SAVE_DIR / "tokenizer.json").exists():
        logger.info(f"Loading tokenizer from {MODEL_SAVE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SAVE_DIR))
    else:
        logger.info(f"Downloading tokenizer {MODEL_NAME}")
        # DeBERTa-v3 specifically requires `use_fast=False` for some tokenizers, but sentencepiece solves this
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    return tokenizer