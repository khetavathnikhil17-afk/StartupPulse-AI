"""
Tokenizer module for StartupPulse AI.

This module handles loading and saving the Hugging Face tokenizer.
"""
from transformers import AutoTokenizer
from src.config.config import MODEL_NAME, MODEL_SAVE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_tokenizer():
    """
    Load tokenizer from local save directory if exists, otherwise download.
    
    Returns:
        AutoTokenizer instance
    """
    if MODEL_SAVE_DIR.exists() and (MODEL_SAVE_DIR / "tokenizer.json").exists():
        logger.info(f"Loading tokenizer from {MODEL_SAVE_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SAVE_DIR))
    else:
        logger.info(f"Downloading tokenizer {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(MODEL_SAVE_DIR))
    return tokenizer
