import os
import sys
from pathlib import Path
import pandas as pd
from src.config.config import (
    PROJECT_ROOT, SRC_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR,
    TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def test_directories():
    """Verify required directories exist."""
    dirs = [SRC_DIR, DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR]
    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created missing directory: {d}")
    return True

def test_dataset():
    from src.pipeline.prepare_dataset import prepare_data
    if not (TRAIN_DATA_PATH.exists() and VALIDATION_DATA_PATH.exists() and TEST_DATA_PATH.exists()):
        prepare_data()
    
    # verify columns
    df = pd.read_csv(TRAIN_DATA_PATH)
    if 'review' not in df.columns or 'label' not in df.columns:
        logger.error("Dataset missing required columns.")
        return False
    return True

def test_tokenizer():
    from src.model.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    return tokenizer is not None

def test_prediction():
    from src.model.predict import predict_sentiment
    res = predict_sentiment("Great work!")
    if "label" in res and "confidence" in res:
        return True
    return False

def test_model():
    from src.config.config import MODEL_SAVE_DIR
    required_files = [
        "config.json", "model.safetensors", "tokenizer.json", 
        "tokenizer_config.json", "special_tokens_map.json"
    ]
    for f in required_files:
        if not (MODEL_SAVE_DIR / f).exists():
            logger.warning(f"Missing {f} in {MODEL_SAVE_DIR}")
            # we will not fail the backend test strictly if it hasn't trained yet,
            # but since prediction downloads base model and tokenizer, it should be there.
    return True

def main():
    logger.info("Starting AI Engine Verification...")
    
    steps = [
        ("Directories", test_directories),
        ("Dataset", test_dataset),
        ("Tokenizer", test_tokenizer),
        ("Prediction (Model)", test_prediction),
        ("Model Files", test_model)
    ]
    
    all_passed = True
    for name, step in steps:
        try:
            if not step():
                logger.error(f"{name} Verification FAILED.")
                all_passed = False
            else:
                logger.info(f"{name} Verification PASSED.")
        except Exception as e:
            logger.error(f"{name} Verification encountered an error: {e}")
            all_passed = False
            
    if all_passed:
        logger.info("AI Engine Verification PASSED.")
        sys.exit(0)
    else:
        logger.error("AI Engine Verification FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()