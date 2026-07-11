"""
Dataset preparation module for StartupPulse AI.

This module handles loading, splitting, and preparing data for training.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config import RAW_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH, SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)

def prepare_data():
    """Reads raw data, splits it, and saves train/validation/test datasets."""
    if not RAW_DATA_PATH.exists():
        logger.warning(f"Raw data file not found at {RAW_DATA_PATH}")
        logger.info("Creating dummy data for testing purposes.")
        dummy_data = pd.DataFrame({
            "review": [
                "Great place to work!", "Terrible management.", "It is okay.",
                "I love this company.", "Awful experience.", "Average.",
                "Fantastic team.", "Worst job ever.", "Neutral feelings.",
                "Good work life balance.", "Toxic culture.", "Just fine.",
                "Amazing benefits.", "Very poor leadership.", "Acceptable.",
                "Excellent workplace.", "Bad environment.", "Nothing special."
            ],
            "label": [
                2, 0, 1,
                2, 0, 1,
                2, 0, 1,
                2, 0, 1,
                2, 0, 1,
                2, 0, 1
            ]
        })
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        dummy_data.to_csv(RAW_DATA_PATH, index=False)

    df = pd.read_csv(RAW_DATA_PATH)
    
    # Preprocessing: ensure required columns exist
    if 'review' not in df.columns or 'label' not in df.columns:
        logger.error("Dataset must contain 'review' and 'label' columns.")
        return

    df = df.dropna(subset=['review', 'label'])
    df['label'] = df['label'].astype(int)

    # Train (80%), Val (10%), Test (10%)
    stratify_col = df['label'] if len(df) > 50 else None
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=stratify_col)
    
    stratify_col_temp = temp_df['label'] if len(temp_df) > 50 else None
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=stratify_col_temp)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VALIDATION_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    logger.info(f"Data prepared successfully. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    prepare_data()