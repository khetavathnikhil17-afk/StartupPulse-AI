"""
Label creation utilities for StartupPulse AI.

This module provides functionality to create sentiment labels from ratings.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rating_to_sentiment(rating: float) -> int:
    """
    Convert numeric rating to sentiment label.
    
    Args:
        rating: Numeric rating (1-5 scale)
        
    Returns:
        Sentiment label: 0 (Negative), 1 (Neutral), 2 (Positive)
    """
    if rating <= 2:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def create_labels(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Create sentiment labels from ratings.
    
    Args:
        input_path: Path to cleaned dataset
        output_path: Path to save labeled dataset
        
    Returns:
        Labeled DataFrame
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    df["label"] = df["overall-ratings"].apply(rating_to_sentiment)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    logger.info(f"Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    from src.config.config import DATA_DIR
    
    input_path = DATA_DIR / "processed" / "clean_employee_reviews.csv"
    output_path = DATA_DIR / "processed" / "labeled_employee_reviews.csv"
    
    df = create_labels(input_path, output_path)
    
    logger.info("\nLabel Distribution:")
    logger.info(f"\n{df['label'].value_counts()}")
