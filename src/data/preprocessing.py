"""
Data preprocessing module for StartupPulse AI.

This module provides functions to clean and preprocess employee review data.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load raw employee reviews dataset.
    
    Args:
        data_path: Path to raw data CSV
        
    Returns:
        Loaded DataFrame
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not needed for analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unnecessary columns removed
    """
    drop_columns = ["Unnamed: 0", "link", "advice-to-mgmt"]
    df = df.drop(columns=drop_columns, errors="ignore")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing summary values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values filled
    """
    df["summary"] = df["summary"].fillna("")
    return df


def create_review_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine summary, pros, and cons into single review column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with review column added
    """
    df["review"] = (
        df["summary"].astype(str)
        + ". "
        + df["pros"].astype(str)
        + ". "
        + df["cons"].astype(str)
    )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate reviews.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_count = len(df)
    df = df.drop_duplicates(subset=["review"])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate reviews")
    return df


def remove_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty or whitespace-only reviews.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with empty reviews removed
    """
    df = df[df["review"].str.strip() != ""]
    return df


def keep_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns needed for analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with only useful columns
    """
    useful_columns = [
        "company",
        "location",
        "job-title",
        "overall-ratings",
        "work-balance-stars",
        "culture-values-stars",
        "carrer-opportunities-stars",
        "comp-benefit-stars",
        "senior-mangemnet-stars",
        "review",
    ]
    return df[useful_columns]


def preprocess_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    df = load_raw_data(input_path)
    df = drop_unnecessary_columns(df)
    df = fill_missing_values(df)
    df = create_review_column(df)
    df = remove_duplicates(df)
    df = remove_empty_reviews(df)
    df = keep_useful_columns(df)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Preprocessing completed. Shape: {df.shape}")
    logger.info(f"Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    from src.config.config import RAW_DATA_PATH, DATA_DIR
    
    output_path = DATA_DIR / "processed" / "clean_employee_reviews.csv"
    df = preprocess_data(RAW_DATA_PATH, output_path)
    
    logger.info(f"\nFirst 5 rows:")
    logger.info(f"\n{df.head()}")
