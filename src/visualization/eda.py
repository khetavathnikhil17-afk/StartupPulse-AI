"""Exploratory Data Analysis report generator."""

import pandas as pd
from src.config.config import RAW_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_eda_report():
    """Generates basic statistics for the raw dataset."""
    if not RAW_DATA_PATH.exists():
        logger.warning("Raw data not found for EDA.")
        return None
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    stats = {
        "Total Reviews": len(df),
        "Missing Values": df.isnull().sum().to_dict()
    }
    
    if "label" in df.columns:
        stats["Class Counts"] = df["label"].value_counts().to_dict()
        
    # Additional statistics if columns exist
    if "company" in df.columns:
        stats["Top Companies"] = df["company"].value_counts().head(5).to_dict()
        
    if "rating" in df.columns:
        stats["Average Rating"] = df["rating"].mean()
        
    return stats