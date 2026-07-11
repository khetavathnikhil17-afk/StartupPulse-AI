"""
Dataset download utilities for StartupPulse AI.

This module provides functionality to download employee review datasets.
"""
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_dataset(output_dir: Path) -> Path:
    """
    Download employee review dataset.
    
    Args:
        output_dir: Directory to save the dataset
        
    Returns:
        Path to the output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset should be manually placed in data/ directory
    logger.info("Dataset download not automated.")
    logger.info("Please place your employee_reviews.csv in the data/ directory.")
    
    return output_dir


if __name__ == "__main__":
    from src.config.config import DATA_DIR
    
    download_dataset(DATA_DIR)
