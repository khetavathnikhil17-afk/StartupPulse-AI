"""
Dataset validation utilities for StartupPulse AI.

This module provides functionality to check and validate datasets.
"""
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_csv_files(data_folder: Path) -> list:
    """
    Find all CSV files in a directory.
    
    Args:
        data_folder: Path to search for CSV files
        
    Returns:
        List of CSV file paths
    """
    if not data_folder.exists():
        logger.warning(f"Directory does not exist: {data_folder}")
        return []
    
    return list(data_folder.glob("*.csv"))


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """
    Load dataset with automatic separator detection.
    
    Args:
        dataset_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(dataset_path, sep=sep)
            if len(df.columns) > 1:
                logger.info(f"Loaded dataset using separator: '{sep}'")
                return df
        except Exception as e:
            logger.debug(f"Separator '{sep}' failed: {e}")
            continue
    
    raise ValueError(f"Could not load dataset: {dataset_path}")


def check_dataset(data_folder: Path) -> dict:
    """
    Check and validate dataset in the specified folder.
    
    Args:
        data_folder: Path to data directory
        
    Returns:
        Dictionary with dataset information
    """
    csv_files = find_csv_files(data_folder)
    
    if not csv_files:
        logger.warning(f"No CSV files found in {data_folder}")
        return {"files_found": 0, "dataframes": []}
    
    results = {
        "files_found": len(csv_files),
        "dataframes": []
    }
    
    for csv_file in csv_files:
        try:
            df = load_dataset(csv_file)
            info = {
                "file": str(csv_file),
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "missing_values": df.isnull().sum().to_dict()
            }
            results["dataframes"].append(info)
            logger.info(f"Validated: {csv_file.name} - Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    return results


if __name__ == "__main__":
    from src.config.config import DATA_DIR
    
    results = check_dataset(DATA_DIR)
    
    logger.info(f"\nFiles found: {results['files_found']}")
    for df_info in results["dataframes"]:
        logger.info(f"\nFile: {df_info['file']}")
        logger.info(f"Shape: {df_info['shape']}")
        logger.info(f"Columns: {df_info['columns']}")
