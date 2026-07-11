"""
Helper utilities module for StartupPulse AI.

This module provides common utility functions for file I/O operations.
"""
import json
from pathlib import Path


def save_json(data: dict, file_path: Path) -> None:
    """
    Save dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to the output JSON file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> dict:
    """
    Load a JSON file into a dictionary.
    
    Args:
        file_path: Path to the input JSON file
        
    Returns:
        Loaded dictionary, or empty dict if file doesn't exist
    """
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
