"""
Random seed configuration module for StartupPulse AI.

This module provides functions to set random seeds for reproducibility.
"""
import random
import numpy as np
import torch
import os
from src.config.config import SEED


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: SEED from config)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
