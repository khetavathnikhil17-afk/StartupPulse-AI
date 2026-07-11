"""
Logging configuration module for StartupPulse AI.

This module provides a centralized logging setup with console and file handlers.
Includes log rotation to prevent unbounded file growth.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from src.config.config import LOGS_DIR, LOG_FILENAME


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger with console and file handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler with rotation (max 5MB, keep 3 backups)
    log_file = LOGS_DIR / LOG_FILENAME
    fh = RotatingFileHandler(
        log_file, 
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
