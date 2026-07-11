"""
Test suite for StartupPulse AI.

This module contains unit tests for the core functionality.
Tests are designed to run quickly without loading ML models.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


class TestConfig:
    """Test configuration module."""
    
    def test_imports(self):
        """Test that config module imports correctly."""
        from src.config.config import (
            PROJECT_ROOT, SRC_DIR, DATA_DIR, MODELS_DIR,
            REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR,
            MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS,
            LEARNING_RATE, SENTIMENT_MAPPING, NUM_LABELS, SEED
        )
        assert PROJECT_ROOT.exists()
        assert MODEL_NAME == "microsoft/deberta-v3-base"
        assert MAX_LENGTH == 128
        assert NUM_LABELS == 3
    
    def test_directories_exist(self):
        """Test that required directories are created."""
        from src.config.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR
        assert DATA_DIR.exists()
        assert MODELS_DIR.exists()
        assert REPORTS_DIR.exists()
        assert LOGS_DIR.exists()
    
    def test_sentiment_mapping(self):
        """Test sentiment mapping is correct."""
        from src.config.config import SENTIMENT_MAPPING, INVERSE_SENTIMENT_MAPPING
        assert SENTIMENT_MAPPING == {0: "Negative", 1: "Neutral", 2: "Positive"}
        assert INVERSE_SENTIMENT_MAPPING == {"Negative": 0, "Neutral": 1, "Positive": 2}


class TestLogger:
    """Test logging module."""
    
    def test_get_logger(self):
        """Test that get_logger returns a configured logger."""
        from src.utils.logger import get_logger
        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"
        assert len(logger.handlers) == 2  # Console and file handlers


class TestPreprocessing:
    """Test data preprocessing module."""
    
    def test_rating_to_sentiment(self):
        """Test rating to sentiment conversion."""
        from src.data.create_labels import rating_to_sentiment
        assert rating_to_sentiment(1) == 0  # Negative
        assert rating_to_sentiment(2) == 0  # Negative
        assert rating_to_sentiment(3) == 1  # Neutral
        assert rating_to_sentiment(4) == 2  # Positive
        assert rating_to_sentiment(5) == 2  # Positive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
