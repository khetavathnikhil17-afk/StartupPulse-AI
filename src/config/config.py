from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data Paths
RAW_DATA_PATH = DATA_DIR / "employee_reviews.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
VALIDATION_DATA_PATH = DATA_DIR / "validation.csv"

# Model Config
MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_SAVE_DIR = MODELS_DIR / "deberta_v3"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Explainability
SHAP_SAMPLES = 100

# Sentiment Mapping
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
INVERSE_SENTIMENT_MAPPING = {v: k for k, v in SENTIMENT_MAPPING.items()}
NUM_LABELS = len(SENTIMENT_MAPPING)

# Random Seed
SEED = 42