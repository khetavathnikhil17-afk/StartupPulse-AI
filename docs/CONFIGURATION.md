# Configuration Guide

Customize StartupPulse AI settings and behavior.

---

## Configuration File

All settings are centralized in `src/config/config.py`:

```python
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

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

# Random Seed
SEED = 42
```

---

## Path Configuration

### Custom Data Directory

```python
# Change data directory
DATA_DIR = Path("/your/custom/data/path")
RAW_DATA_PATH = DATA_DIR / "employee_reviews.csv"
```

### Custom Model Directory

```python
# Change model save location
MODELS_DIR = Path("/your/custom/models/path")
MODEL_SAVE_DIR = MODELS_DIR / "deberta_v3"
```

### Custom Reports Directory

```python
# Change reports output location
REPORTS_DIR = Path("/your/custom/reports/path")
```

---

## Model Configuration

### Hyperparameters

| Parameter | Default | Description | Range |
|-----------|---------|-------------|-------|
| `MODEL_NAME` | `microsoft/deberta-v3-base` | Hugging Face model ID | Any HF model |
| `MAX_LENGTH` | 128 | Maximum token length | 32-512 |
| `BATCH_SIZE` | 16 | Training batch size | 8-64 |
| `EPOCHS` | 3 | Training epochs | 1-10 |
| `LEARNING_RATE` | 2e-5 | Learning rate | 1e-6 to 5e-5 |
| `SHAP_SAMPLES` | 100 | SHAP approximation samples | 50-500 |

### Changing Max Length

```python
# Increase for longer reviews (requires more memory)
MAX_LENGTH = 256

# Decrease for faster inference
MAX_LENGTH = 64
```

### Changing Batch Size

```python
# Larger batch = faster training, more memory
BATCH_SIZE = 32

# Smaller batch = slower training, less memory
BATCH_SIZE = 8
```

---

## Sentiment Mapping

### Default Mapping

```python
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
```

### Custom Mapping

```python
# Example: Binary sentiment (Negative/Positive only)
SENTIMENT_MAPPING = {
    0: "Negative",
    1: "Positive"
}

# Example: Custom labels
SENTIMENT_MAPPING = {
    0: "Dissatisfied",
    1: "Mixed",
    2: "Satisfied"
}
```

**Note**: If you change the mapping, update `NUM_LABELS` accordingly:

```python
NUM_LABELS = len(SENTIMENT_MAPPING)
```

---

## Random Seed

### Purpose

The seed ensures reproducibility across runs:

```python
SEED = 42
```

### What It Controls

- Train/test split randomness
- Model weight initialization
- Dropout layers
- Data shuffling

### Changing the Seed

```python
# Use a different seed for experimentation
SEED = 123
```

---

## Environment Variables

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | All | GPU device selection |
| `TOKENIZERS_PARALLELISM` | true | Enable tokenizer parallelism |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | false | Suppress HF warnings |

### Setting Environment Variables

```bash
# Windows
set CUDA_VISIBLE_DEVICES=0
set TOKENIZERS_PARALLELISM=false

# macOS/Linux
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

---

## Dashboard Configuration

### Streamlit Configuration

Create `.streamlit/config.toml` for custom Streamlit settings:

```toml
[theme]
primaryColor = "#6366f1"
backgroundColor = "#09090b"
secondaryBackgroundColor = "#18181b"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 8501
address = "localhost"

[browser]
gatherUsageStats = false
```

### Custom CSS

The dashboard uses custom CSS defined in `dashboard/app.py`. Key variables:

```css
:root {
    --bg-primary: #09090b;
    --bg-secondary: #18181b;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --positive: #22c55e;
    --negative: #ef4444;
    --neutral: #3b82f6;
}
```

---

## Logging Configuration

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Detailed debugging information |
| `INFO` | General information |
| `WARNING` | Warning messages |
| `ERROR` | Error messages |
| `CRITICAL` | Critical errors |

### Changing Log Level

In `src/utils/logger.py`:

```python
logging.basicConfig(level=logging.INFO)  # Change to DEBUG for more verbosity
```

### Log Files

| File | Purpose |
|------|---------|
| `logs/app.log` | Dashboard application logs |
| `logs/training.log` | Model training logs |

---

## Performance Tuning

### GPU Memory Optimization

```python
# Reduce batch size for limited GPU memory
BATCH_SIZE = 8

# Enable gradient accumulation
# (in trainer.py)
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # Effective batch size = 16
)
```

### CPU Optimization

```python
# Reduce SHAP samples for faster CPU inference
SHAP_SAMPLES = 50

# Use smaller max length
MAX_LENGTH = 64
```

---

## Configuration Examples

### Development Configuration

```python
# Faster training, smaller model
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 1
SHAP_SAMPLES = 50
```

### Production Configuration

```python
# Full training, production quality
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
SHAP_SAMPLES = 100
```

### Research Configuration

```python
# Extended training, more thorough analysis
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 5
SHAP_SAMPLES = 200
```
