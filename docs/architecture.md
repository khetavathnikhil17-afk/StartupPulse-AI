# Project Architecture

Technical architecture and system design of StartupPulse AI.

---

## System Overview

StartupPulse AI follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Presentation Layer                           │
│              Streamlit Dashboard (5 pages)                      │
├─────────────────────────────────────────────────────────────────┤
│                    Inference Layer                              │
│         SentimentPredictor (singleton) + SHAPExplainer          │
├─────────────────────────────────────────────────────────────────┤
│                    Model Layer                                  │
│           DeBERTa-v3 + Classification Head                     │
├─────────────────────────────────────────────────────────────────┤
│                    Explainability Layer                         │
│              SHAP Token-level Analysis                         │
├─────────────────────────────────────────────────────────────────┤
│                    Data Layer                                   │
│         Preprocessing + Stratified Splitting                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

```
Employee Review (Input)
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  Clean, concatenate text fields, deduplicate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenizer     │  DeBERTa-v3 SentencePiece (128 max length)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   DeBERTa-v3    │  12-layer transformer with disentangled attention
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │  Dropout → Linear (768 → 3) → Softmax
│     Head        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Prediction    │  Label + Confidence + Probability Distribution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SHAP Explainer │  Per-token Shapley values for predicted class
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Dashboard     │  Interactive Streamlit visualization
└─────────────────┘
```

---

## Directory Structure

```
StartupPulse-AI/
├── assets/                    # Branding assets and screenshots
├── dashboard/
│   └── app.py                 # Streamlit application (5 pages)
├── src/
│   ├── config/
│   │   └── config.py          # Central configuration
│   ├── data/
│   │   ├── preprocessing.py   # Raw data cleaning
│   │   ├── create_labels.py   # Rating-to-sentiment mapping
│   │   └── check_dataset.py   # Dataset inspection
│   ├── model/
│   │   ├── tokenizer.py       # Tokenizer loading
│   │   ├── train.py           # Training orchestration
│   │   ├── trainer.py         # Hugging Face TrainingArguments
│   │   ├── predict.py         # SentimentPredictor class
│   │   └── evaluate.py        # Evaluation + confusion matrix
│   ├── pipeline/
│   │   ├── dataset.py         # PyTorch Dataset class
│   │   └── prepare_dataset.py # Stratified data splitting
│   ├── explainability/
│   │   └── shap_explainer.py  # SHAPExplainer class
│   ├── inference/
│   │   └── inference.py       # Inference wrapper
│   ├── utils/
│   │   ├── logger.py          # Dual-output logger
│   │   ├── metrics.py         # Training metric computation
│   │   ├── seed.py            # Reproducibility seeding
│   │   └── helpers.py         # JSON utilities
│   └── visualization/
│       ├── charts.py          # Distribution plots
│       └── eda.py             # EDA report generator
├── models/
│   └── deberta_v3/            # Model weights + tokenizer
├── data/
│   ├── employee_reviews.csv   # Raw dataset
│   └── processed/             # Cleaned and split datasets
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── model_metrics.csv
│   ├── figures/               # EDA visualizations
│   └── shap/                  # SHAP output files
├── logs/                      # Application and training logs
├── test_backend.py            # Integration test suite
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Module Responsibilities

### Configuration (`src/config/config.py`)

Central configuration file managing:
- Project paths (data, models, reports, logs)
- Model hyperparameters (learning rate, batch size, epochs)
- Sentiment label mapping
- Random seed for reproducibility

### Data Layer (`src/data/`, `src/pipeline/`)

| Module | Responsibility |
|--------|----------------|
| `preprocessing.py` | Cleans raw CSV: drops columns, fills NaN, concatenates text fields, removes duplicates |
| `create_labels.py` | Maps numeric ratings (1-5) to sentiment labels (Negative/Neutral/Positive) |
| `prepare_dataset.py` | Performs stratified 80/10/10 train/validation/test split |
| `dataset.py` | PyTorch Dataset class for loading tokenized data |

### Model Layer (`src/model/`)

| Module | Responsibility |
|--------|----------------|
| `tokenizer.py` | Loads DeBERTa-v3 SentencePiece tokenizer with fallback |
| `train.py` | Orchestrates training loop with Hugging Face Trainer |
| `trainer.py` | Configures TrainingArguments (lr, batch size, early stopping) |
| `predict.py` | Singleton SentimentPredictor for real-time inference |
| `evaluate.py` | Generates classification report, confusion matrix, metrics CSV |

### Explainability Layer (`src/explainability/`)

| Module | Responsibility |
|--------|----------------|
| `shap_explainer.py` | SHAPExplainer class: wraps model in SHAP-compatible function, generates token importance scores, produces waterfall/bar/text visualizations |

### Presentation Layer (`dashboard/`)

| Module | Responsibility |
|--------|----------------|
| `app.py` | 5-page Streamlit dashboard with dark theme, session state management, real-time prediction |

### Utilities (`src/utils/`)

| Module | Responsibility |
|--------|----------------|
| `logger.py` | Dual-output logger (console + file) |
| `metrics.py` | Computes accuracy, precision, recall, F1 for training |
| `seed.py` | Seeds Python, NumPy, PyTorch for reproducibility |
| `helpers.py` | JSON serialization utilities |

---

## Design Patterns

### Singleton Pattern

The `SentimentPredictor` and `SHAPExplainer` classes use singleton instances to avoid repeated model loading:

```python
_predictor = None

def predict_sentiment(text: str):
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor.predict(text)
```

**Benefit**: Model loads once into memory, subsequent predictions are fast.

### Fallback Loading

When fine-tuned weights are unavailable, the system downloads the base model from Hugging Face Hub:

```python
if not (MODEL_SAVE_DIR / "config.json").exists():
    logger.warning("Model not found. Downloading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(str(MODEL_SAVE_DIR))
```

### Stratified Splitting

Data is split using stratified sampling to maintain class distribution:

```python
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
```

---

## Data Flow

### Training Flow

```
Raw CSV → Preprocessing → Label Mapping → Stratified Split → Tokenization → Training → Model Weights
```

### Inference Flow

```
Text Input → Tokenization → Model Forward Pass → Softmax → Prediction Output
```

### Explanation Flow

```
Text Input → SHAP Perturbation → Multiple Forward Passes → Token Importance Scores → Visualizations
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Model | DeBERTa-v3-base | Transformer backbone |
| Training | Hugging Face Trainer | Training loop management |
| Explainability | SHAP | Token-level Shapley values |
| Frontend | Streamlit | Interactive dashboard |
| Backend | Python 3.10+ | Runtime environment |
| Deep Learning | PyTorch 2.2 | Neural network framework |
| Evaluation | scikit-learn | Metrics and confusion matrix |
| Visualization | Matplotlib/Seaborn | Charts and plots |
