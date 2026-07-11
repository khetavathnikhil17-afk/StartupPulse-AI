# Usage Guide

Complete workflow for using StartupPulse AI.

---

## Dashboard Workflow

### 1. Launch the Application

```bash
python -m streamlit run dashboard/app.py
```

### 2. Navigate Pages

| Page | Purpose |
|------|---------|
| **Home** | Project overview, features, quick stats |
| **Analyze Review** | Submit text for sentiment prediction |
| **Explainability** | View SHAP token-level explanations |
| **Model Metrics** | Evaluation results and dataset statistics |
| **About** | Technical documentation and architecture |

### 3. Analyze a Review

1. Navigate to **Analyze Review** in the sidebar
2. Enter employee feedback text (or click an example button)
3. Click **Analyze Sentiment**
4. Wait for the prediction pipeline to complete
5. Review the prediction card with:
   - Sentiment label (Positive/Neutral/Negative)
   - Confidence percentage
   - Three-class probability distribution

### 4. View Explanations

1. After prediction, navigate to **Explainability**
2. View token-level SHAP visualizations:
   - **Waterfall Plot**: Cumulative token contributions
   - **Bar Plot**: Token importance ranking
   - **Interactive HTML**: Color-coded text visualization
3. Examine the Top 10 Influential Tokens table

### 5. Check Model Metrics

1. Navigate to **Model Metrics**
2. Review evaluation metrics (Accuracy, Precision, Recall, F1)
3. Examine the confusion matrix
4. View dataset statistics and class distribution

---

## Python API

### Sentiment Prediction

```python
from src.model.predict import predict_sentiment

# Single prediction
result = predict_sentiment("Your employee review text here")
# Returns:
# {
#     'label': 'Positive',
#     'confidence': 0.9423,
#     'probabilities': {'Positive': 0.9423, 'Neutral': 0.0412, 'Negative': 0.0165}
# }
```

### SHAP Explanation

```python
from src.explainability.shap_explainer import explain_prediction

# Generate explanation
explanation = explain_prediction("Great culture but poor work-life balance.")
# Returns:
# {
#     'label': 'Positive',
#     'confidence': 0.72,
#     'probabilities': {...},
#     'shap_values': <SHAP values object>,
#     'token_importance': {'great': 0.42, 'culture': 0.31, 'but': -0.15, ...},
#     'predicted_class_index': 2
# }
```

### Generate SHAP Plots

```python
from src.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer()
result = explainer.explain_prediction("Your review text")

# Generate all plot types
explainer.generate_plots(result, prefix="my_analysis")
# Saves to reports/shap/:
#   - my_analysis_waterfall.png
#   - my_analysis_bar_summary.png
#   - my_analysis_text.html
```

### Model Evaluation

```bash
# Run full evaluation on test set
python -m src.model.evaluate
```

Generates:
- `reports/classification_report.txt`
- `reports/model_metrics.csv`
- `reports/confusion_matrix.png`

### Data Preparation

```bash
# Prepare dataset from raw data
python -m src.pipeline.prepare_dataset
```

---

## Command-Line Operations

| Command | Description |
|---------|-------------|
| `python -m streamlit run dashboard/app.py` | Launch dashboard |
| `python -m src.model.train` | Train the model |
| `python -m src.model.evaluate` | Evaluate on test set |
| `python -m src.pipeline.prepare_dataset` | Prepare data splits |
| `python test_backend.py` | Run integration tests |

---

## Input Format

The model accepts free-form English text. For best results:

- Use complete sentences
- Include specific feedback (management, compensation, culture, etc.)
- Minimum length: 3-5 words recommended
- Maximum length: 128 tokens (approximately 80-100 words)

### Good Input Examples

```
"The management team is supportive and provides clear career growth paths."
"Work-life balance is terrible. I regularly work 60+ hours with no overtime pay."
"The salary is competitive but the company culture needs improvement."
```

### Poor Input Examples

```
"good"              # Too short
"asdfghjkl"         # Not meaningful text
""                  # Empty string
```

---

## Output Formats

### Prediction Output

```python
{
    "label": str,           # "Positive", "Neutral", or "Negative"
    "confidence": float,    # 0.0 to 1.0
    "probabilities": {
        "Positive": float,
        "Neutral": float,
        "Negative": float
    }
}
```

### SHAP Explanation Output

```python
{
    "label": str,
    "confidence": float,
    "probabilities": dict,
    "shap_values": shap.Explanation,  # SHAP explanation object
    "token_importance": {
        "token": float,  # SHAP value (positive = supports prediction)
        ...
    },
    "predicted_class_index": int  # 0, 1, or 2
}
```
