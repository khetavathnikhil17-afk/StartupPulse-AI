# Quick Start

Get StartupPulse AI running in under 5 minutes.

---

## One-Minute Setup

```bash
# Clone and enter directory
git clone https://github.com/NikhilKhetavath/StartupPulse-AI.git
cd StartupPulse-AI

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Launch the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Opens at `http://localhost:8501` with the interactive dashboard.

---

## Your First Prediction

### Via Dashboard

1. Open `http://localhost:8501`
2. Click **Analyze Review** in the sidebar
3. Type or paste an employee review
4. Click **Analyze Sentiment**
5. View the prediction, confidence scores, and SHAP explanation

### Via Python

```python
from src.model.predict import predict_sentiment

result = predict_sentiment("The management is fantastic and supports growth.")
print(result)
# {'label': 'Positive', 'confidence': 0.94, 'probabilities': {...}}
```

### Via SHAP Explanation

```python
from src.explainability.shap_explainer import explain_prediction

explanation = explain_prediction("Great culture but poor work-life balance.")
print(explanation["label"])           # 'Positive'
print(explanation["token_importance"]) # {'great': 0.42, 'culture': 0.31, 'poor': -0.28, ...}
```

---

## Example Reviews to Try

| Sentiment | Example |
|-----------|---------|
| **Positive** | "The company provides excellent work-life balance, supportive managers, and great career growth opportunities." |
| **Neutral** | "The salary is average. The workload is manageable but there is little room for growth." |
| **Negative** | "Management ignores employee concerns. Compensation is poor and the work environment is stressful." |

---

## What's Next?

- [Usage Guide](USAGE.md) - Complete workflow documentation
- [Dashboard Guide](DASHBOARD.md) - Dashboard features and navigation
- [SHAP Explanation](SHAP.md) - Understanding explainability outputs
- [Model Documentation](MODEL.md) - Model architecture and training
