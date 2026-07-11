# SHAP Explainability

Understanding token-level explanations for sentiment predictions.

---

## What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a game-theoretic framework for explaining individual predictions. It computes the marginal contribution of each feature (token) to the final prediction.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Shapley Values** | Fair distribution of prediction among tokens |
| **Base Value** | Average model output across all inputs |
| **SHAP Value** | Token's contribution to deviation from base value |
| **Expected Value** | Weighted average of SHAP values |

### Why SHAP for NLP?

- **Theoretically grounded**: Based on cooperative game theory
- **Per-token granularity**: Explains individual word contributions
- **Consistent**: Satisfies efficiency, symmetry, dummy, and additivity axioms
- **Local and global**: Works for single predictions and model-wide patterns

---

## SHAP Integration

### How It Works

1. **Tokenization**: Input text is tokenized using DeBERTa-v3 tokenizer
2. **Perturbation**: SHAP creates multiple modified inputs by masking tokens
3. **Inference**: Each perturbed input is passed through the model
4. **Computation**: SHAP values are computed by comparing outputs
5. **Aggregation**: Token importance scores are generated for each class

### Technical Implementation

```python
class SHAPExplainer:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
        self.explainer = shap.Explainer(self._predict_func, self.tokenizer)
    
    def _predict_func(self, texts):
        # Tokenize and run inference
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()
```

---

## Explanation Types

### 1. Waterfall Plot

Shows the cumulative journey from base value to final prediction.

```
Base Value (E[f(x)]): 0.33
├── "great"      +0.42  ████████████
├── "culture"    +0.31  ████████
├── "but"        -0.15  ████
├── "poor"       -0.28  ███████
├── "work-life"  -0.19  █████
└── Final Output: 0.72  (Positive)
```

**Interpretation**:
- Tokens above the baseline push toward the predicted class
- Tokens below push away from it
- Length of bar indicates magnitude of influence

### 2. Bar Plot (Summary)

Ranks tokens by absolute SHAP value.

```
Token Importance:
"great"     ████████████████  0.42
"culture"   ████████████      0.31
"poor"      ████████          0.28
"work-life" ██████            0.19
"but"       █████             0.15
```

**Use Case**: Identifying the most influential words across multiple predictions.

### 3. Interactive HTML

Color-coded inline text visualization:

```html
<span style="background-color: rgba(34,197,94,0.4)">great</span>
<span style="background-color: rgba(34,197,94,0.3)">culture</span>
<span style="background-color: rgba(239,68,68,0.2)">but</span>
<span style="background-color: rgba(239,68,68,0.35)">poor</span>
<span style="background-color: rgba(239,68,68,0.25)">work-life</span>
```

**Color Coding**:
- **Green**: Positive contribution to predicted class
- **Red**: Negative contribution to predicted class
- **Intensity**: Proportional to SHAP value magnitude

### 4. Token Importance Table

```python
{
    "great": 0.42,
    "culture": 0.31,
    "supportive": 0.24,
    "but": -0.15,
    "poor": -0.28,
    "work-life": -0.19,
    "balance": -0.12,
    "terrible": -0.35,
    "management": 0.18,
    "growth": 0.22
}
```

---

## Using SHAP Explainer

### Python API

```python
from src.explainability.shap_explainer import explain_prediction, SHAPExplainer

# Method 1: Simple function call
result = explain_prediction("Great culture but poor work-life balance.")

# Method 2: Full explainer class
explainer = SHAPExplainer()
result = explainer.explain_prediction("Your review text here")

# Generate visualizations
explainer.generate_plots(result, prefix="my_analysis")
```

### Output Structure

```python
{
    "label": "Positive",              # Predicted sentiment
    "confidence": 0.72,               # Prediction confidence
    "probabilities": {                # Full probability distribution
        "Positive": 0.72,
        "Neutral": 0.21,
        "Negative": 0.07
    },
    "shap_values": <Explanation>,     # SHAP Explanation object
    "token_importance": {             # Token → SHAP value mapping
        "great": 0.42,
        "culture": 0.31,
        ...
    },
    "predicted_class_index": 2        # Index of predicted class
}
```

---

## Dashboard Integration

### SHAP Page Features

1. **Source Text Banner**: Displays the analyzed review
2. **Prediction Badge**: Shows predicted sentiment with confidence
3. **Waterfall Plot**: Cumulative token contributions
4. **Bar Summary**: Token importance ranking
5. **Interactive HTML**: Color-coded text visualization
6. **Top 10 Tokens Table**: Tabular view with exact SHAP values

### Export Capabilities

All SHAP visualizations can be downloaded:
- Waterfall plots as PNG
- Bar summaries as PNG
- Text visualizations as HTML

---

## Interpretation Guide

### Reading SHAP Values

| SHAP Value | Meaning |
|------------|---------|
| **Positive (> 0)** | Token supports the predicted class |
| **Negative (< 0)** | Token opposes the predicted class |
| **Zero (≈ 0)** | Token has negligible influence |
| **Large magnitude** | Strong influence on prediction |

### Example Interpretation

**Input**: "The management is fantastic but the pay is terrible"

**SHAP Values**:
- `"fantastic"`: +0.45 (strong positive)
- `"management"`: +0.22 (moderate positive)
- `"but"`: -0.12 (slight negative, contrast marker)
- `"terrible"`: -0.52 (strong negative)
- `"pay"`: -0.18 (moderate negative)

**Result**: Positive prediction (0.62 confidence) because positive tokens outweigh negative ones.

### Common Patterns

| Pattern | Example | SHAP Behavior |
|---------|---------|---------------|
| Negation | "not good" | "not" flips "good" contribution |
| Contrast | "great but terrible" | Both tokens have opposing values |
| Intensifiers | "very poor" | "very" amplifies "poor" |
| Domain terms | "work-life balance" | Compound phrase treated as unit |

---

## Performance Considerations

| Factor | Impact |
|--------|--------|
| **CPU Inference** | ~20 seconds per explanation |
| **GPU Inference** | ~3-5 seconds per explanation |
| **Text Length** | Longer texts take more time |
| **SHAP Samples** | More samples = more accurate but slower |

### Optimization Tips

1. **Cache explanations**: Use session state in Streamlit
2. **Limit text length**: Keep inputs under 100 words
3. **Use GPU**: Significant speedup for SHAP computations
4. **Batch processing**: Explain multiple texts together when possible

---

## Limitations

1. **Computational Cost**: SHAP requires multiple model forward passes
2. **Approximation**: SHAP values are approximations, not exact
3. **Tokenization Dependency**: Explanation quality depends on tokenization
4. **Context Window**: Limited to 128 tokens (model constraint)
5. **English Only**: Currently supports English text only
