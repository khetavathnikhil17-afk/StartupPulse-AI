# Dashboard Guide

Complete guide to the StartupPulse AI Streamlit dashboard.

---

## Overview

The StartupPulse AI dashboard is a 5-page Streamlit application with a premium dark theme designed for non-technical HR stakeholders.

| Property | Value |
|----------|-------|
| Framework | Streamlit 1.33 |
| Pages | 5 |
| Theme | Dark (custom CSS) |
| Port | 8501 (default) |

---

## Launching the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### Launch Options

```bash
# Custom port
python -m streamlit run dashboard/app.py --server.port 8502

# Headless mode (no browser auto-open)
python -m streamlit run dashboard/app.py --server.headless true

# Network access
python -m streamlit run dashboard/app.py --server.address 0.0.0.0
```

---

## Page Navigation

### Sidebar

The sidebar contains:
1. **Logo and branding**
2. **Navigation radio buttons** (5 pages)
3. **Technology badges**
4. **Version information**

### Pages

| Page | Icon | Purpose |
|------|------|---------|
| Home | 🏠 | Project overview, features, quick stats |
| Analyze Review | 🔍 | Submit text for sentiment prediction |
| Explainability | 💡 | View SHAP token-level explanations |
| Model Metrics | 📊 | Evaluation results and dataset statistics |
| About | ℹ️ | Technical documentation and architecture |

---

## Page 1: Home

### Features

- **Hero card** with project title and tagline
- **Feature cards** (4 cards):
  - Transformer Model
  - Real-time Prediction
  - Explainable AI
  - Interactive Dashboard
- **Quick stats** (4 metrics):
  - Total Reviews
  - Sentiment Classes
  - Transformer Layers
  - Hidden Dimensions
- **Project highlights** (8 highlight cards)
- **Model Performance** metrics
- **SHAP preview** with explanation card
- **Quick start tip**

### Layout

```
┌─────────────────────────────────────────┐
│              Hero Card                  │
├───────────────┬─────────────────────────┤
│ Feature Card  │     Feature Card        │
├───────────────┼─────────────────────────┤
│ Feature Card  │     Feature Card        │
├───────────────┴─────────────────────────┤
│         Quick Stats (4 columns)         │
├─────────────────────────────────────────┤
│        Project Highlights (2 cols)      │
├─────────────────────────────────────────┤
│         Model Performance Metrics       │
├─────────────────────────────────────────┤
│           SHAP Preview Section          │
├─────────────────────────────────────────┤
│              Footer                     │
└─────────────────────────────────────────┘
```

---

## Page 2: Analyze Review

### Features

- **Example review buttons** (3 quick demos)
- **Text input area** (160px height)
- **Analyze Sentiment button**
- **Clear button**
- **Multi-stage loading animation** with progress
- **Prediction result card** with:
  - Sentiment badge (color-coded)
  - Confidence percentage
  - Probability distribution bars
  - Source text preview

### Workflow

1. Click an example button or type your own review
2. Click **Analyze Sentiment**
3. Watch the loading animation (5 stages):
   - Loading DeBERTa-v3 model
   - Tokenizing review
   - Running inference
   - Generating SHAP explanations
   - Rendering dashboard
4. View the prediction result card

### Loading Stages

```
Processing Pipeline
├── ● Loading DeBERTa-v3 model...      (20%)
├── ● Tokenizing review...             (50%)
├── ● Running inference...             (75%)
├── ● Generating SHAP explanations...  (90%)
└── ● Rendering dashboard...           (100%)
```

---

## Page 3: Explainability

### Features

- **Analyzed text banner** with sentiment badge
- **SHAP visualization tabs**:
  - Waterfall Plot (PNG)
  - Bar Summary (PNG)
  - Token Impact HTML (interactive)
- **Top 10 Influential Tokens** table with:
  - Token name
  - Impact value (color-coded)
  - Exportable to CSV

### Empty State

When no prediction exists:
```
No prediction available
Run a sentiment analysis to generate explainability visualizations.
Navigate to Analyze Review to get started.
```

---

## Page 4: Model Metrics

### Features

- **Dataset statistics** (4 cards):
  - Total Reviews
  - Training Set
  - Validation Set
  - Test Set
- **Performance metrics** (loaded from CSV):
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- **Average words per review**
- **Class distribution bar chart**
- **Confusion matrix image**
- **Classification report** (expandable)

---

## Page 5: About

### Sections

1. **Project Goal**: Explainable AI for HR analytics
2. **Problem Statement**: Why traditional sentiment analysis fails
3. **Solution**: DeBERTa-v3 + SHAP integration
4. **Key Features**: Detailed feature list
5. **Technology Stack**: Pill badges for all technologies
6. **Model Architecture**: Technical specifications
7. **Explainability**: SHAP methodology explanation
8. **Future Improvements**: Roadmap items
9. **Application Areas**: Use cases for different stakeholders
10. **System Architecture**: Architecture diagram
11. **Links and License**: GitHub, portfolio, MIT license

---

## Custom Styling

### Theme Colors

```css
--bg-primary: #09090b
--bg-secondary: #18181b
--accent: #6366f1
--accent-light: #818cf8
--positive: #22c55e
--negative: #ef4444
--neutral: #3b82f6
```

### CSS Classes

| Class | Purpose |
|-------|---------|
| `.hero-card` | Main hero section |
| `.glass-card` | Glassmorphism cards |
| `.metric-card` | Statistics cards |
| `.result-badge` | Sentiment badges |
| `.prob-bar-*` | Probability bars |
| `.loading-stage` | Loading animation |
| `.sidebar-brand` | Sidebar branding |

---

## Session State

The dashboard uses Streamlit session state for:

| Key | Purpose |
|-----|---------|
| `prediction_result` | Stores latest prediction output |
| `review_text` | Stores input text for persistence |

### State Flow

```
User Input → review_text (session state)
    │
    ▼
Prediction → prediction_result (session state)
    │
    ▼
Explainability Page → Reads prediction_result
```

---

## Responsive Design

### Breakpoints

| Width | Behavior |
|-------|----------|
| > 1200px | Full layout with side-by-side columns |
| 768-1200px | Reduced padding, stacked columns |
| < 768px | Mobile-friendly stacked layout |

### Image Handling

- Images use `use_container_width=True` for responsive sizing
- Confusion matrix displays at 65% width
- Dashboard screenshots at 100% width in tables

---

## Performance Optimization

### Caching

```python
@st.cache_data(show_spinner=False)
def load_metrics():
    # Cached metric loading
    pass

@st.cache_data(show_spinner=False)
def get_dataset_stats():
    # Cached dataset statistics
    pass
```

### Singleton Pattern

Model and SHAP explainer are loaded once and reused:
```python
_predictor = None  # Global singleton

def predict_sentiment(text):
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor.predict(text)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dashboard won't start | Check `pip install streamlit` |
| Port already in use | Use `--server.port 8502` |
| Slow first load | Model downloads on first run (~500MB) |
| SHAP explanation slow | GPU recommended for faster computation |
| Images not loading | Verify `assets/` directory exists |
