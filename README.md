<p align="center">
  <img src="assets/github-banner.png" width="100%" alt="StartupPulse AI Banner">
</p>

<p align="center">
  <img src="assets/logo.png" width="140" alt="StartupPulse AI Logo">
</p>

<h1 align="center">StartupPulse AI</h1>

<p align="center">
  <strong>Explainable Aspect-Based Sentiment Analysis for Employee Feedback</strong>
</p>

<p align="center">
  Enterprise AI platform analyzing employee reviews using DeBERTa-v3 Transformers and SHAP Explainable AI
  to deliver transparent, accurate, and actionable insights for HR decision-making.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-4.40-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  <img src="https://img.shields.io/badge/DeBERTa--v3-Transformer-6366F1?style=for-the-badge" alt="DeBERTa-v3">
  <img src="https://img.shields.io/badge/SHAP-Explainability-FF6B35?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Version-1.2.1-6366F1?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/github/stars/NikhilKhetavath/StartupPulse-AI?style=for-the-badge&color=facc15" alt="GitHub Stars">
  <img src="https://img.shields.io/github/forks/NikhilKhetavath/StartupPulse-AI?style=for-the-badge&color=3b82f6" alt="GitHub Forks">
  <img src="https://img.shields.io/github/issues/NikhilKhetavath/StartupPulse-AI?style=for-the-badge&color=ef4444" alt="GitHub Issues">
  <img src="https://img.shields.io/github/last-commit/NikhilKhetavath/StartupPulse-AI?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/github/repo-size/NikhilKhetavath/StartupPulse-AI?style=for-the-badge" alt="Repo Size">
</p>

<br>

---

## Overview

Employee feedback is one of the most valuable signals a startup can collect — but most organizations lack the infrastructure to analyze it at scale. Manual tagging does not scale. Traditional sentiment analysis misses contextual nuance. And black-box predictions give HR teams no actionable diagnostic.

**StartupPulse AI** solves this by combining a fine-tuned Microsoft DeBERTa-v3 transformer with SHAP (SHapley Additive exPlanations) to deliver sentiment predictions that are both accurate and fully interpretable. Every token in an employee review receives an importance score, making the model's decision-making process transparent and auditable.

The system processes over **66,000 employee reviews** from major technology companies and presents results through an interactive Streamlit dashboard designed for non-technical stakeholders — HR teams, startup founders, and business managers.

### Why Explainable AI Matters

In HR analytics, predictions directly affect people's careers. A sentiment label without explanation carries no diagnostic value. StartupPulse AI provides token-level SHAP explanations that answer not just *what* the model predicts, but *why* — enabling targeted, evidence-backed interventions.

---

## Features

<table>
  <tr>
    <td width="50%" valign="top">

**Sentiment Classification**
Fine-tuned Microsoft DeBERTa-v3 with disentangled attention for context-aware three-class sentiment classification on employee review data.

**SHAP Explainability**
Per-token importance scores with waterfall plots, bar charts, and interactive HTML visualizations for transparent AI decisions.

**Real-time Prediction**
Singleton-loaded model ensures instant inference with confidence scores and full probability distributions.

    </td>
    <td width="50%" valign="top">

**Interactive Dashboard**
Five-page Streamlit application with dark premium theme, designed for non-technical HR stakeholders.

**Model Evaluation**
Automated pipeline generating classification reports, confusion matrices, and weighted F1 metrics.

**Exportable Visualizations**
Downloadable SHAP plots and confusion matrices for stakeholder reporting and presentations.

    </td>
  </tr>
</table>

---

## Architecture

<p align="center">
  <img src="assets/architecture.png" width="85%" alt="System Architecture">
</p>

### Pipeline

```
Employee Review
       |
       v
  [Preprocessing]  ----  Clean, concatenate text fields, deduplicate
       |
       v
  [Tokenizer]  ----  DeBERTa-v3 SentencePiece (128 max length)
       |
       v
  [DeBERTa-v3]  ----  12-layer transformer with disentangled attention
       |
       v
  [Classification Head]  ----  Dropout -> Linear (768 -> 3) -> Softmax
       |
       v
  [Prediction]  ----  Label + Confidence + Probability Distribution
       |
       v
  [SHAP Explainability]  ----  Per-token Shapley values for predicted class
       |
       v
  [Dashboard]  ----  Interactive Streamlit visualization
```

---

## Dashboard Preview

<table>
  <tr>
    <td align="center"><b>Dashboard Home</b></td>
    <td align="center"><b>Prediction Result</b></td>
  </tr>
  <tr>
    <td><img src="assets/dashboard_home.png" width="100%" alt="Dashboard Home"></td>
    <td><img src="assets/prediction_result.png" width="100%" alt="Prediction Result"></td>
  </tr>
</table>

---

## SHAP Explainability

<table>
  <tr>
    <td align="center"><b>Token Impact Visualization</b></td>
    <td align="center"><b>Confusion Matrix</b></td>
  </tr>
  <tr>
    <td><img src="assets/shap_explanation.png" width="100%" alt="SHAP Explanation"></td>
    <td><img src="assets/confusion_matrix.png" width="100%" alt="Confusion Matrix"></td>
  </tr>
</table>

---

## Results

The model was fine-tuned on **66,557 employee reviews** split into training (53,245), validation (6,656), and test (6,656) sets using stratified sampling.

### Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.2% |
| **Weighted F1** | 0.938 |
| **Precision** | 0.939 |
| **Recall** | 0.942 |

### Dataset

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 53,245 | 80% |
| Validation | 6,656 | 10% |
| Test | 6,656 | 10% |
| **Total** | **66,557** | **100%** |

### Labeling Schema

| Rating | Sentiment | Label |
|--------|-----------|-------|
| 1 -- 2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4 -- 5 | Positive | 2 |

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | microsoft/deberta-v3-base |
| Hidden Size | 768 |
| Layers | 12 |
| Attention Heads | 12 |
| Attention Type | Disentangled (content-to-position + position-to-content) |
| Max Sequence Length | 128 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Early Stopping | Patience = 3 |
| Model Selection | Weighted F1 |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip
- Git
- CUDA-capable GPU (optional, recommended for training)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/NikhilKhetavath/StartupPulse-AI.git
cd StartupPulse-AI

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Launch the Dashboard

```bash
python -m streamlit run dashboard/app.py
```

Opens the interactive dashboard at `http://localhost:8501`.

### Train the Model

```bash
python -m src.model.train
```

Fine-tunes DeBERTa-v3 on the employee review dataset. Training requires GPU and takes approximately 30-45 minutes on Google Colab with a T4 GPU.

### Evaluate the Model

```bash
python -m src.model.evaluate
```

Generates classification report, confusion matrix, and metric CSV in `reports/`.

### Run Prediction

```python
from src.model.predict import predict_sentiment

result = predict_sentiment("The management is fantastic and supports growth.")
print(result)
# {'label': 'Positive', 'confidence': 0.94, 'probabilities': {...}}
```

### Generate SHAP Explanations

```python
from src.explainability.shap_explainer import explain_prediction

explanation = explain_prediction("Great culture but poor work-life balance.")
print(explanation["token_importance"])
```

### Run Tests

```bash
python test_backend.py
```

---

## Repository Structure

```
StartupPulse-AI/
|
|-- assets/                         # Branding assets and screenshots
|   |-- logo.png                    # Primary logo
|   |-- logo-dark.png               # Dark variant
|   |-- logo-light.png              # Light variant
|   |-- logo.svg                    # Scalable vector
|   |-- favicon.png                 # Browser icon
|   |-- github-banner.png           # GitHub hero banner
|   |-- architecture.png            # System architecture
|   |-- dashboard_home.png          # Dashboard screenshot
|   |-- prediction_result.png       # Prediction screenshot
|   |-- shap_explanation.png        # SHAP visualization
|   |-- confusion_matrix.png        # Confusion matrix
|   +-- demo.gif                    # Live demo
|
|-- dashboard/
|   +-- app.py                      # Streamlit application (5 pages)
|
|-- src/
|   |-- config/
|   |   +-- config.py               # Paths, hyperparameters, sentiment mapping
|   |-- data/
|   |   |-- preprocessing.py        # Raw data cleaning pipeline
|   |   |-- create_labels.py        # Rating-to-sentiment mapping
|   |   +-- check_dataset.py        # Dataset inspection utility
|   |-- model/
|   |   |-- tokenizer.py            # Tokenizer loading with fallback
|   |   |-- train.py                # Training orchestration
|   |   |-- trainer.py              # Hugging Face TrainingArguments
|   |   |-- predict.py              # SentimentPredictor (singleton)
|   |   +-- evaluate.py             # Evaluation + confusion matrix
|   |-- pipeline/
|   |   |-- dataset.py              # PyTorch Dataset class
|   |   +-- prepare_dataset.py      # Stratified 80/10/10 split
|   |-- explainability/
|   |   +-- shap_explainer.py       # SHAPExplainer class
|   |-- inference/
|   |   +-- inference.py            # Inference wrapper
|   |-- utils/
|   |   |-- logger.py               # Dual-output logger
|   |   |-- metrics.py              # Training metric computation
|   |   |-- seed.py                 # Reproducibility seeding
|   |   +-- helpers.py              # JSON utilities
|   +-- visualization/
|       |-- charts.py               # Distribution plots
|       +-- eda.py                  # EDA report generator
|
|-- models/
|   +-- deberta_v3/                 # Fine-tuned model weights + tokenizer
|
|-- data/
|   |-- employee_reviews.csv        # Raw dataset
|   +-- processed/                  # Cleaned and split datasets
|
|-- reports/
|   |-- classification_report.txt
|   |-- confusion_matrix.png
|   |-- model_metrics.csv
|   |-- figures/                    # EDA visualizations
|   +-- shap/                       # SHAP output files
|
|-- notebooks/                      # Training notebooks
|-- logs/                           # Application and training logs
|-- test_backend.py                 # Integration test suite
|-- requirements.txt
|-- pyproject.toml
+-- LICENSE
```

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.10+ | Core runtime |
| Deep Learning | PyTorch 2.2 | Model framework |
| Transformers | Hugging Face 4.40 | DeBERTa-v3 architecture and tokenization |
| Explainability | SHAP 0.45 | Token-level Shapley value computation |
| Frontend | Streamlit 1.33 | Interactive dashboard |
| Data Processing | Pandas 2.2, NumPy 1.26 | Dataset manipulation and numerical computation |
| Evaluation | scikit-learn 1.4 | Metrics, confusion matrix, classification reports |
| Visualization | Matplotlib 3.8, Seaborn 0.13 | Charts and heatmaps |
| Tokenization | SentencePiece 0.2 | Subword tokenization for DeBERTa-v3 |

---

## Future Improvements

| Area | Description |
|------|-------------|
| **Aspect Extraction** | Move beyond review-level sentiment to per-dimension analysis (management, compensation, growth, culture) |
| **Multi-language Support** | Extend to non-English feedback using DeBERTa multilingual variants |
| **REST API** | Expose prediction and explainability endpoints via FastAPI for HRIS integration |
| **Cloud Deployment** | Containerized deployment on AWS/GCP with Docker and Kubernetes |
| **Continuous Learning** | Feedback loop where analyst corrections retrain the model periodically |
| **Voice Feedback** | Speech-to-text integration for town halls and exit interviews |
| **Enterprise Dashboard** | Role-based views for executives, managers, and HR business partners |
| **Authentication** | OAuth/JWT-based user management with persistent prediction history |

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

---

## Author

**Nikhil Khetavath**

- [GitHub](https://github.com/NikhilKhetavath)
- [LinkedIn](#)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2026 Nikhil Khetavath

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<p align="center">
  <strong>Built with Python, DeBERTa-v3, and SHAP</strong>
</p>

<p align="center">
  <sub>If this project helps you, consider giving it a star on GitHub.</sub>
</p>
