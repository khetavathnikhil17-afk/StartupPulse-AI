# StartupPulse AI

### Explainable Aspect-Based Sentiment Analysis for Employee Feedback using DeBERTa-v3 and SHAP

**Turn unstructured employee feedback into actionable, transparent insights -- powered by transformer NLP and explainable AI.**

![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![DeBERTa](https://img.shields.io/badge/Model-DeBERTa--v3-6C63FF)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-FF6B35)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)

| | |
|---|---|
| **Role** | AI Engineer & Full-Stack Developer |
| **Duration** | 8 weeks |
| **Repository** | [GitHub](#) |
| **Live Demo** | [Streamlit App](#) |

---

## Executive Summary

StartupPulse AI is an end-to-end explainable AI platform that transforms raw employee feedback into structured, interpretable sentiment insights. Built on Microsoft's DeBERTa-v3 transformer architecture and augmented with SHAP (SHapley Additive exPlanations), the system classifies employee reviews into Positive, Neutral, and Negative sentiments while providing transparent, token-level explanations for every prediction.

The platform addresses a critical gap in HR analytics: most sentiment analysis tools operate as black boxes, offering predictions without justification. StartupPulse AI combines state-of-the-art NLP with model interpretability, allowing HR teams and startup founders to understand not just *what* the model predicts, but *why* it arrives at that conclusion. Every token in an employee review is assigned an importance score, making the decision-making process fully auditable.

A Streamlit-powered interactive dashboard serves as the front-end interface, enabling non-technical stakeholders to submit reviews, view real-time predictions, explore SHAP-based explanations, and monitor aggregate analytics -- all without writing a single line of code. The system processes over 66,000 employee reviews across major technology companies, demonstrating production-grade capability in a domain where transparency and trust are non-negotiable.

---

## Problem Statement

Employee feedback is one of the most valuable signals a startup can collect. It reveals cultural friction points, leadership blind spots, compensation gaps, and operational bottlenecks long before they escalate into attrition events. Yet most organizations lack the infrastructure to analyze this feedback at scale.

**Manual analysis** does not scale. A single HR analyst can read and categorize a few hundred reviews per day. When feedback volumes reach thousands -- spanning multiple companies, departments, and geographies -- manual tagging becomes a bottleneck that delays action by weeks.

**Traditional sentiment analysis** (lexicon-based or bag-of-words models) treats reviews as flat text bags. These approaches miss contextual nuance: "The management is great but the pay is terrible" receives a single aggregate score rather than separate assessments per aspect. They also struggle with sarcasm, domain-specific language, and negation -- common patterns in employee reviews.

**Black-box predictions** compound the problem. Even when a machine learning model achieves high accuracy, HR stakeholders cannot act on predictions they do not understand. A sentiment label without explanation carries no diagnostic value. Teams need to know which specific phrases or themes drove a prediction to take targeted corrective action.

**Aspect-Based Sentiment Analysis** and **Explainable AI** are not optional features -- they are requirements for responsible deployment in organizational contexts. StartupPulse AI was built to satisfy both demands simultaneously.

---

## Objectives

- Build a transformer-based sentiment classification system fine-tuned on 66,000+ real employee reviews from major technology companies
- Integrate SHAP explainability to generate per-token importance scores for every prediction, ensuring full transparency
- Design a multi-page interactive dashboard that makes AI-driven insights accessible to non-technical HR professionals
- Achieve strong classification performance across Positive, Neutral, and Negative sentiment classes using stratified evaluation
- Implement a production-ready inference pipeline with singleton model loading and real-time prediction latency
- Maintain reproducibility through seeded training, structured configuration management, and version-controlled model artifacts
- Demonstrate end-to-end ML engineering -- from raw data preprocessing through model training, evaluation, explainability, and deployment

---

## Features

### Transformer-Based Sentiment Analysis

Fine-tuned Microsoft DeBERTa-v3-base on domain-specific employee review data. The model captures contextual relationships, negation patterns, and nuanced language that traditional NLP approaches miss.

### Explainable AI with SHAP

Every prediction is accompanied by SHAP values that decompose the model's decision into individual token contributions. Waterfall plots, bar summaries, and interactive HTML visualizations reveal exactly which words influenced the sentiment score.

### Interactive Streamlit Dashboard

A five-page application (Home, Sentiment Analysis, Explainability, Analytics, About) built with a custom dark theme. Users submit text, view predictions, explore explanations, and monitor aggregate metrics -- all within a single interface.

### Prediction Confidence Scores

The system returns a full probability distribution across all three sentiment classes, not just the top prediction. Confidence scores help stakeholders calibrate trust in individual results.

### Real-Time Inference

A singleton-loaded SentimentPredictor class ensures the model loads once and serves predictions on demand. No cold-start delays after the initial load.

### Aggregate Analytics

The Analytics page displays dataset statistics, training/validation/test splits, class distributions, and a confusion matrix heatmap -- giving stakeholders a holistic view of system performance.

### Model Evaluation Pipeline

Automated evaluation generates classification reports, CSV metrics files, and confusion matrix visualizations using scikit-learn. Weighted F1 score drives model selection during training.

### Data Preprocessing Pipeline

A multi-stage pipeline cleans raw employee reviews: drops irrelevant columns, concatenates text fields (summary, pros, cons), removes duplicates, and maps numeric ratings to sentiment labels.

---

## System Architecture

![Architecture](assets/architecture.png)

StartupPulse AI follows a layered architecture with clear separation of concerns:

**Data Layer** -- Raw employee reviews are ingested as CSV files containing company metadata, rating scores, and free-text feedback. The preprocessing module cleans, deduplicates, and concatenates text fields. Labels are derived from numeric ratings using a defined threshold mapping (1-2: Negative, 3: Neutral, 4-5: Positive). A stratified 80/10/10 split produces training, validation, and test datasets.

**Model Layer** -- Microsoft DeBERTa-v3-base is loaded from Hugging Face and fine-tuned with a classification head for three sentiment classes. Training uses the Hugging Face Trainer API with early stopping, weighted F1 model selection, and optional FP16 mixed precision. The fine-tuned model is saved locally with its tokenizer and configuration.

**Explainability Layer** -- SHAP wraps the model's forward pass in a callable prediction function. For each input text, SHAP perturbs tokens and observes output probability changes, computing Shapley values per token for each sentiment class. Three visualization types (waterfall, bar, HTML text) are generated.

**Inference Layer** -- The SentimentPredictor class loads the fine-tuned model once into memory (singleton pattern) and serves predictions. It returns structured output: sentiment label, confidence score, and full probability distribution. The SHAPExplainer operates as a parallel singleton for explanation requests.

**Presentation Layer** -- A Streamlit dashboard communicates with the inference and explainability layers via session state. Custom CSS styles the interface with a dark theme and color-coded sentiment badges. Five pages cover the complete user journey from text input to aggregate analytics.

---

## AI Workflow

```
Employee Review
       |
       v
  [Preprocessing]
  Clean text, concatenate fields, remove noise
       |
       v
  [Tokenizer]
  DeBERTa-v3 SentencePiece tokenization (128 max length)
       |
       v
  [DeBERTa-v3 Model]
  12-layer transformer with disentangled attention
       |
       v
  [Classification Head]
  Dropout -> Linear (768 -> 3) -> Softmax
       |
       v
  [Prediction]
  Label + Confidence + Probability Distribution
       |
       v
  [SHAP Explainability]
  Per-token Shapley values for the predicted class
       |
       v
  [Dashboard]
  Interactive visualization of results
       |
       v
  [Business Insights]
  Actionable feedback for HR teams and founders
```

The pipeline operates as a single pass: input text enters the system, flows through tokenization and model inference, and exits as a structured prediction with full explainability. The SHAP explainer runs as a separate computation on the same model, ensuring that predictions and explanations are always consistent.

---

## Dataset

### Source

Employee reviews from major technology companies including Google, Amazon, Apple, Microsoft, Meta, and others. Each review contains structured metadata (company, location, job title, five sub-ratings) and unstructured text (summary, pros, cons).

### Scale

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 53,245 | 80% |
| Validation | 6,656 | 10% |
| Test | 6,656 | 10% |
| **Total** | **66,557** | **100%** |

### Labeling Strategy

Sentiment labels are derived from the `overall-ratings` column (1-5 scale):

| Rating Range | Sentiment | Label |
|-------------|-----------|-------|
| 1 -- 2 | Negative | 0 |
| 3 | Neutral | 1 |
| 4 -- 5 | Positive | 2 |

### Preprocessing

1. Drop non-textual columns (links, management advice, unnamed indices)
2. Fill missing summary fields with empty strings
3. Concatenate `summary + ". " + pros + ". " + cons` into a unified review field
4. Remove duplicate reviews (exact match on concatenated text)
5. Remove empty or whitespace-only reviews
6. Retain company, location, job title, and rating columns for analytics

### Splitting Strategy

Stratified 80/10/10 split using scikit-learn's `train_test_split` with `random_state=42`. Stratification preserves class distribution across all splits, preventing bias in evaluation metrics when classes are imbalanced.

---

## Model Architecture

### Why DeBERTa-v3

DeBERTa (Decoding-enhanced BERT with disentangled attention) represents a significant architectural advance over BERT and RoBERTa. Three design choices make it particularly well-suited for sentiment analysis on employee feedback:

**Disentangled Attention** -- Standard transformer attention computes a single attention score between each token pair. DeBERTa decomposes this into two components: content-to-position and position-to-content attention. This allows the model to separately learn what a word means and where it appears in the sequence -- critical for understanding negation ("not good"), contrast ("but"), and conditional phrasing ("if only").

**Enhanced Mask Decoder** -- DeBERTa re-injects absolute position information during decoding, resolving the position-agnostic limitation of the original transformer architecture. This improves performance on tasks where word order changes meaning.

**Vocabulary and Pre-training** -- DeBERTa-v3 uses a SentencePiece tokenizer with 128,100 tokens, providing robust coverage of domain-specific terminology found in employee reviews. Pre-training on large-scale corpora gives the model strong general language understanding before fine-tuning.

### Transformer Backbone

| Component | Specification |
|-----------|---------------|
| Hidden Size | 768 |
| Num Hidden Layers | 12 |
| Num Attention Heads | 12 |
| Intermediate Size | 3,072 |
| Max Position Embeddings | 512 |
| Hidden Activation | GELU |
| Dropout | 0.1 |
| Attention Type | Disentangled (p2c + c2p) |

### Classification Head

```
[CLS] Token Embedding (768-dim)
        |
   Dropout (0.1)
        |
   Linear (768 -> 3)
        |
   Softmax -> [P(Negative), P(Neutral), P(Positive)]
```

The `[CLS]` token's final hidden state is projected from 768 dimensions to 3 output logits via a linear layer. Softmax converts logits to a probability distribution. The class with the highest probability becomes the predicted sentiment.

### Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| Optimizer | AdamW (Hugging Face default) |
| Early Stopping Patience | 3 epochs |
| Model Selection Metric | Weighted F1 |
| Mixed Precision | FP16 (when CUDA available) |
| Checkpoint Limit | 2 saved |
| Reproducibility Seed | 42 |

---

## Explainable AI

### SHAP Integration

SHAP (SHapley Additive exPlanations) provides a game-theoretic framework for explaining individual predictions. For each input text, SHAP treats every token as a "player" in a cooperative game and computes the marginal contribution of each token to the final prediction. This produces theoretically grounded, locally faithful explanations.

The `SHAPExplainer` class wraps the fine-tuned DeBERTa-v3 model in a SHAP-compatible prediction function. The function tokenizes input, runs a forward pass, and returns softmax probabilities. SHAP then perturbs input tokens and measures their impact on output probabilities, generating per-token importance scores for each of the three sentiment classes.

### Token Importance

Every token in the input receives a SHAP value for each sentiment class. A positive SHAP value for the "Positive" class means that token increased the likelihood of a positive prediction. A negative value means it decreased it. The magnitude indicates the strength of influence. For example, in the review "great work culture but terrible pay," the tokens "great" and "culture" would receive positive SHAP values for the Positive class, while "terrible" and "pay" would receive negative values.

### Waterfall Plots

Waterfall visualizations show the cumulative journey from the model's base value (average prediction across all inputs) to the final output. Tokens are listed in order of contribution magnitude, with arrows indicating positive or negative impact. This format makes it easy to identify the top 3-5 tokens driving any specific prediction.

### Bar Plots

Bar summaries rank tokens by absolute SHAP value, providing a quick overview of the most influential words. This visualization is particularly useful for identifying consistent patterns across multiple reviews -- for example, whether compensation-related terms consistently drive negative sentiment.

### Interactive HTML Visualization

The text visualization renders the original review with color-coded tokens: green for positive contribution, red for negative, with intensity proportional to SHAP value magnitude. This is embedded directly in the Streamlit dashboard via an HTML component, allowing non-technical users to visually parse explanations without interpreting abstract charts.

### Business Transparency

Explainability transforms the model from a black box into a diagnostic tool. HR teams can trace a negative sentiment prediction back to specific phrases like "no growth opportunities" or "micromanagement," enabling targeted interventions rather than generic feedback programs. This level of transparency is essential for adoption in organizational contexts where model decisions must be auditable.

---

## Dashboard

![Dashboard Home](assets/dashboard_home.png)

![Prediction Result](assets/prediction_result.png)

![SHAP Explanation](assets/shap_explanation.png)

### Page 1: Home

The landing page introduces StartupPulse AI with a project overview, three feature cards (Advanced NLP, XAI Integration, Dynamic Analytics), and a quick-start guide. The custom dark theme (#0E1117 background) creates a professional, modern aesthetic.

### Page 2: Sentiment Analysis

A text area accepts employee review input. The "Analyze Sentiment" button triggers the full prediction pipeline: tokenization, DeBERTa-v3 inference, and SHAP explanation in a single pass. Results display as a color-coded badge (green for Positive, red for Negative, yellow for Neutral), a confidence percentage, and a probability bar chart showing the model's confidence across all three classes.

### Page 3: Explainability

This page loads the most recent prediction from session state and generates three SHAP visualizations: a waterfall plot showing cumulative token contributions, a bar chart ranking token importance, and an interactive HTML text view with color-coded tokens. A table displays the top 10 most influential tokens with their SHAP values, color-coded by contribution direction.

### Page 4: Analytics

Aggregate metrics displayed across four stat cards: total reviews, training set size, validation set size, and test set size. A performance metrics table loaded from the evaluation CSV shows accuracy, precision, recall, and F1 score. The class distribution bar chart and confusion matrix heatmap provide insight into model behavior across sentiment classes.

### Page 5: About

Technical documentation covering the project's architecture stack, sentiment mapping schema, and future development roadmap. Designed as a reference page for stakeholders evaluating the system.

---

## Results

### Prediction Capability

StartupPulse AI successfully classifies employee reviews into three sentiment categories with confidence scores that reflect the model's certainty. The DeBERTa-v3 backbone captures contextual nuances that simpler models miss, including negation handling, contrast detection, and domain-specific terminology common in employee feedback.

### Explainability

Every prediction is accompanied by per-token SHAP values that decompose the model's decision into interpretable components. Waterfall plots, bar charts, and interactive HTML visualizations provide multiple levels of detail for different stakeholder needs -- from quick token-level summaries to deep-dive cumulative analyses.

### Interactive Dashboard

The five-page Streamlit application delivers a complete user experience without requiring technical expertise. Session state management ensures predictions persist across page navigation. Custom CSS theming produces a polished, production-grade interface.

### Business Usability

The system is designed for adoption by non-technical teams. HR managers can paste a review, view the sentiment prediction, explore which words drove the result, and navigate to aggregate analytics -- all within a single session. No code execution, no command-line interfaces, no configuration files.

### Model Evaluation

The evaluation pipeline generates classification reports, confusion matrix heatmaps, and metric CSVs using weighted averaging to handle class imbalance. Early stopping with patience=3 prevents overfitting during training. Stratified data splits ensure representative evaluation across sentiment classes.

---

## Challenges

### Dataset Preparation

The raw employee review dataset required extensive cleaning: dropping irrelevant columns, handling missing values in summary fields, concatenating multiple text fields into a unified review, removing duplicates, and mapping numeric ratings to categorical sentiment labels. Column name inconsistencies (e.g., "carrer-opportunities-stars") and mixed data types required careful handling during preprocessing.

### Transformer Integration

Fine-tuning a 12-layer transformer on a domain-specific dataset required careful hyperparameter selection. Learning rate too high caused catastrophic forgetting of pre-trained representations; too low produced underfitting within the 3-epoch budget. The 128-token max length required truncation decisions that occasionally removed meaningful review context.

### SHAP Performance

SHAP explainability introduces significant computational overhead. Each explanation requires multiple model forward passes with perturbed inputs, resulting in approximately 20 seconds per explanation on CPU. This latency required careful UX design -- the dashboard uses spinners and session state caching to maintain a responsive feel despite the underlying computation.

### Dashboard Integration

Integrating SHAP visualizations into Streamlit required workarounds for file handling conflicts. PIL's `Image.open()` is used to load images into memory before passing them to `st.image()`, avoiding Streamlit's MediaFileHandler conflicts. Interactive HTML visualizations are embedded via `st.components.html()` to preserve color-coded formatting.

### Model Deployment

The fine-tuned model weights (~500 MB) are excluded from Git via `.gitignore` to keep the repository lightweight. This required implementing fallback logic in both the predictor and SHAP explainer: when local weights are unavailable, the system downloads the base model from Hugging Face Hub as a graceful degradation path.

---

## Future Improvements

### Aspect Extraction

Move beyond review-level sentiment to aspect-level analysis. Extract specific dimensions (management quality, compensation, work-life balance, career growth) and assign independent sentiment scores to each. This transforms a single prediction into a multi-dimensional diagnostic.

### Multi-Language Support

Extend the system to handle employee feedback in languages beyond English. DeBERTa's multilingual variants and cross-lingual transfer learning make this architecturally feasible without retraining from scratch.

### Voice Feedback Analysis

Integrate speech-to-text transcription to enable analysis of verbal feedback from town halls, exit interviews, and pulse surveys. This expands the input modality beyond written text.

### Enterprise Dashboard

Build role-based dashboards for different organizational levels: executive summaries for C-suite, department-level trends for managers, and team-specific insights for HR business partners. Add filtering by time period, department, and sentiment trajectory.

### Cloud Deployment

Deploy the system on AWS, GCP, or Azure with containerized infrastructure (Docker + Kubernetes). This enables horizontal scaling, auto-scaling based on traffic, and geographic distribution for global organizations.

### REST API

Expose prediction and explainability endpoints via FastAPI or Flask. This enables integration with existing HRIS platforms, Slack bots, and automated reporting pipelines.

### Authentication & Database Integration

Add user authentication (OAuth, JWT) and a persistent database (PostgreSQL, MongoDB) to store prediction history, track sentiment trends over time, and enable longitudinal analysis.

### Continuous Learning

Implement a feedback loop where HR analysts can correct misclassifications. These corrections feed back into the training pipeline for periodic model retraining, improving accuracy on domain-specific language over time.

---

## Key Learnings

- Fine-tuning a pre-trained transformer requires balancing learning rate, epochs, and early stopping to avoid both underfitting and catastrophic forgetting of pre-trained representations
- SHAP explainability, while powerful, introduces non-trivial computational overhead that must be accounted for in system design and user experience
- Singleton model loading patterns are essential for serving real-time predictions in Python-based applications without repeated initialization overhead
- Stratified data splitting is critical for evaluation integrity when class distributions are imbalanced, as is common in real-world employee feedback datasets
- Streamlit provides rapid prototyping for ML dashboards but requires careful state management and file handling workarounds for production-quality applications
- Disentangled attention mechanisms in DeBERTa capture positional and contextual relationships that standard transformer attention misses, particularly relevant for negation-heavy sentiment text
- End-to-end ML engineering extends far beyond model training: data cleaning, preprocessing, evaluation, explainability, and presentation each require dedicated engineering effort
- Weighted F1 score is a more informative evaluation metric than raw accuracy when classes are imbalanced, as it accounts for per-class performance variation
- Model weight management in Git repositories requires deliberate .gitignore configuration and fallback loading strategies for collaborative development
- Custom CSS theming in Streamlit transforms a functional prototype into a polished interface suitable for stakeholder demonstrations
- Tokenizer selection impacts model performance: DeBERTa-v3's SentencePiece tokenizer handles domain-specific terminology and subword patterns more effectively than BERT's WordPiece
- Session state in Streamlit enables multi-page application flows without backend infrastructure, but requires careful initialization to avoid stale state bugs
- End-to-end reproducibility requires seeding all random number generators (Python, NumPy, PyTorch, hash seed) and maintaining strict configuration management
- SHAP waterfall plots are the most intuitive explanation format for non-technical stakeholders, while bar plots serve analytical use cases requiring token-level ranking

---

## Project Impact

StartupPulse AI demonstrates how explainable AI can bridge the gap between machine learning capability and organizational adoption. In the HR technology space, where decisions directly affect people's careers and livelihoods, black-box predictions are not acceptable. Every sentiment classification must be traceable, interpretable, and auditable.

By combining DeBERTa-v3's contextual understanding with SHAP's theoretical rigor in explainability, StartupPulse AI provides a blueprint for responsible AI deployment in sensitive domains. HR teams gain the ability to identify systemic issues in employee experience -- toxic management patterns, compensation inequities, career stagnation -- with evidence-backed explanations that support concrete action.

For startups operating with limited HR headcount, this system offers scalable feedback analysis that would otherwise require a dedicated people analytics team. The interactive dashboard removes the technical barrier, making AI-driven insights accessible to founders, managers, and operators who need to act on employee sentiment quickly.

---

## Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| Deep Learning | Transformer architectures, fine-tuning, classification heads, dropout regularization |
| Transformers | DeBERTa-v3, Hugging Face Transformers, AutoModelForSequenceClassification, Trainer API |
| NLP | Tokenization, sequence classification, text preprocessing, subword encoding |
| Explainable AI | SHAP values, Shapley theory, token importance, waterfall/bar/text visualizations |
| PyTorch | Tensor operations, GPU acceleration, FP16 mixed precision, model serialization |
| Python | Object-oriented design, singleton patterns, type hints, error handling |
| Data Science | Pandas, NumPy, scikit-learn, stratified splitting, confusion matrices |
| Data Visualization | Matplotlib, Seaborn, Streamlit charts, color-coded HTML embeddings |
| Streamlit | Multi-page apps, session state, custom CSS, component embedding |
| Git | Repository management, .gitignore configuration, commit hygiene |
| Software Engineering | Configuration management, logging, modular architecture, reproducibility |

---

## Repository Structure

```
StartupPulse-AI/
|
|-- configs/                        # Configuration files
|-- dashboard/
|   +-- app.py                      # Streamlit application (5 pages)
|-- data/
|   |-- employee_reviews.csv        # Raw review dataset
|   |-- train.csv                   # Training split
|   |-- validation.csv              # Validation split
|   |-- test.csv                    # Test split
|   +-- processed/
|       |-- clean_employee_reviews.csv
|       |-- labeled_employee_reviews.csv
|       |-- train.csv               # Full training set (53,245 rows)
|       |-- validation.csv          # Full validation set (6,656 rows)
|       +-- test.csv                # Full test set (6,656 rows)
|-- docs/
|   +-- architecture.md             # Mermaid architecture diagram
|-- logs/
|   |-- app.log                     # Application runtime logs
|   +-- training.log                # Training session logs
|-- models/
|   +-- deberta_v3/
|       |-- config.json             # Model configuration
|       |-- model.safetensors       # Fine-tuned weights (gitignored)
|       |-- tokenizer.json          # Tokenizer vocabulary
|       |-- tokenizer_config.json   # Tokenizer settings
|       |-- special_tokens_map.json
|       +-- spm.model               # SentencePiece model
|-- notebooks/
|   +-- train_deberta_colab.ipynb   # Colab training notebook
|-- outputs/
|   |-- plots/                      # Generated plots
|   |-- reports/                    # Generated reports
|   +-- shap/                       # SHAP output files
|-- reports/
|   |-- classification_report.txt   # Evaluation report
|   |-- confusion_matrix.png        # Confusion matrix heatmap
|   |-- model_metrics.csv           # Metric summary
|   |-- figures/                    # EDA visualizations
|   +-- shap/                       # SHAP visualizations
|-- src/
|   |-- config/
|   |   +-- config.py               # Central configuration (paths, hyperparams)
|   |-- data/
|   |   |-- preprocessing.py        # Raw data cleaning
|   |   |-- create_labels.py        # Rating-to-sentiment mapping
|   |   +-- check_dataset.py        # Dataset inspection utility
|   |-- model/
|   |   |-- tokenizer.py            # Tokenizer loading
|   |   |-- train.py                # Training orchestration
|   |   |-- trainer.py              # Training arguments
|   |   |-- predict.py              # SentimentPredictor class
|   |   +-- evaluate.py             # Evaluation + confusion matrix
|   |-- pipeline/
|   |   |-- dataset.py              # PyTorch Dataset class
|   |   +-- prepare_dataset.py      # Stratified data splitting
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
|-- tests/
|   +-- test_backend.py             # Integration test suite
|-- assets/
|   |-- architecture.png
|   |-- dashboard_home.png
|   |-- prediction_result.png
|   |-- shap_explanation.png
|   |-- confusion_matrix.png
|   +-- demo.gif
|-- requirements.txt
|-- pyproject.toml
|-- LICENSE                         # MIT License
+-- README.md
```

---

## Screenshots

![Dashboard Home](assets/dashboard_home.png)

![Prediction Result](assets/prediction_result.png)

![SHAP Explanation](assets/shap_explanation.png)

![Confusion Matrix](assets/confusion_matrix.png)

---

## Conclusion

StartupPulse AI is a production-ready AI system that combines state-of-the-art NLP, transparent explainability, and interactive analytics into a cohesive platform for employee feedback analysis. Built on Microsoft DeBERTa-v3 with SHAP integration and a Streamlit dashboard, it demonstrates end-to-end ML engineering from data preprocessing through deployment.

The system addresses a real organizational need: understanding employee sentiment at scale while maintaining full transparency into model decisions. Every prediction includes token-level explanations that make the AI auditable -- a requirement, not a luxury, in HR technology.

StartupPulse AI is not a proof of concept. It is a working system trained on 66,000+ real employee reviews, packaged with a polished dashboard, and built with production patterns (singleton loading, session state management, fallback model loading, reproducible training). It represents the kind of engineering rigor that separates demo projects from deployable AI products.

---

*Built by Nikhil Khetavath*
*MIT License -- 2026*
