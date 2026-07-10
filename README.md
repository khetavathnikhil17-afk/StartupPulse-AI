# 🚀 StartupPulse AI

### Explainable AI-Powered Employee Sentiment Analysis using DeBERTa-v3 and SHAP

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)

![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)

![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-success?style=for-the-badge)

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 📖 Overview

StartupPulse AI is an Explainable Artificial Intelligence (XAI) platform designed to analyze employee reviews using a fine-tuned DeBERTa-v3 Transformer model.

Unlike traditional sentiment analysis systems, StartupPulse AI not only predicts whether employee feedback is Positive, Neutral, or Negative, but also explains **why** the prediction was made using SHAP (SHapley Additive Explanations).

The project combines Natural Language Processing (NLP), Deep Learning, Explainable AI, and Interactive Data Visualization into a single platform that helps organizations understand employee sentiment in a transparent and trustworthy manner.

---

# ✨ Key Features

- 🤖 Fine-tuned **DeBERTa-v3** Transformer model for employee sentiment classification.
- 🧠 Explainable AI using **SHAP** to visualize token-level importance.
- 📊 Interactive **Streamlit Dashboard** for real-time sentiment analysis.
- 💬 Predicts **Positive**, **Neutral**, and **Negative** employee sentiment.
- 📈 Model evaluation with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- 🔍 Real-time prediction confidence scores.
- 📁 Modular and production-ready project structure.
- 🎯 Designed specifically for startup employee feedback analysis.

---

# 🏗️ System Architecture

The overall workflow of StartupPulse AI is illustrated below.

<p align="center">
    <img src="assets/architecture.png" width="900">
</p>

### Workflow

1. Employee reviews are collected.
2. Text preprocessing cleans the input.
3. Reviews are tokenized using the DeBERTa-v3 tokenizer.
4. The fine-tuned Transformer predicts sentiment.
5. SHAP explains the prediction.
6. Results are displayed in the Streamlit dashboard.

---

# 📸 Dashboard Preview

## Home Dashboard

<p align="center">
<img src="assets/dashboard_home.png" width="900">
</p>

---

## Prediction Result

<p align="center">
<img src="assets/prediction_result.png" width="900">
</p>

---

## SHAP Explainability

<p align="center">
<img src="assets/shap_explanation.png" width="900">
</p>

---

## Confusion Matrix

<p align="center">
<img src="assets/confusion_matrix.png" width="700">
</p>