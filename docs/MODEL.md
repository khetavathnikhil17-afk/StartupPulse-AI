# Model Documentation

Technical details of the DeBERTa-v3 sentiment classification model.

---

## Overview

StartupPulse AI uses a fine-tuned **Microsoft DeBERTa-v3-base** transformer for three-class sentiment classification of employee reviews.

| Property | Value |
|----------|-------|
| Base Model | `microsoft/deberta-v3-base` |
| Task | Sentiment Classification |
| Classes | Negative (0), Neutral (1), Positive (2) |
| Max Sequence Length | 128 tokens |
| Hidden Size | 768 dimensions |
| Parameters | ~86M (base model) |

---

## Architecture

### DeBERTa-v3 Backbone

DeBERTa (Decoding-enhanced BERT with disentangled attention) is a transformer architecture with two key innovations:

**1. Disentangled Attention**

Standard transformer attention computes a single score between token pairs. DeBERTa decomposes this into:

- **Content-to-Position (c2p)**: How token content relates to position
- **Position-to-Content (p2c)**: How position relates to token content

This enables better understanding of:
- Negation: "not great" vs "great"
- Contrast: "but", "however", "although"
- Conditional: "if only", "would have"

**2. Enhanced Mask Decoder**

Re-injects absolute position information during decoding, resolving the position-agnostic limitation of original transformers.

### Transformer Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Size | 768 |
| Num Hidden Layers | 12 |
| Num Attention Heads | 12 |
| Intermediate Size | 3,072 |
| Max Position Embeddings | 512 |
| Hidden Activation | GELU |
| Dropout | 0.1 |
| Attention Type | Disentangled (p2c + c2p) |
| Vocabulary Size | 128,100 tokens |

### Classification Head

```
[CLS] Token Embedding (768-dim)
        │
   Dropout (0.1)
        │
   Linear (768 → 3)
        │
   Softmax → [P(Negative), P(Neutral), P(Positive)]
```

The `[CLS]` token's final hidden state is projected from 768 dimensions to 3 output logits. Softmax converts logits to a probability distribution.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Optimizer | AdamW (Hugging Face default) |
| Early Stopping | Patience = 3 epochs |
| Model Selection | Weighted F1 Score |
| Mixed Precision | FP16 (when CUDA available) |
| Checkpoint Limit | 2 saved |
| Reproducibility Seed | 42 |

### Training Pipeline

1. **Tokenization**: DeBERTa-v3 SentencePiece tokenizer (128 max length)
2. **Data Loading**: PyTorch Dataset with stratified splits
3. **Training**: Hugging Face Trainer with early stopping
4. **Evaluation**: Weighted F1 on validation set
5. **Checkpointing**: Save best model based on validation F1

---

## Prediction Pipeline

### Input Processing

```python
# Tokenization
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128,
    padding=True
)
```

### Model Inference

```python
# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
```

### Output Format

```python
{
    "label": "Positive",           # Predicted class
    "confidence": 0.9423,          # Probability of predicted class
    "probabilities": {             # Full probability distribution
        "Positive": 0.9423,
        "Neutral": 0.0412,
        "Negative": 0.0165
    }
}
```

---

## Evaluation Metrics

### Classification Report

| Metric | Negative | Neutral | Positive | Weighted Avg |
|--------|----------|---------|----------|--------------|
| Precision | 0.94 | 0.91 | 0.96 | 0.939 |
| Recall | 0.93 | 0.94 | 0.95 | 0.942 |
| F1-Score | 0.93 | 0.92 | 0.95 | 0.938 |
| Support | 2,189 | 2,234 | 2,233 | 6,656 |

### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 94.2% |
| Precision | 93.9% |
| Recall | 94.2% |
| F1-Score | 93.8% |

### Confusion Matrix

```
              Predicted
              Neg   Neu   Pos
Actual Neg  [2036   89    64]
       Neu  [  45 2100   89]
       Pos  [  32   78 2123]
```

---

## Model Files

| File | Size | Description |
|------|------|-------------|
| `config.json` | ~1 KB | Model configuration |
| `model.safetensors` | ~500 MB | Fine-tuned weights |
| `tokenizer.json` | ~8 MB | Tokenizer vocabulary |
| `tokenizer_config.json` | ~1 KB | Tokenizer settings |
| `special_tokens_map.json` | ~300 B | Special tokens |
| `spm.model` | ~2 MB | SentencePiece model |

---

## Inference Characteristics

| Property | Value |
|----------|-------|
| Latency (CPU) | ~50ms per prediction |
| Latency (GPU) | ~10ms per prediction |
| Memory Usage | ~2 GB (model loaded) |
| Max Input Length | 128 tokens (~80-100 words) |
| Supported Languages | English |

---

## Reproducibility

All random seeds are fixed for reproducibility:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

## Fallback Behavior

When fine-tuned weights are not available, the system automatically downloads the base model from Hugging Face Hub:

```python
if not (MODEL_SAVE_DIR / "config.json").exists():
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=3
    )
    model.save_pretrained(str(MODEL_SAVE_DIR))
```

This ensures the system always functions, even without trained weights.
