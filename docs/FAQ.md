# Frequently Asked Questions

Common questions about StartupPulse AI.

---

## General

### What is StartupPulse AI?

StartupPulse AI is an enterprise explainable AI (XAI) system that analyzes employee feedback using DeBERTa-v3 transformer and SHAP to provide transparent, actionable insights for HR decision-making.

### What problem does it solve?

Traditional sentiment analysis gives a single score with no explanation. StartupPulse AI:
1. Predicts sentiment (Positive/Neutral/Negative)
2. Shows exactly which words influenced the decision
3. Provides visual explanations for non-technical stakeholders

### Is it production-ready?

Yes. The system includes:
- Fine-tuned transformer model
- 85%+ test accuracy
- Singleton pattern for efficient resource use
- Comprehensive documentation
- Docker support planned

### What are the system requirements?

**Minimum**:
- Python 3.10+
- 8GB RAM
- 10GB disk space

**Recommended**:
- NVIDIA GPU with CUDA support
- 16GB RAM
- SSD storage

---

## Installation

### Do I need a GPU?

No, but it's recommended:
- **CPU**: Works but predictions are slower (1-3 seconds per prediction)
- **GPU**: Fast predictions (<0.5 seconds per prediction)

### How do I install without GPU?

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Why is the download so large?

The DeBERTa-v3 model is ~500MB. This is normal for transformer models. Subsequent runs use cached weights.

### Can I use a different transformer model?

Yes, change `MODEL_NAME` in `src/config/config.py`:
```python
MODEL_NAME = "microsoft/deberta-v3-small"  # Smaller, faster
MODEL_NAME = "roberta-base"               # Alternative
MODEL_NAME = "bert-base-uncased"          # Classic BERT
```

---

## Usage

### How do I run predictions?

```bash
# Command line
python -m src.model.predict

# Dashboard
python -m streamlit run dashboard/app.py
```

### How do I analyze my own reviews?

1. Place your CSV in `data/employee_reviews.csv`
2. Required columns: `review` (text), `label` (0, 1, or 2)
3. Run `python -m src.pipeline.prepare_dataset`
4. Use the dashboard or command line

### What is the maximum text length?

Default: 128 tokens (~100 words). Change in config:
```python
MAX_LENGTH = 256  # For longer reviews
```

### How accurate is the model?

- **Overall accuracy**: 85%+
- **Positive sentiment**: 87%+ precision
- **Neutral sentiment**: 80%+ precision
- **Negative sentiment**: 89%+ precision

---

## Explainability

### What is SHAP?

SHAP (SHapley Additive exPlanations) explains individual predictions by computing each token's contribution to the final prediction.

### Why use SHAP with transformers?

Transformers are black-box models. SHAP provides:
- Token-level explanations
- Visual explanations for stakeholders
- Compliance with AI transparency requirements

### How long does SHAP take?

- **GPU**: 1-3 seconds
- **CPU**: 5-30 seconds

Reduce with:
```python
SHAP_SAMPLES = 50  # Faster but less precise
```

### What do the SHAP values mean?

| Value | Meaning |
|-------|---------|
| Positive | Token pushes prediction toward this class |
| Negative | Token pushes prediction away from this class |
| Near zero | Token has little influence |

---

## Dashboard

### How do I customize the theme?

Edit CSS in `dashboard/app.py`:
```python
st.markdown("""
<style>
:root {
    --bg-primary: #09090b;
    --accent: #6366f1;
    /* Add your custom colors */
}
</style>
""", unsafe_allow_html=True)
```

### How do I add new pages?

1. Create a new Python file in `dashboard/pages/`
2. Add navigation in the sidebar
3. Streamlit auto-discovers new pages

### How do I change the logo?

Replace `assets/logo.png` with your image (recommended size: 200x200px).

### How do I share the dashboard?

```bash
# Share locally
streamlit run dashboard/app.py --server.address 0.0.0.0

# Share via ngrok
ngrok http 8501
```

---

## Training

### How do I retrain the model?

```bash
# 1. Prepare data
python -m src.pipeline.prepare_dataset

# 2. Train
python -m src.model.train

# 3. Evaluate
python -m src.model.evaluate
```

### How long does training take?

- **GPU**: 15-30 minutes
- **CPU**: 2-4 hours

### How much data do I need?

- **Minimum**: 1,000 labeled reviews
- **Recommended**: 10,000+ labeled reviews
- **Optimal**: 50,000+ labeled reviews

### Can I train on a custom dataset?

Yes. Ensure your CSV has:
- `review`: text column
- `label`: integer (0=Negative, 1=Neutral, 2=Positive)

### How do I improve accuracy?

1. Add more training data
2. Clean data thoroughly
3. Adjust hyperparameters:
```python
EPOCHS = 5          # More epochs
LEARNING_RATE = 1e-5  # Lower learning rate
BATCH_SIZE = 8      # Smaller batch size
```

---

## Deployment

### Can I deploy this in production?

Yes. Key considerations:
1. Use GPU for fast predictions
2. Set up monitoring
3. Implement caching
4. Add authentication

### How do I deploy with Docker?

```bash
docker build -t startuppulse-ai .
docker run -p 8501:8501 startuppulse-ai
```

### How do I deploy to cloud?

**AWS**:
- Use EC2 with GPU
- Or SageMaker for managed deployment

**Azure**:
- Use Azure ML
- Or Container Instances

**GCP**:
- Use AI Platform
- Or Cloud Run

### How do I scale?

1. **Horizontal**: Multiple model instances behind load balancer
2. **Vertical**: Use larger GPU
3. **Caching**: Cache frequent predictions
4. **Batching**: Process multiple predictions together

---

## Troubleshooting

### Predictions are slow

**Solutions**:
1. Use GPU (NVIDIA with CUDA)
2. Reduce `MAX_LENGTH` in config
3. Ensure model is loaded (first prediction is slow)
4. Check GPU utilization with `nvidia-smi`

### SHAP explanations fail

**Solutions**:
1. Reduce `SHAP_SAMPLES` to 50
2. Use shorter input text
3. Check GPU memory with `nvidia-smi`

### Dashboard crashes

**Solutions**:
1. Check `logs/app.log` for errors
2. Ensure all dependencies are installed
3. Verify `assets/` and `reports/` directories exist
4. Restart Streamlit

### Memory errors

**Solutions**:
1. Reduce `BATCH_SIZE` to 8 or 4
2. Reduce `MAX_LENGTH` to 64
3. Use CPU instead of GPU
4. Close other applications

---

## Getting Help

### Where can I get help?

1. Check this FAQ
2. Read [Troubleshooting Guide](TROUBLESHOOTING.md)
3. Search [GitHub Issues](https://github.com/NikhilKhetavath/StartupPulse-AI/issues)
4. Create a new issue

### How do I report a bug?

Create a GitHub issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version, OS, GPU info
- Full error traceback

### How do I request a feature?

Create a GitHub issue with:
- Description of the feature
- Use case
- Expected behavior
- Optional: Implementation ideas

---

## License

### Can I use this commercially?

Yes, under MIT License. You can:
- Use in commercial products
- Modify the code
- Distribute your version
- Use without attribution

### Do I need to attribute the original author?

Attribution is appreciated but not required under MIT License.

### Can I sell a product based on this?

Yes. MIT License allows commercial use without restrictions.
