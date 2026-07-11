# Troubleshooting

Solutions for common issues with StartupPulse AI.

---

## Installation Issues

### pip install fails

**Symptom**: `pip install -r requirements.txt` fails with errors.

**Solutions**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install with verbose output
pip install -r requirements.txt -v

# Install packages individually
pip install torch==2.2.1
pip install transformers==4.40.1
pip install shap==0.45.0
pip install streamlit==1.33.0
```

### CUDA not available

**Symptom**: `torch.cuda.is_available()` returns `False`.

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Import errors

**Symptom**: `ModuleNotFoundError: No module named 'src'`

**Solutions**:
```bash
# Run from project root
cd StartupPulse-AI

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use module syntax
python -m src.model.predict
```

---

## Model Issues

### Model not found

**Symptom**: `WARNING: Model not found at models/deberta_v3`

**Cause**: Fine-tuned model weights not present.

**Solution**: The system automatically downloads the base model. Wait for download to complete (~500MB).

```bash
# Manual download
python -c "
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=3)
model.save_pretrained('models/deberta_v3')
"
```

### CUDA out of memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size in src/config/config.py
BATCH_SIZE = 8  # or 4

# Reduce max sequence length
MAX_LENGTH = 64

# Use CPU instead
# Set device to CPU in predict.py
self.device = torch.device("cpu")
```

### Slow predictions

**Symptom**: Predictions take > 1 second.

**Solutions**:
1. Ensure GPU is available and CUDA is installed
2. First prediction is slow (model loading), subsequent predictions are fast
3. Check if model is already loaded (singleton pattern)

---

## SHAP Issues

### SHAP explanation fails

**Symptom**: `RuntimeError: Failed to generate SHAP explanation`

**Solutions**:
```python
# Reduce SHAP samples
SHAP_SAMPLES = 50  # in config.py

# Use shorter text
text = "Your review text here"  # Keep under 100 words

# Check model is loaded
from src.model.predict import predict_sentiment
result = predict_sentiment("Test")  # Verify model works first
```

### SHAP is slow

**Symptom**: SHAP explanations take > 30 seconds.

**Causes**: SHAP requires multiple model forward passes.

**Solutions**:
1. Use GPU for faster computation
2. Reduce `SHAP_SAMPLES` in config
3. Cache explanations in session state
4. Use shorter input text

### SHAP plots not saving

**Symptom**: `reports/shap/` directory is empty.

**Solutions**:
```bash
# Create directory manually
mkdir -p reports/shap

# Check write permissions
ls -la reports/

# Run SHAP manually
python -m src.explainability.shap_explainer
```

---

## Dashboard Issues

### Streamlit won't start

**Symptom**: `python -m streamlit run dashboard/app.py` fails.

**Solutions**:
```bash
# Reinstall Streamlit
pip install --force-reinstall streamlit==1.33.0

# Check port availability
netstat -an | grep 8501

# Use different port
python -m streamlit run dashboard/app.py --server.port 8502
```

### Dashboard shows errors

**Symptom**: Error messages in the dashboard.

**Solutions**:
1. Check `logs/app.log` for detailed errors
2. Ensure all dependencies are installed
3. Verify `assets/` directory exists with images
4. Check `reports/` directory exists

### Images not loading

**Symptom**: Broken image icons in dashboard.

**Solutions**:
```bash
# Verify assets exist
ls -la assets/

# Required files:
# assets/logo.png
# assets/architecture.png
# assets/dashboard_home.png
# assets/prediction_result.png
# assets/shap_explanation.png
# assets/confusion_matrix.png
```

### Session state issues

**Symptom**: Predictions not persisting between pages.

**Solution**: This is expected behavior. Session state is per-session. Navigate to **Analyze Review** to run a new prediction.

---

## Data Issues

### Dataset not found

**Symptom**: `WARNING: Raw data file not found`

**Solution**: The system creates dummy data for testing. For real data:

```bash
# Place your CSV in data/
cp /path/to/your/data.csv data/employee_reviews.csv

# CSV must have columns:
# - review (text)
# - label (0, 1, or 2)
```

### Invalid CSV format

**Symptom**: `KeyError: 'review'` or `KeyError: 'label'`

**Solution**: Ensure your CSV has the required columns:

```python
import pandas as pd
df = pd.read_csv("data/employee_reviews.csv")
print(df.columns)  # Should include 'review' and 'label'
```

---

## Training Issues

### Training fails to start

**Symptom**: `python -m src.model.train` fails.

**Solutions**:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Ensure data exists
python -m src.pipeline.prepare_dataset

# Check disk space
df -h  # Linux/macOS
dir   # Windows
```

### Training is too slow

**Symptom**: Training takes > 1 hour.

**Solutions**:
1. Use GPU (NVIDIA with CUDA)
2. Reduce `EPOCHS` to 1
3. Increase `BATCH_SIZE` if memory allows
4. Reduce `MAX_LENGTH`

### Early stopping triggers immediately

**Symptom**: Training stops after 1 epoch.

**Cause**: Model may be overfitting or learning rate too high.

**Solutions**:
```python
# Reduce learning rate
LEARNING_RATE = 1e-5

# Increase early stopping patience
# (in trainer.py)
early_stopping_patience = 5
```

---

## Evaluation Issues

### Evaluation fails

**Symptom**: `python -m src.model.evaluate` fails.

**Solutions**:
```bash
# Ensure test data exists
python -m src.pipeline.prepare_dataset

# Check model is loaded
python -c "from src.model.predict import predict_sentiment; print(predict_sentiment('test'))"

# Verify reports directory
mkdir -p reports
```

### Low accuracy

**Symptom**: Accuracy < 80%.

**Causes**:
1. Insufficient training data
2. Poor data quality
3. Wrong label mapping
4. Model not fine-tuned

**Solutions**:
1. Use more training data (>10,000 samples)
2. Clean data thoroughly
3. Verify `SENTIMENT_MAPPING` in config
4. Train for more epochs

---

## General Debugging

### Enable verbose logging

```python
# In src/utils/logger.py
logging.basicConfig(level=logging.DEBUG)
```

### Check system status

```bash
# Python version
python --version

# Installed packages
pip list

# GPU status
nvidia-smi

# Disk space
df -h
```

### Reset everything

```bash
# Remove model weights
rm -rf models/deberta_v3/

# Remove processed data
rm -rf data/processed/

# Remove reports
rm -rf reports/

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Getting Help

If your issue is not listed here:

1. Check the [FAQ](FAQ.md)
2. Search existing [GitHub Issues](https://github.com/NikhilKhetavath/StartupPulse-AI/issues)
3. Create a new issue with:
   - Error message
   - Steps to reproduce
   - Python version
   - OS and GPU info
   - Full traceback
