# Installation Guide

Complete instructions for setting up StartupPulse AI on your local machine.

---

## Prerequisites

| Requirement | Minimum Version | Recommended |
|-------------|-----------------|-------------|
| Python | 3.10 | 3.11 |
| pip | 21.0 | Latest |
| Git | 2.30 | Latest |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| GPU | Optional | CUDA-capable (NVIDIA) |

### Checking Prerequisites

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Check Git version
git --version

# Check CUDA availability (optional)
nvidia-smi
```

---

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/NikhilKhetavath/StartupPulse-AI.git
cd StartupPulse-AI
```

### 2. Create a Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; import shap; import streamlit; print('All dependencies installed successfully')"
```

---

## GPU Setup (Optional)

For faster training and inference, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verifying GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

---

## Dependency Overview

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.2.1 | Deep learning framework |
| `transformers` | 4.40.1 | DeBERTa-v3 model architecture |
| `shap` | 0.45.0 | Explainable AI computations |
| `streamlit` | 1.33.0 | Interactive dashboard |
| `pandas` | 2.2.1 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `scikit-learn` | 1.4.1 | Metrics and evaluation |
| `matplotlib` | 3.8.3 | Plotting and visualization |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `sentencepiece` | 0.2.0 | Tokenization for DeBERTa |
| `tqdm` | 4.66.2 | Progress bars |

---

## Troubleshooting Installation

| Issue | Solution |
|-------|----------|
| `pip` not found | Install pip: `python -m ensurepip --upgrade` |
| Virtual environment fails | Try: `python -m venv --without-pip .venv` then bootstrap pip |
| CUDA out of memory | Reduce batch size in `src/config/config.py` |
| Import errors | Run `pip install -r requirements.txt` again |
| SHAP installation fails | Install build tools: `pip install setuptools wheel` |

---

## Next Steps

- [Quick Start](QUICKSTART.md) - Get running in 5 minutes
- [Usage Guide](USAGE.md) - Detailed usage instructions
- [Configuration](CONFIGURATION.md) - Customize settings
