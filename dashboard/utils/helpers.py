"""Shared utility functions for the dashboard."""

import html as _html
import base64
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def sanitize_html(text: str) -> str:
    return _html.escape(str(text))


def load_image_safe(path: Path):
    try:
        if path.exists():
            return Image.open(str(path))
    except Exception:
        pass
    return None


def get_logo_base64() -> str:
    logo_path = PROJECT_ROOT / "assets" / "logo.png"
    if logo_path.exists():
        return base64.b64encode(open(str(logo_path), "rb").read()).decode()
    return ""


def load_metrics():
    import pandas as pd
    from src.config.config import REPORTS_DIR
    metrics_file = REPORTS_DIR / "model_metrics.csv"
    if metrics_file.exists():
        return pd.read_csv(metrics_file)
    return None


def get_dataset_stats():
    import pandas as pd
    from src.config.config import (
        TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH
    )
    stats = {
        "train": 0, "val": 0, "test": 0, "total": 0,
        "classes": {}, "avg_length": 0,
    }
    dfs = []
    if TRAIN_DATA_PATH.exists():
        df = pd.read_csv(TRAIN_DATA_PATH)
        stats["train"] = len(df)
        dfs.append(df)
    if VALIDATION_DATA_PATH.exists():
        df = pd.read_csv(VALIDATION_DATA_PATH)
        stats["val"] = len(df)
        dfs.append(df)
    if TEST_DATA_PATH.exists():
        df = pd.read_csv(TEST_DATA_PATH)
        stats["test"] = len(df)
        dfs.append(df)
    if dfs:
        full_df = pd.concat(dfs)
        stats["total"] = len(full_df)
        stats["classes"] = full_df.get("label", pd.Series([])).value_counts().to_dict()
        if "review" in full_df.columns:
            stats["avg_length"] = full_df["review"].apply(lambda x: len(str(x).split())).mean()
    return stats
