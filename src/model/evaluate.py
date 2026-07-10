import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from src.config.config import TEST_DATA_PATH, REPORTS_DIR, SENTIMENT_MAPPING
from src.model.predict import predict_sentiment
from src.utils.logger import get_logger
from src.pipeline.prepare_dataset import prepare_data

logger = get_logger(__name__)

def evaluate_model():
    if not TEST_DATA_PATH.exists():
        logger.info("Test data not found, preparing data...")
        prepare_data()

    logger.info("Loading test dataset...")
    df = pd.read_csv(TEST_DATA_PATH)
    
    true_labels = df["label"].tolist()
    pred_labels = []
    
    logger.info("Running predictions on test set...")
    inv_map = {v: k for k, v in SENTIMENT_MAPPING.items()}
    
    for text in tqdm(df["review"], desc="Evaluating"):
        res = predict_sentiment(text)
        pred_labels.append(inv_map[res["label"]])

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)
    
    logger.info(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    target_names = ["Negative", "Neutral", "Positive"]
    labels_list = [0, 1, 2]
    
    report = classification_report(true_labels, pred_labels, labels=labels_list, target_names=target_names, zero_division=0)
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)
        
    metrics_df = pd.DataFrame([{
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }])
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    
    cm = confusion_matrix(true_labels, pred_labels, labels=labels_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Evaluation artifacts saved in {REPORTS_DIR}")
    
if __name__ == "__main__":
    evaluate_model()