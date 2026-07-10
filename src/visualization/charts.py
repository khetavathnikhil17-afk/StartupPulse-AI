import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config.config import REPORTS_DIR

def plot_class_distribution(df: pd.DataFrame, label_col: str = "label", save_path=None):
    """Plots and saves the class distribution."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=label_col, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str, save_path=None):
    """Plots and saves the sentiment distribution."""
    plt.figure(figsize=(8, 6))
    df[sentiment_col].value_counts().plot.pie(autopct='%1.1f%%', cmap="viridis")
    plt.title("Sentiment Distribution")
    plt.ylabel("")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        return plt.gcf()
