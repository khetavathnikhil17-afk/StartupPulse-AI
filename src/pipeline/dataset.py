"""
PyTorch Dataset module for sentiment analysis.

This module provides a custom Dataset class for loading and tokenizing
text data for transformer-based sentiment analysis.
"""
import torch
from torch.utils.data import Dataset
from typing import List


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for Sentiment Analysis.
    
    Handles tokenization and encoding of text data for transformer models.
    
    Args:
        texts: List of input text strings
        labels: List of integer labels
        tokenizer: Hugging Face tokenizer instance
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
