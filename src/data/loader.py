"""
Dataset loading and management for keystroke dynamics experiments.

This module provides unified interfaces for loading both synthetic demo data
and the real Keystroke Dynamics Benchmark dataset from Kaggle.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class SequenceDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for keystroke sequences.
    """
    features: torch.Tensor  # (n_samples, seq_len, n_features)
    labels: torch.Tensor    # (n_samples,)
    sequence_ids: Optional[List[str]] = None
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_keystroke_dataset(
    data_dir: Path,
    split: str = 'train',
    seq_features: Optional[List[str]] = None,
    label_col: str = 'subject',
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42
) -> SequenceDataset:
    """
    Load keystroke dataset from raw CSV files (Kaggle DSL-StrongPasswordData).
    
    Expected file structure:
        data/raw/DSL-StrongPasswordData.csv
    
    Args:
        data_dir: Path to data directory (contains raw/ and processed/)
        split: One of 'train', 'val', 'test'
        seq_features: List of timing feature columns to use. If None, auto-detect.
        label_col: Column name for labels (default: 'subject')
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        seed: Random seed for reproducibility
    
    Returns:
        SequenceDataset with features and labels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    raw_path = data_dir / 'raw' / 'DSL-StrongPasswordData.csv'
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}\n"
            f"Please download DSL-StrongPasswordData.csv from Kaggle and place it in data/raw/"
        )
    
    # Load raw data
    df = pd.read_csv(raw_path)
    
    # Auto-detect timing features if not provided
    if seq_features is None:
        seq_features = [col for col in df.columns if col not in [label_col, 'sessionIndex', 'rep']]
    
    # Extract labels
    labels_map = {user: idx for idx, user in enumerate(sorted(df[label_col].unique()))}
    y = np.array([labels_map[user] for user in df[label_col]])
    
    # Extract features
    X = df[seq_features].values.astype(np.float32)
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features
    means = np.mean(X, axis=0, keepdims=True)
    stds = np.std(X, axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    X = (X - means) / stds
    
    # Reshape to sequences (assuming already windowed)
    # If X is 2D, reshape to (n_samples, 1, n_features) for temporal LSTM
    if len(X.shape) == 2:
        X = X[:, np.newaxis, :]
    
    # Split data
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    n_val = int(n_train * val_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    if split == 'train':
        idx = indices[:n_train - n_val]
    elif split == 'val':
        idx = indices[n_train - n_val:n_train]
    elif split == 'test':
        idx = indices[n_train:]
    else:
        raise ValueError(f"Unknown split: {split}")
    
    X_split = torch.tensor(X[idx], dtype=torch.float32)
    y_split = torch.tensor(y[idx], dtype=torch.long)
    
    return SequenceDataset(
        features=X_split,
        labels=y_split,
        sequence_ids=None
    )


def create_demo_dataset(
    n_samples: int = 100,
    n_subjects: int = 10,
    seq_len: int = 50,
    n_features: int = 31,
    seed: int = 42
) -> SequenceDataset:
    """
    Create a synthetic demo dataset for testing and development.
    
    Args:
        n_samples: Total number of sequences
        n_subjects: Number of subjects/users
        seq_len: Length of each sequence
        n_features: Number of timing features per time step
        seed: Random seed
    
    Returns:
        SequenceDataset with synthetic data
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate synthetic features (normal distribution)
    X = np.random.normal(0, 1, (n_samples, seq_len, n_features)).astype(np.float32)
    
    # Generate labels (uniform distribution over subjects)
    y = np.random.randint(0, n_subjects, n_samples)
    
    return SequenceDataset(
        features=torch.tensor(X, dtype=torch.float32),
        labels=torch.tensor(y, dtype=torch.long),
    )


def create_train_val_test_split(
    dataset: SequenceDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset: Input dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        seed: Random seed
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    def make_split(idx):
        return SequenceDataset(
            features=dataset.features[idx],
            labels=dataset.labels[idx],
        )
    
    return make_split(train_idx), make_split(val_idx), make_split(test_idx)
