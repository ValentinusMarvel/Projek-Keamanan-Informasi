"""
Evaluation metrics and analysis functions for keystroke dynamics models.

This module provides comprehensive metrics for both identification (open-world)
and authentication (closed-world) tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """Compute precision score."""
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """Compute recall score."""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """Compute F1 score."""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy from class probabilities or logits."""
    if y_proba.ndim != 2:
        raise ValueError('y_proba must be a 2D array of shape (n_samples, n_classes).')
    if k < 1:
        raise ValueError('k must be at least 1.')

    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [true_label in row for true_label, row in zip(y_true, top_k)]
    return float(np.mean(hits)) if len(hits) else 0.0


def evaluate_identification(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model on user identification task (multi-class classification).
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data (x, y)
        device: Device to evaluate on
    
    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, loss
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0))
    
    metrics = {
        'loss': total_loss / len(test_loader),
        'accuracy': compute_accuracy(all_labels, all_preds),
        'precision': compute_precision(all_labels, all_preds),
        'recall': compute_recall(all_labels, all_preds),
        'f1': compute_f1(all_labels, all_preds),
    }

    if all_logits.size:
        metrics['top3_accuracy'] = compute_top_k_accuracy(all_labels, all_logits, k=min(3, all_logits.shape[1]))
        metrics['top5_accuracy'] = compute_top_k_accuracy(all_labels, all_logits, k=min(5, all_logits.shape[1]))
        metrics['confusion_matrix'] = compute_confusion_matrix(all_labels, all_preds).tolist()
    
    return metrics


def evaluate_authentication(
    model: torch.nn.Module,
    gallery_loader: torch.utils.data.DataLoader,
    probe_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate model on authentication task (1-vs-rest verification).
    
    Args:
        model: Trained model
        gallery_loader: DataLoader with gallery samples (enrollment)
        probe_loader: DataLoader with probe samples (verification)
        device: Device to evaluate on
        threshold: Verification threshold. If None, use ROC curve to find optimal
    
    Returns:
        Dictionary with metrics: verification_rate, false_accept_rate, false_reject_rate, eer (if threshold=None)
    """
    model.eval()
    
    # Get embeddings
    def get_embeddings(loader):
        embeddings = []
        labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                emb = model.get_embedding(x)
                embeddings.append(emb.cpu().numpy())
                labels.extend(y.cpu().numpy())
        return np.vstack(embeddings), np.array(labels)
    
    gallery_emb, gallery_labels = get_embeddings(gallery_loader)
    probe_emb, probe_labels = get_embeddings(probe_loader)
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(probe_emb, gallery_emb, metric='euclidean')
    
    # Generate labels: 1 for same user, 0 for different user
    genuine_scores = []
    impostor_scores = []
    
    for i in range(len(probe_labels)):
        probe_label = probe_labels[i]
        for j in range(len(gallery_labels)):
            gallery_label = gallery_labels[j]
            distance = distances[i, j]
            
            if probe_label == gallery_label:
                genuine_scores.append(distance)
            else:
                impostor_scores.append(distance)
    
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    
    if threshold is None:
        # Compute EER
        y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
        scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, fnr, thresholds = _compute_roc_fpr_fnr(scores, y_true)
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        threshold = thresholds[eer_idx]
        
        metrics = {
            'eer': float(eer),
            'threshold': float(threshold),
        }
    else:
        genuine_accept = np.mean(genuine_scores <= threshold)
        impostor_accept = np.mean(impostor_scores <= threshold)
        
        metrics = {
            'verification_rate': float(genuine_accept),
            'false_accept_rate': float(impostor_accept),
            'false_reject_rate': float(1 - genuine_accept),
            'threshold': float(threshold),
        }
    
    return metrics


def _compute_roc_fpr_fnr(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper to compute FPR and FNR across thresholds for authentication using optimized sklearn roc_curve."""
    fpr, tpr, thresholds = roc_curve(labels, -scores)
    fnr = 1 - tpr
    return fpr, fnr, -thresholds


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)
