"""
Membership Inference Attack (MIA) evaluation for keystroke dynamics models.

This module implements a shadow-classifier based MIA that uses prediction
confidences and losses to infer if a typing sample was used in training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def extract_mia_features(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract prediction features (logits, confidence, entropy, and loss) for MIA.
    
    Args:
        model: Target model
        loader: DataLoader containing samples
        device: Target device
        
    Returns:
        features: 2D array of extracted prediction features
        targets: 1D array of original labels
    """
    model.eval()
    all_features = []
    all_targets = []
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            # Loss per sample
            loss = criterion(logits, y).cpu().numpy()
            
            # Prediction probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Confidence (probability of the true label)
            confidence = np.array([p[target] for p, target in zip(probs, y.cpu().numpy())])
            
            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            
            # Max probability
            max_prob = np.max(probs, axis=1)
            
            # Feature matrix: [loss, confidence, entropy, max_prob]
            features = np.column_stack([loss, confidence, entropy, max_prob])
            all_features.append(features)
            all_targets.extend(y.cpu().numpy())
            
    return np.vstack(all_features), np.array(all_targets)


def evaluate_mia(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    random_state: int = 42
) -> Dict[str, float]:
    """
    Run Membership Inference Attack (MIA) evaluation.
    
    Args:
        model: Target model
        train_loader: DataLoader with member samples (used in training)
        test_loader: DataLoader with non-member samples (not used in training)
        device: Device to run on
        random_state: Random state for train-test split
        
    Returns:
        Dictionary with attack metrics: accuracy, precision, recall, roc_auc
    """
    # Extract features for members and non-members
    member_feats, _ = extract_mia_features(model, train_loader, device)
    non_member_feats, _ = extract_mia_features(model, test_loader, device)
    
    # Create labels: 1 for members, 0 for non-members
    member_labels = np.ones(len(member_feats))
    non_member_labels = np.zeros(len(non_member_feats))
    
    X_attack = np.vstack([member_feats, non_member_feats])
    y_attack = np.concatenate([member_labels, non_member_labels])
    
    # Train-test split for shadow classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X_attack, y_attack, test_size=0.3, random_state=random_state, stratify=y_attack
    )
    
    # Train Logistic Regression shadow classifier
    shadow_clf = LogisticRegression(max_iter=1000, random_state=random_state)
    shadow_clf.fit(X_train, y_train)
    
    # Predict
    y_pred = shadow_clf.predict(X_test)
    y_proba = shadow_clf.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    metrics = {
        'attack_accuracy': float(accuracy_score(y_test, y_pred)),
        'attack_precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'attack_recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'attack_auc': float(roc_auc_score(y_test, y_proba))
    }
    
    return metrics
