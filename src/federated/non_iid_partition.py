"""
Non-IID partitioning and habit perturbation for keystroke dynamics federated learning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def create_non_iid_partitions(
    X: np.ndarray,
    y: np.ndarray,
    sequence_ids: np.ndarray,
    num_clients: int = 5,
    random_state: int = 42
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Partition the dataset into non-IID sets based on subject skews.
    Also perturbs typing speed for certain clients to simulate distinct user cohorts.
    
    Args:
        X: Feature tensor of shape (N, seq_len, num_features)
        y: Integer labels of shape (N,)
        sequence_ids: Array of raw sequence identifier strings
        num_clients: Number of clients
        random_state: Seed
        
    Returns:
        client_indices: List of numpy arrays containing indices for each client
        client_speed_scalers: Speed scale factor applied to each client
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    
    # Severe label skew: Sort by label to assign non-overlapping contiguous groups
    unique_labels = sorted(list(set(y.tolist())))
    num_labels = len(unique_labels)
    
    # Map labels to clients
    labels_per_client = np.array_split(unique_labels, num_clients)
    
    client_indices = []
    for cid in range(num_clients):
        # Client gets all samples matching its designated labels
        allowed_labels = set(labels_per_client[cid])
        indices = np.where(np.isin(y, list(allowed_labels)))[0]
        rng.shuffle(indices)
        client_indices.append(indices)
        
    # Habit variations: Perturb typing speeds (scale feature columns)
    # The feature columns are: [Dwell, Flight-UD, Flight-DD]
    # We will perturb the speed of clients:
    # Client 0: Slow typist (scale hold/flight times by 1.25x)
    # Client 1: Fast typist (scale hold/flight times by 0.75x)
    # Client 2-4: Normal speed typists
    client_speed_scalers = [1.25, 0.75, 1.0, 1.0, 1.0]
    
    return client_indices, client_speed_scalers
