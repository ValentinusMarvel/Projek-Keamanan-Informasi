"""
Integrated Gradients implementation for PyTorch LSTM models to explain keystroke dynamics attributions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

def compute_integrated_gradients(
    model: nn.Module,
    input_sequence: torch.Tensor,
    target_class: int,
    steps: int = 50,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Computes Integrated Gradients attribution for a given input sequence.
    
    Args:
        model: The trained LSTM model
        input_sequence: Keystroke sequence of shape (1, seq_len, num_features)
        target_class: The predicted integer class index
        steps: Number of integration steps (Riemann sum approximation)
        device: Device
        
    Returns:
        attributions: Attribution map of shape (seq_len, num_features)
    """
    model.eval()
    model.zero_grad()
    
    input_sequence = input_sequence.to(device).clone().detach()
    baseline = torch.zeros_like(input_sequence, device=device) # baseline is all zeros (neutral timing)
    
    # Generate scaled inputs along the path from baseline to input sequence
    # path = baseline + alpha * (input - baseline)
    scaled_inputs = []
    for alpha in np.linspace(0.0, 1.0, steps):
        scaled_input = baseline + alpha * (input_sequence - baseline)
        scaled_input.requires_grad = True
        scaled_inputs.append(scaled_input)
        
    # Stack inputs to process in a single batch
    batch_inputs = torch.cat(scaled_inputs, dim=0)
    
    # Forward pass
    logits = model(batch_inputs)
    probs = torch.softmax(logits, dim=1)
    
    # Calculate gradients of the predicted class score with respect to inputs
    score = logits[:, target_class].sum()
    grads = torch.autograd.grad(score, batch_inputs)[0]
    
    # Approximate the integral using the average of gradients
    avg_grads = torch.mean(grads, dim=0, keepdim=True) # shape: (1, seq_len, num_features)
    
    # Attribute = (input - baseline) * average gradient
    attributions = (input_sequence - baseline) * avg_grads
    attributions_np = attributions.detach().cpu().numpy().squeeze()
    
    return attributions_np
