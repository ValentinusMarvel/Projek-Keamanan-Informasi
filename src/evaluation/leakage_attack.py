"""
Gradient Leakage Reconstruction Attack using LBFGS optimization.
Reconstructs a keystroke timing sequence (11x3) from local model gradients.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict

def perform_gradient_leakage_attack(
    model: nn.Module,
    original_sequence: torch.Tensor,
    original_label: torch.Tensor,
    num_iterations: int = 50,
    lr: float = 0.1,
    device: str = 'cpu'
) -> Tuple[np.ndarray, float, float]:
    """
    Simulates a Behavioral Reconstruction Attack by matching gradients.
    
    Args:
        model: PyTorch model being attacked
        original_sequence: Original keystroke sequence of shape (1, seq_len, input_dim)
        original_label: Target label of shape (1,)
        num_iterations: LBFGS optimization steps
        lr: Learning rate
        device: Device
        
    Returns:
        reconstructed_sequence: The reconstructed timing features as a numpy array
        mse: Mean Squared Error between original and reconstructed sequence
        cosine_sim: Cosine similarity between original and reconstructed sequence
    """
    model.eval()
    model.zero_grad()
    
    # 1. Compute original gradients
    criterion = nn.CrossEntropyLoss()
    original_sequence = original_sequence.to(device)
    original_label = original_label.to(device)
    
    out = model(original_sequence)
    loss = criterion(out, original_label)
    original_dy_dx = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    original_dy_dx = [g.detach().clone() for g in original_dy_dx]
    
    # 2. Initialize dummy sequence to optimize
    dummy_sequence = torch.randn_like(original_sequence, requires_grad=True, device=device)
    
    # 3. LBFGS optimizer for matching gradients
    optimizer = torch.optim.LBFGS([dummy_sequence], lr=lr)
    
    best_reconstruction = None
    best_loss = float('inf')
    
    # Gradient matching optimization loop
    for iteration in range(num_iterations):
        def closure():
            optimizer.zero_grad()
            dummy_out = model(dummy_sequence)
            dummy_loss = criterion(dummy_out, original_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            
            grad_loss = 0.0
            for g_dummy, g_orig in zip(dummy_dy_dx, original_dy_dx):
                grad_loss += ((g_dummy - g_orig) ** 2).sum()
                
            grad_loss.backward()
            return grad_loss
            
        optimizer.step(closure)
        
        # Save current reconstruction
        best_reconstruction = dummy_sequence.detach().cpu().numpy().copy()
                
    if best_reconstruction is None:
        best_reconstruction = dummy_sequence.detach().cpu().numpy()
        
    # Calculate reconstruction quality metrics
    orig_np = original_sequence.detach().cpu().numpy().squeeze()
    recon_np = best_reconstruction.squeeze()
    
    mse = float(np.mean((orig_np - recon_np) ** 2))
    
    # Cosine Similarity
    orig_flat = orig_np.flatten()
    recon_flat = recon_np.flatten()
    norm_orig = np.linalg.norm(orig_flat)
    norm_recon = np.linalg.norm(recon_flat)
    if norm_orig > 0 and norm_recon > 0:
        cosine_sim = float(np.dot(orig_flat, recon_flat) / (norm_orig * norm_recon))
    else:
        cosine_sim = 0.0
        
    return recon_np, mse, cosine_sim
