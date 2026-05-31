"""
Baseline LSTM model for keystroke dynamics user identification.

This module provides a standard LSTM-based architecture for learning
temporal patterns in keystroke data.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class KeystrokeLSTM(nn.Module):
    """
    LSTM model for keystroke sequence classification.
    
    Args:
        input_dim: Number of input features (e.g., 31 timing features)
        hidden_dim: Hidden dimension for LSTM layers
        num_layers: Number of stacked LSTM layers
        num_classes: Number of output classes (e.g., 51 subjects)
        dropout: Dropout probability between layers (default: 0.3)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3
    ):
        super(KeystrokeLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state from last layer
        last_hidden = h_n[-1]  # shape: (batch_size, hidden_dim)
        
        # Classification
        logits = self.fc(last_hidden)
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the intermediate embedding (before classification head).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Embedding tensor of shape (batch_size, hidden_dim)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return last_hidden


def create_baseline_model(
    input_dim: int = 31,
    hidden_dim: int = 64,
    num_layers: int = 2,
    num_classes: int = 51,
    device: str = 'cpu'
) -> KeystrokeLSTM:
    """
    Create and initialize a baseline LSTM model.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        num_classes: Number of classes
        device: Device to place model on ('cpu' or 'cuda')
    
    Returns:
        Initialized KeystrokeLSTM model on the specified device
    """
    model = KeystrokeLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes
    )
    return model.to(device)
