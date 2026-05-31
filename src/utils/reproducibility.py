from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_project_root(reference_path: Path | None = None) -> Path:
    """Resolve the project root from a notebook or source file path."""
    if reference_path is None:
        reference_path = Path.cwd()

    if reference_path.name == "notebooks":
        return reference_path.parent

    if reference_path.name == "src":
        return reference_path.parent

    return reference_path if (reference_path / "data").exists() else reference_path.parent
