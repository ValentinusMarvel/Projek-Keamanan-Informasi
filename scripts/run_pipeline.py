"""Run the core pipeline of modular notebooks with a reliable kernel selection.

Usage:
    python scripts/run_pipeline.py

This script prefers the Jupyter kernel name 'projek-keystroke' (created by bootstrap_venv.ps1).
If that kernel is not available, it falls back to 'python3'.
"""
from pathlib import Path
import sys


def choose_kernel(preferred: str = "projek-keystroke") -> str:
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        ksm = KernelSpecManager()
        specs = ksm.find_kernel_specs()
        if preferred in specs:
            return preferred
    except Exception:
        # jupyter_client may not be installed; rely on fallback
        pass
    return "python3"


def main() -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:
        raise SystemExit("Missing dependencies for notebook runner. Install with: pip install nbclient nbformat jupyter-client") from exc

    root = Path(__file__).resolve().parents[1]
    notebooks_dir = root / "notebooks" / "modular"

    ordered = [
        "01_audit_dataset.ipynb",
        "02_preprocessing.ipynb",
        "03_baseline_lstm.ipynb",
        "04_differential_privacy.ipynb",
        "05_federated_learning.ipynb",
        "06_privacy_attack_eval.ipynb",
        "07_ablation_explainability.ipynb",
        "99_report_artifacts.ipynb",
    ]

    kernel_name = choose_kernel()
    print(f"Selected kernel for execution: {kernel_name}")

    for name in ordered:
        path = notebooks_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")

        print(f"Running: {name} (kernel={kernel_name})")
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=1800, kernel_name=kernel_name)
        client.execute(cwd=str(root))
        nbformat.write(nb, path)
        print(f"Done: {name}\n")

    print("All modular notebooks executed successfully.")


if __name__ == "__main__":
    main()
