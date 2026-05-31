"""Run modular notebooks in sequence using nbclient.

Usage:
    python scripts/run_modular_notebooks.py
"""

from pathlib import Path


def main() -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except Exception as exc:
        raise SystemExit(
            "Missing dependencies for notebook runner. Install with: pip install nbclient nbformat"
        ) from exc

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

    for name in ordered:
        path = notebooks_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Notebook not found: {path}")

        print(f"Running: {name}")
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=1800, kernel_name="python3")
        client.execute(cwd=str(root))
        nbformat.write(nb, path)
        print(f"Done: {name}\n")

    print("All modular notebooks executed successfully.")


if __name__ == "__main__":
    main()
