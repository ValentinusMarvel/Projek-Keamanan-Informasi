# Project Index — Keystroke Dynamics (Modular Notebooks)

Quick navigation and one-click setup for the repository.

**Quick Start**
- **Bootstrap environment:** run the PowerShell script `bootstrap_venv.ps1` from project root (Windows PowerShell):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned; .\bootstrap_venv.ps1
```

- After install, pick kernel `projek-keystroke (.venv)` in VS Code / Jupyter.
- To run the full modular pipeline (locally):

```powershell
python scripts/run_modular_notebooks.py
```

**Notebooks (modular)**
- [01_audit_dataset.ipynb](notebooks/modular/01_audit_dataset.ipynb) — Audit & dataset validation
- [02_preprocessing.ipynb](notebooks/modular/02_preprocessing.ipynb) — Preprocessing & sequence bundle
- [03_baseline_lstm.ipynb](notebooks/modular/03_baseline_lstm.ipynb) — Baseline LSTM training
- [04_differential_privacy.ipynb](notebooks/modular/04_differential_privacy.ipynb) — DP training (Opacus)
- [05_federated_learning.ipynb](notebooks/modular/05_federated_learning.ipynb) — Flower FedAvg pipeline
- [06_privacy_attack_eval.ipynb](notebooks/modular/06_privacy_attack_eval.ipynb) — Privacy attack evaluation (MIA placeholder)
- [07_ablation_explainability.ipynb](notebooks/modular/07_ablation_explainability.ipynb) — Ablation & explainability
- [99_report_artifacts.ipynb](notebooks/modular/99_report_artifacts.ipynb) — Aggregate artifacts & final report

**Scripts & Helpers**
- [scripts/run_modular_notebooks.py](scripts/run_modular_notebooks.py) — Sequential notebook runner
- [bootstrap_venv.ps1](bootstrap_venv.ps1) — One-click venv + deps + kernel registration
- [requirements.txt](requirements.txt) — Packages used by `bootstrap_venv.ps1`

**Artifacts locations**
- Processed data: `data/processed/`
- Models: `outputs/models/`
- Reports & metrics: `outputs/reports/`

**If something fails**
- If `torch` or other heavy packages fail to install on Windows/Python 3.14, install a compatible wheel manually (see PyTorch docs) or use Python 3.11/3.12 virtualenv.
- Open an issue or paste the install error so I can help debug.
