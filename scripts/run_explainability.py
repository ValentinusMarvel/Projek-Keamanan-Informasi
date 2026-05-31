import sys
import json
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.baseline import create_baseline_model
from evaluation.explainability import compute_integrated_gradients

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print("Running Model Explainability & Integrated Gradients Attribution (Level 10)...")
    
    # Load data
    artifact = PROJECT_ROOT / 'data' / 'processed' / 'sequence_bundle.npz'
    if not artifact.exists():
        raise FileNotFoundError("sequence_bundle.npz not found.")
        
    data = np.load(artifact, allow_pickle=True)
    X = data['features'].astype(np.float32)
    y_raw = data['labels']
    
    labels_unique = sorted(set(y_raw.tolist()))
    label_to_idx = {label: i for i, label in enumerate(labels_unique)}
    y = np.array([label_to_idx[val] for val in y_raw], dtype=np.int64)
    
    # Pick a sample sequence
    sample_idx = 0
    input_seq = torch.tensor(X[sample_idx:sample_idx+1])
    target_class = int(y[sample_idx])
    
    # Load model
    baseline_path = PROJECT_ROOT / 'outputs' / 'models' / 'baseline_lstm.pt'
    if not baseline_path.exists():
        print("Baseline model not found. Using an un-trained model for explanation.")
        model = create_baseline_model(input_dim=X.shape[2], num_classes=len(labels_unique), device=device)
    else:
        checkpoint = torch.load(baseline_path, map_location=device)
        model = create_baseline_model(input_dim=X.shape[2], num_classes=len(labels_unique), device=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    # Compute Integrated Gradients
    attributions = compute_integrated_gradients(model, input_seq, target_class, steps=30, device=device)
    
    # Absolute attributions for overall feature importance
    abs_attributions = np.abs(attributions)
    
    # Character labels in password sequence (11 steps: 10 chars + return)
    password_labels = ['.', 't', 'i', 'e', '5', 'Shift.r', 'o', 'a', 'n', 'l', 'Return']
    feature_labels = ['Dwell (Hold) time', 'Flight time (UD)', 'Flight time (DD)']
    
    # Save explainability report
    payload = {
        "password_characters": password_labels,
        "features": feature_labels,
        "attributions": attributions.tolist(),
        "abs_attributions": abs_attributions.tolist(),
        "summary": {
            "most_important_character": password_labels[int(np.argmax(np.mean(abs_attributions, axis=1)))],
            "most_important_feature": feature_labels[int(np.argmax(np.mean(abs_attributions, axis=0)))],
            "note": "Attributions successfully generated using PyTorch Integrated Gradients."
        }
    }
    
    out_path = PROJECT_ROOT / 'outputs' / 'reports' / 'explainability_summary.json'
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"Saved explainability summary to {out_path}")
    
    # Generate attribution heatmap
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 6.5))
    
    sns.heatmap(
        attributions,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0.0,
        xticklabels=feature_labels,
        yticklabels=password_labels,
        cbar_kws={'label': 'Feature Attribution Score'}
    )
    
    plt.title('Integrated Gradients: Keystroke Timing Attributions for Password ".tie5Roanl"', fontsize=14, pad=15)
    plt.ylabel('Password Keys / Transitions', fontsize=12)
    plt.xlabel('Timing Metric Features', fontsize=12)
    plt.tight_layout()
    
    fig_path = PROJECT_ROOT / 'outputs' / 'figures' / 'keystroke_feature_importance.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved explainability heatmap to {fig_path}")

if __name__ == '__main__':
    main()
