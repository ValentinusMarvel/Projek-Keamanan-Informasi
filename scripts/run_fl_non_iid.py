import sys
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import flwr as fl

# Setup sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.baseline import create_baseline_model
from federated.non_iid_partition import create_non_iid_partitions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
artifact = PROJECT_ROOT / 'data' / 'processed' / 'sequence_bundle.npz'
if not artifact.exists():
    raise FileNotFoundError("sequence_bundle.npz not found.")

data = np.load(artifact, allow_pickle=True)
X = data['features'].astype(np.float32)
y_raw = data['labels']
sequence_ids = data['sequence_ids']

labels_unique = sorted(set(y_raw.tolist()))
label_to_idx = {label: i for i, label in enumerate(labels_unique)}
y = np.array([label_to_idx[val] for val in y_raw], dtype=np.int64)

num_clients = 5
rounds = 5
local_epochs = 1
batch_size = 64
learning_rate = 1e-3

# Split data
X_train_full, X_test_global, y_train_full, y_test_global = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply non-IID partitioning
train_sequence_ids = sequence_ids[:len(X_train_full)] # proxy for matching split
client_indices, speed_scalers = create_non_iid_partitions(
    X_train_full, y_train_full, train_sequence_ids, num_clients=num_clients, random_state=42
)
client_sizes = [len(idx) for idx in client_indices]

def make_loader(x_arr, y_arr, shuffle):
    ds = torch.utils.data.TensorDataset(torch.tensor(x_arr), torch.tensor(y_arr))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

client_train_loaders = {}
client_eval_loaders = {}

for cid in range(num_clients):
    idx = client_indices[cid]
    x_part = X_train_full[idx].copy()
    # Apply habit speed perturbation to features (Dwell, Flight-UD, Flight-DD)
    x_part = x_part * speed_scalers[cid]
    y_part = y_train_full[idx]
    
    if len(x_part) > 8:
        x_tr, x_ev, y_tr, y_ev = train_test_split(x_part, y_part, test_size=0.2, random_state=42)
        client_train_loaders[cid] = make_loader(x_tr, y_tr, shuffle=True)
        client_eval_loaders[cid] = make_loader(x_ev, y_ev, shuffle=False)
    else:
        client_train_loaders[cid] = make_loader(x_part, y_part, shuffle=True)
        client_eval_loaders[cid] = make_loader(x_part, y_part, shuffle=False)

global_test_loader = make_loader(X_test_global, y_test_global, shuffle=False)

# Model Helpers
def get_model_parameters(model) -> List[np.ndarray]:
    return [param.detach().cpu().numpy() for _, param in model.state_dict().items()]

def set_model_parameters(model, parameters: List[np.ndarray]):
    state_dict = model.state_dict()
    new_state_dict = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), parameters)}
    model.load_state_dict(new_state_dict, strict=True)

def train_one_epoch(model, loader) -> float:
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0.0
    batches = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)

def evaluate_model(model, loader) -> tuple[float, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            batches += 1
    return total_loss / max(batches, 1), correct / max(total, 1)

input_dim = int(X.shape[2])
num_classes = int(len(labels_unique))

global_model = create_baseline_model(
    input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes, device=device
)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid
        self.model = create_baseline_model(
            input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=num_classes, device=device
        )

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        last_loss = train_one_epoch(self.model, client_train_loaders[self.cid])
        num_examples = len(client_train_loaders[self.cid].dataset)
        return get_model_parameters(self.model), num_examples, {"train_loss": float(last_loss)}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        loss, acc = evaluate_model(self.model, client_eval_loaders[self.cid])
        num_examples = len(client_eval_loaders[self.cid].dataset)
        return float(loss), num_examples, {"accuracy": float(acc)}

def server_evaluate(server_round: int, parameters, config):
    ndarrays = parameters
    set_model_parameters(global_model, ndarrays)
    loss, acc = evaluate_model(global_model, global_test_loader)
    print(f"[Server Eval] Round {server_round} | Loss={loss:.4f} | Accuracy={acc:.4f}")
    temp_res_path = PROJECT_ROOT / 'outputs' / 'reports' / f'temp_round_non_iid_{server_round}.json'
    temp_res_path.write_text(json.dumps({"round": server_round, "global_loss": float(loss), "global_accuracy": float(acc)}))
    return float(loss), {"accuracy": float(acc)}

def start_server_fn():
    print("Starting Non-IID FL Server...")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(global_model)),
        evaluate_fn=server_evaluate,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8091",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

def start_client_fn(cid: int):
    print(f"Starting Non-IID FL Client {cid}...")
    import time
    max_retries = 15
    for attempt in range(1, max_retries + 1):
        try:
            fl.client.start_client(
                server_address="127.0.0.1:8091",
                client=FlowerClient(cid).to_client(),
            )
            print(f"Client {cid} completed execution successfully.")
            break
        except Exception as e:
            if attempt == max_retries:
                print(f"Client {cid} failed to connect after {max_retries} attempts: {e}")
                raise e
            print(f"Client {cid} connection failed. Retrying in 2 seconds (attempt {attempt}/{max_retries})...")
            time.sleep(2.0)

def main():
    if "--server" in sys.argv:
        start_server_fn()
    elif "--client" in sys.argv:
        cid_idx = sys.argv.index("--client") + 1
        cid = int(sys.argv[cid_idx])
        start_client_fn(cid)
    else:
        # Main Orchestrator Process
        print("Orchestrator: Starting Non-IID FL Server Process...")
        server_proc = subprocess.Popen([sys.executable, __file__, "--server"])
        
        # Give server a brief moment to start socket listener
        time.sleep(5.0)

        print("Orchestrator: Starting Non-IID FL Client Processes...")
        client_procs = []
        for cid in range(num_clients):
            p = subprocess.Popen([sys.executable, __file__, "--client", str(cid)])
            client_procs.append(p)
            time.sleep(0.5)

        # Wait for all clients to finish
        print("Orchestrator: Waiting for clients to complete...")
        for p in client_procs:
            p.wait()

        # Wait for server to finish
        print("Orchestrator: Waiting for server to conclude...")
        server_proc.wait()

        # Read back temp round files
        print("Orchestrator: Collecting round results...")
        round_history = []
        for r in range(1, rounds + 1):
            temp_path = PROJECT_ROOT / 'outputs' / 'reports' / f'temp_round_non_iid_{r}.json'
            if temp_path.exists():
                round_history.append(json.loads(temp_path.read_text()))
                temp_path.unlink()  # clean up

        final_loss, final_acc = evaluate_model(global_model, global_test_loader)
        print(f"\nFinal global centralized evaluation: Loss={final_loss:.4f}, Accuracy={final_acc:.4f}")

        payload = {
            'framework': 'flower',
            'strategy': 'FedAvg',
            'num_clients': num_clients,
            'client_sizes': client_sizes,
            'speed_scalers': speed_scalers,
            'rounds': rounds,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'final_global_loss': float(final_loss),
            'final_global_accuracy': float(final_acc),
            'history': round_history,
            'label_count': num_classes,
            'sample_count': int(len(X)),
        }
        
        out_path = PROJECT_ROOT / 'outputs' / 'reports' / 'fl_non_iid_metrics.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f"Saved non-IID federated learning metrics to {out_path}")

        model_path = PROJECT_ROOT / 'outputs' / 'models' / 'fl_non_iid_lstm.pt'
        torch.save({'state_dict': global_model.state_dict(), 'label_to_idx': label_to_idx}, model_path)
        print(f"Saved global Non-IID FL model to {model_path}")

if __name__ == "__main__":
    main()
