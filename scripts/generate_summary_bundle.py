import json
from pathlib import Path

def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    REPORTS_DIR = PROJECT_ROOT / 'outputs' / 'reports'
    
    files = {
        'baseline': REPORTS_DIR / 'baseline_metrics.json',
        'dp': REPORTS_DIR / 'dp_metrics.json',
        'fl': REPORTS_DIR / 'fl_metrics.json',
        'fl_dp': REPORTS_DIR / 'fl_dp_metrics.json',
        'non_iid': REPORTS_DIR / 'fl_non_iid_metrics.json',
        'fl_transfer': REPORTS_DIR / 'fl_transfer_metrics.json',
        'attack': REPORTS_DIR / 'attack_metrics.json',
        'leakage': REPORTS_DIR / 'leakage_metrics.json',
    }
    
    loaded = {}
    for name, path in files.items():
        if path.exists():
            loaded[name] = json.loads(path.read_text(encoding='utf-8'))
        else:
            loaded[name] = {}
            
    json_path = REPORTS_DIR / 'final_summary_bundle.json'
    json_path.write_text(json.dumps(loaded, indent=2), encoding='utf-8')
    print(f"Generated and saved final summary bundle to: {json_path}")

if __name__ == '__main__':
    main()
