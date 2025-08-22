from pathlib import Path
import json, yaml

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
