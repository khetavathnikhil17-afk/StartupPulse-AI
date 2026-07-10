import json
from pathlib import Path

def save_json(data: dict, file_path: Path):
    """Saves dictionary to a JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(file_path: Path) -> dict:
    """Loads a JSON file into a dictionary."""
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
