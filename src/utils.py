import torch
from pathlib import Path
import yaml


def set_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps:0":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        return torch.device("cpu")
    

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config