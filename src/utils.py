import warnings
from pathlib import Path

import torch
import yaml


def _backend_available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return torch.backends.mps.is_available()
    return kind == "cpu"


def set_device(device: str = None) -> torch.device:
    """Resolve a device string, falling back to CPU when the backend is missing.

    Accepts "auto"/"gpu" (pick the best available), "cpu", and "cuda"/"mps" with
    an optional index. Previously only the exact string "mps:0" ever selected a
    GPU: "gpu" -- the value shipped in config/config.yaml -- "mps" and "cuda:0"
    all fell through to CPU silently, so the whole framework ran on CPU without
    saying so. Unavailable backends now warn instead of degrading quietly.
    """
    requested = (device or "auto").strip().lower()

    if requested in ("auto", "gpu"):
        for kind in ("cuda", "mps"):
            if _backend_available(kind):
                return torch.device(kind)
        return torch.device("cpu")

    kind = requested.split(":")[0]
    if not _backend_available(kind):
        warnings.warn(f"device {device!r} requested but {kind} is unavailable; "
                      f"falling back to CPU", RuntimeWarning, stacklevel=2)
        return torch.device("cpu")

    return torch.device(requested)


def load_config(config_path: Path) -> dict:
    """Load a YAML file as a plain dict.

    Note: `src.config.load_config` has the same name but returns a typed
    `Config` via dacite. Prefer that one for anything model-facing.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
