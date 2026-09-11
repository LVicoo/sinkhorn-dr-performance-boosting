from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can make some operations slower or unavailable.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def timestamped_dir(root: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = Path(root) / f"{stamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "arrays").mkdir()
    (path / "plots").mkdir()
    return path


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def as_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
