import os, json, random, numpy as np
from typing import Dict

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_metrics_from_preds(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def gpu_info_str() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"GPU: {name} | CC: {cap} | VRAM: {mem:.1f} GB"