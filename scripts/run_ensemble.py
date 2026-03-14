# scripts/run_ensemble.py
"""Evaluate the weighted soft-voting ensemble on validation and test splits."""

import os
import sys
import yaml
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ensemble import EnsembleDetector, evaluate_ensemble
from src.utils import save_json, ensure_dir, gpu_info_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-weight", type=float, default=0.3)
    ap.add_argument("--transformer-weight", type=float, default=0.7)
    args = ap.parse_args()

    print(gpu_info_str())

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    splits_dir = cfg["paths"]["splits_dir"]
    baseline_path = os.path.join("models", "baseline", "tfidf_logreg.joblib")
    transformer_dir = os.path.join("models", "transformer")

    if not os.path.exists(baseline_path):
        print(f"[ERROR] Baseline model not found: {baseline_path}")
        print("Run `python scripts/run_baseline.py` first.")
        return
    if not os.path.isdir(transformer_dir):
        print(f"[ERROR] Transformer model not found: {transformer_dir}")
        print("Run `python scripts/run_transformer.py` first.")
        return

    ensemble = EnsembleDetector(
        baseline_path=baseline_path,
        transformer_dir=transformer_dir,
        baseline_weight=args.baseline_weight,
        transformer_weight=args.transformer_weight,
        max_length=int(cfg["transformer"]["max_length"]),
    )

    out_dir = os.path.join(cfg["paths"]["outputs_dir"], "metrics")
    ensure_dir(out_dir)

    for split in ["valid", "test"]:
        csv_path = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] {csv_path} not found, skipping {split}.")
            continue

        df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
        texts = df["text"].astype(str).tolist()
        labels = df["label"].values

        metrics = evaluate_ensemble(ensemble, texts, labels)
        save_json(metrics, os.path.join(out_dir, f"ensemble_{split}.json"))
        print(f"Ensemble {split}: {metrics}")


if __name__ == "__main__":
    main()
