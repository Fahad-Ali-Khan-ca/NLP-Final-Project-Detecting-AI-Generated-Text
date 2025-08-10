import argparse, yaml, os
import pandas as pd
import sys, os
from src.data import load_raw_dataframe, normalize_schema, make_splits
from src.utils import ensure_dir

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main(cfg_path: str, test_size: float, valid_size: float, seed: int):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_dir = cfg["paths"]["raw_dir"]
    splits_dir = cfg["paths"]["splits_dir"]

    df = load_raw_dataframe(raw_dir)
    df = normalize_schema(df)

    # Basic cleaning: strip
    df["text"] = df["text"].astype(str).str.replace("\r\n", " ").str.replace("\n", " ").str.strip()

    ensure_dir(cfg["paths"]["processed_dir"])  # not heavily used here, but kept for extensibility
    make_splits(df, splits_dir, test_size=test_size, valid_size=valid_size, seed=seed)
    print(f"Saved splits to {splits_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--valid-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, args.test_size, args.valid_size, args.seed)