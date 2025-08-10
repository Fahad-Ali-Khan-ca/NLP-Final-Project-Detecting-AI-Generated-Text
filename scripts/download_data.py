import os, zipfile, argparse, shutil
from pathlib import Path

# Requires kaggle API: pip install kaggle
# Make sure ~/.kaggle/kaggle.json exists with your API token.

DEFAULT_DATASET = "shanegerami/ai-vs-human-text"


def main(dataset: str, out_dir: str):
    from kaggle.api.kaggle_api_extended import KaggleApi

    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    api = KaggleApi(); api.authenticate()
    print(f"Downloading Kaggle dataset: {dataset}")
    api.dataset_download_files(dataset, path=str(raw), force=True, quiet=False)

    # Extract any zips
    for z in raw.glob("*.zip"):
        with zipfile.ZipFile(z, 'r') as f:
            f.extractall(raw)
        z.unlink()

    print(f"Raw data in: {raw}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=str, default="data")
    args = p.parse_args()
    main(args.dataset, args.out_dir)