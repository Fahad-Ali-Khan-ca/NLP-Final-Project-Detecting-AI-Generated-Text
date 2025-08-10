import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from src.utils import ensure_dir

plt.rcParams.update({"figure.dpi": 140})


def run_basic_eda(df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    # Length distribution
    lengths = df["text"].astype(str).apply(lambda s: len(s.split()))
    plt.figure()
    plt.hist(lengths, bins=50)
    plt.title("Text Length Distribution (words)")
    plt.xlabel("words"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "length_hist.png"))

    # Class balance
    counts = Counter(df["label"])  # {0, 1}
    plt.figure()
    plt.bar(["human(0)", "ai(1)"], [counts.get(0,0), counts.get(1,0)])
    plt.title("Class Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "class_balance.png"))