import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir

CANON_TEXT_COLS = ["text", "Text", "content", "prompt", "essay"]
CANON_LABEL_COLS = ["label", "Label", "generated", "is_gpt", "is_ai"]


def load_raw_dataframe(raw_dir: str) -> pd.DataFrame:
    # Heuristically find a CSV in raw_dir
    csvs = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    assert csvs, f"No CSV found in {raw_dir}."
    # Prefer common names
    csvs.sort(key=lambda x: ("train" not in x.lower(), x))
    df = pd.read_csv(os.path.join(raw_dir, csvs[0]))
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    text_col = None
    for c in CANON_TEXT_COLS:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Could not find a text column in {df.columns}")

    label_col = None
    for c in CANON_LABEL_COLS:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Could not find a label column in {df.columns}")

    out = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()
    # Normalize labels to {0,1}
    if out["label"].dtype == object:
        out["label"] = out["label"].str.strip().str.lower().map({"human": 0, "ai": 1, "machine": 1, "gpt": 1})
    out["label"] = out["label"].astype(int)
    return out.dropna(subset=["text", "label"]).reset_index(drop=True)


def make_splits(df: pd.DataFrame, splits_dir: str, test_size=0.15, valid_size=0.15, seed=42):
    ensure_dir(splits_dir)
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    train_df, valid_df = train_test_split(train_df, test_size=valid_size, stratify=train_df["label"], random_state=seed)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(splits_dir, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)