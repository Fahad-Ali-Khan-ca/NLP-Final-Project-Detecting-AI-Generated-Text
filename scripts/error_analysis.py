# scripts/error_analysis.py
import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader

from src.hf_dataset import load_splits_from_csv
from src.utils import ensure_dir


def _as_str_list(values):
    """Coerce a batch 'text' field (may be None/str/list) to list[str]."""
    out = []
    for x in values:
        if x is None:
            out.append("")
        elif isinstance(x, list):
            out.append(" ".join(str(t) for t in x))
        else:
            out.append(str(x))
    return out


def _pick_eval_split(ds_dict):
    """Choose the best available eval split name."""
    for name in ("validation", "valid", "val", "dev", "test"):
        if name in ds_dict:
            return name
    raise KeyError(f"No eval split found. Available: {list(ds_dict.keys())}")


def main(k: int, seed: int):
    torch.manual_seed(seed)
    print("[INFO] Starting error_analysis...", flush=True)

    # Load config
    print("[INFO] Loading config...", flush=True)
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    out_root = cfg["paths"]["outputs_dir"]
    err_dir = os.path.join(out_root, "error_analysis")
    ensure_dir(err_dir)

    # Load dataset (expects splits_dir to contain train/valid/test CSVs)
    print("[INFO] Loading splits...", flush=True)
    ds = load_splits_from_csv(cfg["paths"]["splits_dir"])
    print(f"[INFO] Splits present: {list(ds.keys())}", flush=True)

    # Pick evaluation split
    eval_split = _pick_eval_split(ds)
    print(f"[INFO] Eval split: {eval_split}", flush=True)

    # Load model/tokenizer from your trained checkpoint dir
    model_dir = os.path.join("models", "transformer")
    print(f"[INFO] Loading model from: {model_dir}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    use_cuda = torch.cuda.is_available()
    print(f"[INFO] CUDA available: {'using GPU' if use_cuda else 'using CPU'}", flush=True)
    if use_cuda:
        model.cuda()

    max_len = int(cfg["transformer"]["max_length"])

    def tokenize_fn(batch):
        texts = _as_str_list(batch["text"])
        return tok(texts, truncation=True, padding=False, max_length=max_len)

    # Only keep 'text' and 'label' during tokenization, then remove raw 'text'
    keep_before = {"text", "label"}
    cols_to_remove = [c for c in ds["train"].column_names if c not in keep_before]

    print("[INFO] Tokenizing...", flush=True)
    ds_tok = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=cols_to_remove,  # removes everything except 'text' & 'label'
        desc="Tokenizing",
    )
    # Now drop 'text' so the collator only sees tensors
    if "text" in ds_tok[eval_split].column_names:
        ds_tok = ds_tok.remove_columns("text")

    # Format for PyTorch
    input_cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in ds_tok[eval_split].column_names:
        input_cols.append("token_type_ids")
    ds_tok = ds_tok.with_format("torch", columns=input_cols + ["label"])

    # DataLoader
    print("[INFO] Building DataLoader...", flush=True)
    collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")
    eval_bs = int(cfg["transformer"]["per_device_eval_batch_size"])
    dl = DataLoader(
        ds_tok[eval_split],
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collator,
    )

    # Inference
    print("[INFO] Running inference...", flush=True)
    preds, refs, probs = [], [], []
    amp_dtype = torch.float16 if use_cuda else None

    with torch.inference_mode():
        for batch in dl:
            # target key may be 'labels' (collator) or 'label' (dataset)
            label_key = "labels" if "labels" in batch else "label"
            labels = batch.pop(label_key)

            if use_cuda:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_cuda, dtype=amp_dtype):
                logits = model(**batch).logits

            pred = logits.argmax(-1)
            prob = torch.softmax(logits, dim=-1).max(dim=-1).values

            preds.extend(pred.detach().cpu().tolist())
            refs.extend(labels.detach().cpu().tolist() if hasattr(labels, "detach") else list(labels))
            probs.extend(prob.detach().cpu().tolist())

    preds = np.array(preds, dtype=int)
    refs = np.array(refs, dtype=int)
    probs = np.array(probs, dtype=float)

    # Build a DataFrame with original indices to help trace back if needed
    # We’ll load the raw CSV for the eval split to retrieve text for inspection.
    split_csv = os.path.join(cfg["paths"]["splits_dir"], f"{eval_split}.csv")
    if not os.path.exists(split_csv) and eval_split == "validation":
        # our prepare script might have saved 'valid.csv'
        alt = os.path.join(cfg["paths"]["splits_dir"], "valid.csv")
        split_csv = alt if os.path.exists(alt) else split_csv

    raw_df = None
    if os.path.exists(split_csv):
        raw_df = pd.read_csv(split_csv)
        # Try to align lengths; if mismatch, just skip text attachment
        if len(raw_df) != len(preds):
            raw_df = None

    df = pd.DataFrame(
        {
            "pred": preds,
            "label": refs,
            "confidence": probs,
        }
    )
    if raw_df is not None and "text" in raw_df.columns:
        df["text"] = raw_df["text"].astype(str)

    # Misclassified examples
    wrong = df[df["pred"] != df["label"]].copy()
    wrong_sorted = wrong.sort_values("confidence", ascending=False)

    # Save outputs
    all_pred_path = os.path.join(err_dir, f"{eval_split}_predictions.csv")
    wrong_path = os.path.join(err_dir, f"{eval_split}_misclassified.csv")
    topk_path = os.path.join(err_dir, f"{eval_split}_top{min(k, len(wrong_sorted))}_confident_wrongs.csv")

    df.to_csv(all_pred_path, index=False)
    wrong_sorted.to_csv(wrong_path, index=False)
    wrong_sorted.head(k).to_csv(topk_path, index=False)

    # Summary
    acc = (preds == refs).mean() if len(refs) else float("nan")
    print(f"[INFO] Eval accuracy on '{eval_split}': {acc:.6f}", flush=True)
    print(f"[INFO] Saved all predictions  -> {all_pred_path}", flush=True)
    print(f"[INFO] Saved misclassified    -> {wrong_path} (count={len(wrong_sorted)})", flush=True)
    print(f"[INFO] Saved top-{k} confident -> {topk_path}", flush=True)
    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=50, help="Top-K most confident wrong predictions to save")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.k, args.seed)
