# scripts/report_metrics.py
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from torch.utils.data import DataLoader
from datasets import DatasetDict
from src.hf_dataset import load_splits_from_csv
from src.utils import ensure_dir


# --- helpers -----------------------------------------------------------------
def _as_str_list(values):
    """Coerce a batch 'text' field to list[str]."""
    out = []
    for x in values:
        if x is None:
            out.append("")
        elif isinstance(x, list):
            out.append(" ".join(str(t) for t in x))
        else:
            out.append(str(x))
    return out


def _compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}


def _pick_eval_split(ds_dict):
    for name in ("validation", "valid", "val", "dev", "test"):
        if name in ds_dict:
            return name
    raise KeyError(f"No eval split found. Available: {list(ds_dict.keys())}")


def _tokenize_all_splits(ds: DatasetDict, tok, max_len: int):
    """Map tokenization per split, removing non-tensor columns."""
    ds_tok = {}
    for split_name, dset in ds.items():
        keep = {"label"}
        remove_cols = [c for c in dset.column_names if c not in keep]

        def tokenize_fn(batch):
            texts = _as_str_list(batch["text"])
            return tok(texts, truncation=True, padding=False, max_length=max_len)

        ds_tok[split_name] = dset.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols,  # removes 'text' (& any extras)
            desc=f"Tokenizing {split_name}",
        )
    return DatasetDict(ds_tok)


def _predict_split(dset, tok, model, batch_size: int):
    # Figure out which label column exists
    label_key = "labels" if "labels" in dset.column_names else ("label" if "label" in dset.column_names else None)

    input_cols = ["input_ids", "attention_mask"]
    if "token_type_ids" in dset.column_names:
        input_cols.append("token_type_ids")
    if label_key:
        input_cols.append(label_key)

    dset = dset.with_format("torch", columns=input_cols)
    collator = DataCollatorWithPadding(tokenizer=tok)
    dl = DataLoader(dset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    preds, refs, probs = [], [], []
    amp_dtype = torch.float16 if torch.cuda.is_available() else None

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    with torch.inference_mode():
        for batch in dl:
            # pop labels robustly
            labels = batch.pop("labels", None)
            if labels is None:
                labels = batch.pop("label", None)

            if torch.cuda.is_available():
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=amp_dtype):
                logits = model(**batch).logits

            pred = logits.argmax(-1).detach().cpu().numpy()
            prob = torch.softmax(logits, dim=-1).max(-1).values.detach().cpu().numpy()
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())

            if labels is not None:
                labels = labels.detach().cpu().numpy().tolist()
                refs.extend(labels)

    return {"preds": preds, "probs": probs, "labels": refs}


def _save_confusion_png(cm, title, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4.2, 3.6))
    im = plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha="center", va="center")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# --- main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="models/transformer",
                        help="Directory with fine-tuned model & tokenizer")
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["paths"]["outputs_dir"]
    plots_dir = os.path.join(out_dir, "plots"); ensure_dir(plots_dir)
    metrics_dir = os.path.join(out_dir, "metrics"); ensure_dir(metrics_dir)
    tables_dir = os.path.join(out_dir, "tables"); ensure_dir(tables_dir)

    # Load data
    ds = load_splits_from_csv(cfg["paths"]["splits_dir"])
    eval_split = _pick_eval_split(ds)

    # Model / tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Tokenize (robust)
    max_len = int(cfg["transformer"]["max_length"])
    ds_tok = _tokenize_all_splits(ds, tok, max_len)

    # Predict on validation (or chosen eval split)
    print(f"[INFO] Evaluating on '{eval_split}'…")
    val_out = _predict_split(
        ds_tok[eval_split], tok, model, int(cfg["transformer"]["per_device_eval_batch_size"])
    )
    val_metrics = _compute_metrics(val_out["labels"], val_out["preds"])
    with open(os.path.join(metrics_dir, "transformer_valid.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    cm_val = confusion_matrix(val_out["labels"], val_out["preds"])
    _save_confusion_png(cm_val, "Transformer Confusion (validation)",
                        os.path.join(plots_dir, "transformer_confusion_validation.png"))
    # Save predictions CSV (validation)
    pd.DataFrame({
        "pred": val_out["preds"],
        "label": val_out["labels"],
        "confidence": val_out["probs"],
    }).to_csv(os.path.join(tables_dir, "validation_predictions.csv"), index=False)

    # Predict on test
    if "test" in ds_tok:
        print("[INFO] Evaluating on 'test'…")
        test_out = _predict_split(
            ds_tok["test"], tok, model, int(cfg["transformer"]["per_device_eval_batch_size"])
        )
        test_metrics = _compute_metrics(test_out["labels"], test_out["preds"])
        with open(os.path.join(metrics_dir, "transformer_test.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        cm_test = confusion_matrix(test_out["labels"], test_out["preds"])
        _save_confusion_png(cm_test, "Transformer Confusion (test)",
                            os.path.join(plots_dir, "transformer_confusion_test.png"))
        # Save predictions CSV (test)
        pd.DataFrame({
            "pred": test_out["preds"],
            "label": test_out["labels"],
            "confidence": test_out["probs"],
        }).to_csv(os.path.join(tables_dir, "test_predictions.csv"), index=False)
    else:
        test_metrics = None

    # Optional: compare with baseline if exists
    baseline_valid = os.path.join(metrics_dir, "baseline_valid.json")
    rows = []
    if os.path.exists(baseline_valid):
        with open(baseline_valid, "r") as f:
            base = json.load(f)
        rows.append({"model": "baseline", **base})

    rows.append({"model": "transformer_valid", **val_metrics})
    if test_metrics:
        rows.append({"model": "transformer_test", **test_metrics})

    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "metrics_comparison.csv"), index=False)
    print("[INFO] Done. Metrics & plots saved in 'outputs/'.")


if __name__ == "__main__":
    main()
