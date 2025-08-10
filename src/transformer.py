# src/transformer.py

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.hf_dataset import load_splits_from_csv


def _local_only_flag(model_name: str) -> bool:
    return os.path.isdir(model_name)


def _clean_hf_dataset(ds):
    """Remove rows with null/empty text to avoid tokenizer errors."""
    def _is_good(ex):
        t = ex.get("text", None)
        if t is None:
            return False
        if isinstance(t, float) and np.isnan(t):
            return False
        s = str(t).strip()
        return len(s) > 0

    out = {}
    for split, d in ds.items():
        d = d.filter(_is_good)
        out[split] = d
    return out


def _normalize_splits(ds_dict):
    """
    Normalize a DatasetDict-like object to have keys: train, valid, test.
    If a validation split is missing, create one from train (10% stratified).
    """
    keys = {k.lower(): k for k in ds_dict.keys()}  # map lowercase->actual

    def pick(cands):
        for c in cands:
            if c in keys:
                return keys[c]
        return None

    train_k = pick(["train", "training"])
    valid_k = pick(["valid", "validation", "val", "dev"])
    test_k  = pick(["test", "testing"])

    out = {}
    if train_k:
        out["train"] = ds_dict[train_k]
    if test_k:
        out["test"] = ds_dict[test_k]
    if valid_k:
        out["valid"] = ds_dict[valid_k]

    # If no valid split, create one from train (10% stratified)
    if "valid" not in out and "train" in out:
        try:
            out_splits = out["train"].train_test_split(
                test_size=0.10,
                seed=42,
                stratify_by_column="label" if "label" in out["train"].column_names else None,
            )
            out["train"] = out_splits["train"]
            out["valid"] = out_splits["test"]
            print("[info] No explicit valid split found; created 10% validation from train.")
        except Exception:
            # Fallback non-stratified
            out_splits = out["train"].train_test_split(test_size=0.10, seed=42)
            out["train"] = out_splits["train"]
            out["valid"] = out_splits["test"]
            print("[info] Created non-stratified 10% validation from train.")

    # Minimal sanity
    for req in ["train", "valid", "test"]:
        if req not in out:
            raise KeyError(f"Required split '{req}' is missing after normalization. Available: {list(out.keys())}")

    return out


def _tokenize_dataset(ds, tokenizer, max_length: int):
    def tok_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    ds_tok = {}
    for split, d in ds.items():
        dt = d.map(tok_fn, batched=True, remove_columns=["text"])
        dt.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        ds_tok[split] = dt
    return ds_tok


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def fine_tune_transformer(
    *,
    splits_dir: str,
    model_name: str,
    out_dir: str,
    max_length: int = 256,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.0,
    fp16: bool = True,
    gradient_checkpointing: bool = False,
    logging_steps: int = 100,
) -> Dict[str, Any]:
    """
    Train DistilBERT-like encoder for binary classification (AI vs Human).
    Returns a dict with train/valid/test metrics.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load and normalize splits
    ds_raw = load_splits_from_csv(splits_dir)  # expects columns: text, label
    ds_raw = _clean_hf_dataset(ds_raw)
    ds = _normalize_splits(ds_raw)

    # Offline-friendly load
    local_only = _local_only_flag(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=local_only,
    )

    # Enable mixed precision & cudnn perf if possible
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Tokenize
    ds_tok = _tokenize_dataset(ds, tokenizer, max_length=max_length)

    data_collator = DataCollatorWithPadding(tokenizer)

    # TrainingArguments without evaluation_strategy (older transformers compatible)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=1000,
        save_total_limit=2,
        report_to=[],                    # disable TB/W&B
        fp16=bool(fp16 and use_cuda),    # only use fp16 on GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    # Train
    train_out = trainer.train()

    # Evaluate explicitly
    valid_metrics = trainer.evaluate(eval_dataset=ds_tok["valid"])
    test_metrics = trainer.evaluate(eval_dataset=ds_tok["test"])

    # Save model + tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    result = {
        "train_loss": float(train_out.metrics.get("train_loss", 0.0)),
        "valid_accuracy": float(valid_metrics.get("eval_accuracy", 0.0)),
        "valid_precision": float(valid_metrics.get("eval_precision", 0.0)),
        "valid_recall": float(valid_metrics.get("eval_recall", 0.0)),
        "valid_f1": float(valid_metrics.get("eval_f1", 0.0)),
        "test_accuracy": float(test_metrics.get("eval_accuracy", 0.0)),
        "test_precision": float(test_metrics.get("eval_precision", 0.0)),
        "test_recall": float(test_metrics.get("eval_recall", 0.0)),
        "test_f1": float(test_metrics.get("eval_f1", 0.0)),
    }

    # Optional: persist a small CSV
    try:
        metrics_dir = os.path.join("outputs", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        pd.DataFrame([{
            "split": "valid", **{k.replace("eval_", ""): v for k, v in valid_metrics.items() if k.startswith("eval_")}
        }, {
            "split": "test", **{k.replace("eval_", ""): v for k, v in test_metrics.items() if k.startswith("eval_")}
        }]).to_csv(os.path.join(metrics_dir, "transformer_metrics.csv"), index=False)
    except Exception:
        pass

    return result
