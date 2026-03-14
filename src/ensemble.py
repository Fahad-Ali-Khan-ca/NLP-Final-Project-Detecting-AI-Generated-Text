# src/ensemble.py
"""Weighted soft-voting ensemble that combines TF-IDF baseline and
DistilBERT transformer predictions for improved detection accuracy.

The ensemble averages class-probability vectors from both models with
configurable weights, enabling a precision–recall trade-off without
retraining either sub-model.
"""

import os
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EnsembleDetector:
    """Soft-voting ensemble of a TF-IDF + LogReg baseline and a fine-tuned
    transformer classifier."""

    def __init__(
        self,
        baseline_path: str,
        transformer_dir: str,
        baseline_weight: float = 0.3,
        transformer_weight: float = 0.7,
        max_length: int = 384,
        device: Optional[str] = None,
    ):
        self.baseline_weight = baseline_weight
        self.transformer_weight = transformer_weight
        self.max_length = max_length

        # Load baseline pipeline (TfidfVectorizer → LogisticRegression)
        self.baseline = joblib.load(baseline_path)

        # Load transformer
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            transformer_dir
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer.to(self.device)
        self.transformer.eval()

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------

    def _baseline_proba(self, texts: list[str]) -> np.ndarray:
        """Return (N, 2) class probabilities from the baseline pipeline."""
        return self.baseline.predict_proba(texts)

    @torch.inference_mode()
    def _transformer_proba(self, texts: list[str]) -> np.ndarray:
        """Return (N, 2) class probabilities from the transformer."""
        all_probs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.transformer(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Weighted average of class probabilities from both models.

        Returns:
            (N, 2) array of [P(human), P(AI)] per sample.
        """
        p_base = self._baseline_proba(texts)
        p_trans = self._transformer_proba(texts)
        combined = (
            self.baseline_weight * p_base + self.transformer_weight * p_trans
        )
        return combined

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return predicted class labels (0=human, 1=AI)."""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def predict_single(self, text: str) -> Dict:
        """Convenience method for single-text inference with details."""
        proba = self.predict_proba([text])[0]
        label = int(np.argmax(proba))
        return {
            "label": label,
            "label_str": "AI-generated" if label == 1 else "Human-written",
            "confidence": float(proba[label]),
            "prob_human": float(proba[0]),
            "prob_ai": float(proba[1]),
        }


def evaluate_ensemble(
    ensemble: EnsembleDetector,
    texts: list[str],
    labels: np.ndarray,
) -> Dict[str, float]:
    """Evaluate ensemble on labelled data, return metric dict."""
    preds = ensemble.predict(texts)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}
