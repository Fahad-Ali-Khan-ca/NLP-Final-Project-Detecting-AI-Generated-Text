# api/app.py
"""FastAPI service for AI-generated text detection.

Exposes endpoints for single and batch prediction using the ensemble
detector (TF-IDF baseline + DistilBERT transformer).

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ensemble import EnsembleDetector
from src.features import extract_features

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_detector: Optional[EnsembleDetector] = None


def _load_detector() -> EnsembleDetector:
    cfg_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    baseline_path = os.path.join("models", "baseline", "tfidf_logreg.joblib")
    transformer_dir = os.path.join("models", "transformer")

    return EnsembleDetector(
        baseline_path=baseline_path,
        transformer_dir=transformer_dir,
        baseline_weight=float(os.getenv("BASELINE_WEIGHT", "0.3")),
        transformer_weight=float(os.getenv("TRANSFORMER_WEIGHT", "0.7")),
        max_length=int(cfg["transformer"]["max_length"]),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    _detector = _load_detector()
    yield
    _detector = None


# ---------------------------------------------------------------------------
# App & schemas
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Text Detector API",
    description="Detect whether text is human-written or AI-generated using an ensemble of TF-IDF + DistilBERT models.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    prob_human: float
    prob_ai: float
    features: dict
    latency_ms: float


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class BatchItem(BaseModel):
    text: str
    label: str
    confidence: float
    prob_human: float
    prob_ai: float


class BatchResponse(BaseModel):
    results: List[BatchItem]
    count: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _detector is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _detector is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()
    result = _detector.predict_single(req.text)
    features = extract_features(req.text)
    elapsed = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=result["label_str"],
        confidence=result["confidence"],
        prob_human=result["prob_human"],
        prob_ai=result["prob_ai"],
        features=features,
        latency_ms=round(elapsed, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest):
    if _detector is None:
        raise HTTPException(503, "Model not loaded")

    start = time.perf_counter()
    proba = _detector.predict_proba(req.texts)
    elapsed = (time.perf_counter() - start) * 1000

    items = []
    for i, text in enumerate(req.texts):
        label_idx = int(proba[i].argmax())
        items.append(BatchItem(
            text=text[:200],
            label="AI-generated" if label_idx == 1 else "Human-written",
            confidence=float(proba[i][label_idx]),
            prob_human=float(proba[i][0]),
            prob_ai=float(proba[i][1]),
        ))

    return BatchResponse(
        results=items,
        count=len(items),
        latency_ms=round(elapsed, 2),
    )
