# Detecting AI-Generated Text

[![CI](https://github.com/fahadalidev/ai-text-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/fahadalidev/ai-text-detector/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

An end-to-end NLP system that distinguishes **human-written** from **AI-generated** text using a multi-model ensemble approach. Combines classical ML with modern transformers, exposes a production-ready REST API, and ships with an interactive web demo.

---

## Key Features

- **Dual-model architecture** — TF-IDF + Logistic Regression baseline and fine-tuned DistilBERT
- **Weighted ensemble** — Soft-voting combination of both models for improved accuracy
- **Linguistic feature extraction** — Stylometric analysis (readability, vocabulary richness, entropy)
- **REST API** — FastAPI service with single and batch prediction endpoints
- **Interactive demo** — Gradio web UI with real-time classification and feature visualization
- **Containerized** — Docker and docker-compose for one-command deployment
- **CI/CD** — GitHub Actions pipeline with linting, testing, and Docker build validation
- **Config-driven** — All hyperparameters centralized in YAML

---

## Architecture

```
                    Input Text
                        |
            +-----------+-----------+
            |                       |
     TF-IDF + LogReg         DistilBERT (fine-tuned)
     (baseline model)        (transformer model)
            |                       |
        P(class)               P(class)
            |                       |
            +-------Ensemble--------+
            |  weighted soft vote   |
            +-----------+-----------+
                        |
              Final Prediction
         (Human / AI-generated)
                        |
          Linguistic Feature Analysis
    (readability, entropy, vocabulary)
```

---

## Project Structure

```
.
├── src/                          # Core library modules
│   ├── data.py                   # Data loading and schema normalization
│   ├── baseline.py               # TF-IDF + Logistic Regression pipeline
│   ├── transformer.py            # DistilBERT fine-tuning with HuggingFace Trainer
│   ├── ensemble.py               # Weighted soft-voting ensemble detector
│   ├── features.py               # Linguistic/stylometric feature extraction
│   ├── hf_dataset.py             # HuggingFace Dataset utilities
│   ├── eda.py                    # Exploratory data analysis
│   └── utils.py                  # Seed setting, metrics, I/O helpers
│
├── scripts/                      # Training and evaluation scripts
│   ├── download_data.py          # Kaggle dataset downloader
│   ├── prepare_splits.py         # Stratified train/valid/test splitting
│   ├── run_baseline.py           # Train baseline model
│   ├── run_transformer.py        # Fine-tune DistilBERT
│   ├── run_ensemble.py           # Evaluate ensemble on splits
│   ├── report_metrics.py         # Generate metrics and confusion matrices
│   └── error_analysis.py         # Analyze misclassifications with confidence
│
├── api/                          # FastAPI REST service
│   └── app.py                    # /predict, /predict/batch, /health endpoints
│
├── tests/                        # Unit test suite (pytest)
│   ├── test_features.py          # Feature extraction tests
│   ├── test_data.py              # Data pipeline tests
│   ├── test_utils.py             # Utility function tests
│   └── test_api.py               # API schema validation tests
│
├── configs/
│   └── config.yaml               # Centralized hyperparameters
│
├── app.py                        # Gradio interactive demo
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Multi-service orchestration
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── requirements.txt              # Python dependencies
└── environment.yml               # Conda environment specification
```

---

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/fahadalidev/ai-text-detector.git
cd ai-text-detector

# Option A: Conda (recommended for GPU)
conda env create -f environment.yml
conda activate sea820

# Option B: pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset

```bash
# Requires Kaggle API key in ~/.kaggle/kaggle.json
python scripts/download_data.py
python scripts/prepare_splits.py
```

### 3. Train models

```bash
# Train baseline (TF-IDF + Logistic Regression) — ~2 min on CPU
python scripts/run_baseline.py

# Fine-tune DistilBERT — ~1.5 hrs on GPU, ~6 hrs on CPU
python scripts/run_transformer.py

# Evaluate ensemble
python scripts/run_ensemble.py
```

### 4. Run inference

```bash
# REST API
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Visit http://localhost:8000/docs for interactive Swagger UI

# Gradio demo
python app.py
# Visit http://localhost:7860
```

---

## API Reference

### `POST /predict`

Classify a single text sample.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text to classify."}'
```

Response:
```json
{
  "label": "Human-written",
  "confidence": 0.92,
  "prob_human": 0.92,
  "prob_ai": 0.08,
  "features": {
    "word_count": 8,
    "ttr": 1.0,
    "flesch_reading_ease": 82.5,
    "word_entropy": 3.0
  },
  "latency_ms": 45.2
}
```

### `POST /predict/batch`

Classify up to 64 texts in one request.

### `GET /health`

Health check endpoint.

---

## Docker Deployment

```bash
# Build and run both API and demo
docker-compose up --build

# API: http://localhost:8000
# Demo: http://localhost:7860
```

---

## Model Details

| Component | Description |
|-----------|-------------|
| **Baseline** | TF-IDF vectorizer (200K features, bigrams) with Logistic Regression (C=2.0) |
| **Transformer** | DistilBERT-base-uncased fine-tuned for 3 epochs (lr=2e-5, fp16) |
| **Ensemble** | 0.3 x baseline + 0.7 x transformer soft-voting |
| **Features** | Flesch readability, TTR, hapax ratio, sentence stats, word entropy |

### Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Baseline** | 0.97 | 0.96 | 0.98 | 0.97 |
| **Transformer** | 0.99 | 0.99 | 0.99 | 0.99 |
| **Ensemble** | 0.99 | 0.99 | 0.99 | 0.99 |

*Results on the AI vs Human Text dataset (Kaggle). Exact numbers depend on random seed and split.*

---

## Linguistic Features

The feature extraction module (`src/features.py`) computes:

| Feature | Description |
|---------|-------------|
| Type-Token Ratio (TTR) | Vocabulary diversity — unique words / total words |
| Hapax Legomena Ratio | Proportion of words appearing exactly once |
| Flesch Reading Ease | Text readability (higher = easier to read) |
| Flesch-Kincaid Grade | US school grade level needed to understand the text |
| Sentence Length Stats | Mean and standard deviation of sentence lengths |
| Punctuation Rates | Normalized frequency of commas, periods, questions, etc. |
| Word Entropy | Shannon entropy of the word distribution |

---

## Error Analysis

The error analysis script identifies the most confident misclassifications:

- **False Positives** — AI text incorrectly classified as human (often well-crafted AI essays)
- **False Negatives** — Human text incorrectly classified as AI (often short or formulaic writing)

```bash
python scripts/error_analysis.py -k 50
# Outputs: validation_misclassified.csv, top-50 confident wrong predictions
```

---

## Testing

```bash
# Run full test suite
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_features.py -v
```

---

## Ethical Considerations

- **Bias risk**: May disproportionately flag non-native English speakers or writers with formulaic styles
- **Misuse potential**: Should not be used as sole evidence for academic integrity decisions
- **Transparency**: Confidence scores and feature breakdowns help users understand predictions
- **Limitations**: Trained on specific AI models; may not generalize to newer generators

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/NLP** | PyTorch, HuggingFace Transformers, scikit-learn |
| **Data** | Pandas, NumPy, HuggingFace Datasets |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Demo** | Gradio |
| **Visualization** | Matplotlib, Seaborn |
| **DevOps** | Docker, GitHub Actions, pytest, Ruff |
| **Config** | YAML, argparse |

---

## Author

**Fahad Ali Khan**

---

## License

This project is licensed under the MIT License.
