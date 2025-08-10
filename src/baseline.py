# src/baseline.py
import os, joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import ensure_dir, compute_metrics_from_preds, save_json

def train_baseline(
    train_csv: str,
    valid_csv: str,
    model_dir: str,
    metrics_out: str,
    plots_dir: str,
    max_features=200000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    lowercase=True,
    C=2.0,
    max_iter=2000,
    n_jobs=-1,
):
    """Train TF-IDF + Logistic Regression baseline and save artifacts."""
    ensure_dir(model_dir); ensure_dir(os.path.dirname(metrics_out)); ensure_dir(plots_dir)

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)


    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["text", "label"]).copy()
        # normalize newlines and trim
        df["text"] = (
            df["text"].astype(str)
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )
        df = df[df["text"] != ""]
        return df

    train_df = _clean(train_df)
    valid_df = _clean(valid_df)



    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
        )),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=n_jobs)),
    ])

    pipe.fit(train_df["text"], train_df["label"])

    # Save model
    joblib.dump(pipe, os.path.join(model_dir, "tfidf_logreg.joblib"))

    # Validate
    val_pred = pipe.predict(valid_df["text"])
    metrics = compute_metrics_from_preds(valid_df["label"], val_pred)
    save_json(metrics, metrics_out)

    # Classification report
    report = classification_report(valid_df["label"], val_pred, digits=4)
    with open(os.path.join(model_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion matrix plot
    import matplotlib.pyplot as plt
    import numpy as np
    cm = confusion_matrix(valid_df["label"], val_pred)
    plt.figure()
    im = plt.imshow(cm)
    plt.title("Baseline Confusion Matrix (valid)")
    plt.xlabel("Pred"); plt.ylabel("True")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha="center", va="center")
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_confusion_valid.png"))

    return metrics
