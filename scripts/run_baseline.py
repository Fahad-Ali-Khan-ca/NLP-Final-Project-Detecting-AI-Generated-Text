import os, yaml
from src.baseline import train_baseline
from src.utils import gpu_info_str

if __name__ == "__main__":
    print(gpu_info_str())
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    metrics = train_baseline(
        train_csv=os.path.join(paths["splits_dir"], "train.csv"),
        valid_csv=os.path.join(paths["splits_dir"], "valid.csv"),
        model_dir=os.path.join("models", "baseline"),
        metrics_out=os.path.join("outputs", "metrics", "baseline_valid.json"),
        plots_dir=os.path.join("outputs", "plots"),
        max_features=cfg["classic"]["tfidf"]["max_features"],
        ngram_range=tuple(cfg["classic"]["tfidf"]["ngram_range"]),
        min_df=cfg["classic"]["tfidf"]["min_df"],
        max_df=cfg["classic"]["tfidf"]["max_df"],
        sublinear_tf=cfg["classic"]["tfidf"]["sublinear_tf"],
        lowercase=cfg["classic"]["tfidf"]["lowercase"],
        C=cfg["classic"]["logreg"]["C"],
        max_iter=cfg["classic"]["logreg"]["max_iter"],
        n_jobs=cfg["classic"]["logreg"]["n_jobs"],
    )
    print("Baseline valid:", metrics)