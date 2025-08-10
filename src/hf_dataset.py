from datasets import load_dataset

def load_splits_from_csv(splits_dir: str, text_col: str = "text", label_col: str = "label"):
    data_files = {
        "train": f"{splits_dir}/train.csv",
        "validation": f"{splits_dir}/valid.csv",
        "test": f"{splits_dir}/test.csv",
    }
    ds = load_dataset("csv", data_files=data_files)
    # Ensure column names are standardized
    def rename_cols(example):
        return {"text": example[text_col], "label": int(example[label_col])}
    ds = ds.map(lambda x: rename_cols(x))
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in ["text", "label"]])
    ds = ds.class_encode_column("label")
    return ds