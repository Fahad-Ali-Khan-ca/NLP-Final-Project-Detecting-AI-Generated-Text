import os, yaml, argparse
from src.transformer import fine_tune_transformer
from src.utils import gpu_info_str


def main(override):
    print(gpu_info_str())
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg["transformer"]
    # CLI overrides
    for k, v in override.items():
        if v is not None and k in tcfg:
            tcfg[k] = v

    metrics = fine_tune_transformer(
        splits_dir=cfg["paths"]["splits_dir"],
        model_name=tcfg["model_name"],
        out_dir=os.path.join("models", "transformer"),
        max_length=tcfg["max_length"],
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        epochs=int(tcfg["epochs"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        fp16=bool(tcfg["fp16"]),
        gradient_checkpointing=bool(tcfg["gradient_checkpointing"]),
        logging_steps=int(tcfg["logging_steps"]),
    )
    print("Transformer test:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int, dest="per_device_train_batch_size")
    ap.add_argument("--eval-batch-size", type=int, dest="per_device_eval_batch_size")
    ap.add_argument("--grad-accum", type=int, dest="gradient_accumulation_steps")
    ap.add_argument("--max-length", type=int)
    args = ap.parse_args()
    main(vars(args))