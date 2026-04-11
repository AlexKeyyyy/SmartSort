from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "models" / "smartsort_transformer"
REPORTS_DIR = BASE_DIR / "reports" / "model_eval" / "transformer"
RANDOM_SEED = 42

DEFAULT_MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_EVAL_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer model for SmartSort file classification."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to dataset.csv")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="HF model name")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to save model")
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR, help="Directory to save metrics")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Tokenizer max_length")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE, help="Eval batch size")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup fraction")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Limit train samples (0 = all).")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Limit validation samples (0 = all).")
    parser.add_argument("--log-interval", type=int, default=20, help="How often to update progress stats.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Speed-first preset (shorter sequence, fewer epochs, smaller subsets).",
    )
    parser.add_argument(
        "--no-auto-quick",
        action="store_true",
        help="Disable automatic quick preset on CPU.",
    )
    parser.add_argument(
        "--freeze-base-model",
        action="store_true",
        help="Freeze transformer encoder and train only classification head for faster runs.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision when CUDA is available.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def featurize_name(filename: str) -> str:
    path = Path(str(filename).lower())
    stem = path.stem.replace("_", " ").replace("-", " ")
    ext = path.suffix.lower().lstrip(".")
    return f"{stem} EXT_{ext}" if ext else stem


def build_text_input(filename: str, content: str) -> str:
    safe_filename = featurize_name(filename)
    safe_content = " ".join(str(content or "").split())
    return f"[FILE] {safe_filename} [CONTENT] {safe_content}".strip()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path).fillna("")
    required_columns = {"filename", "content", "label"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    prepared = df.copy()
    prepared["text"] = [
        build_text_input(filename, content)
        for filename, content in zip(prepared["filename"], prepared["content"])
    ]
    return prepared


def build_label_maps(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted(str(item) for item in labels.unique())
    label2id = {label: index for index, label in enumerate(classes)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


class SmartSortDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.encoded = encoded
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index], dtype=torch.long) for key, value in self.encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


@dataclass
class BatchCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = torch.stack([feature.pop("labels") for feature in features])
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def freeze_backbone(model: AutoModelForSequenceClassification) -> None:
    base_model = getattr(model, model.base_model_prefix, None)
    if base_model is None:
        return
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def stratified_sample(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)

    unique_classes = df["label"].nunique()
    if max_samples < unique_classes:
        raise ValueError(
            f"max_samples={max_samples} is too small for stratification across {unique_classes} classes."
        )

    sampled, _ = train_test_split(
        df,
        train_size=max_samples,
        random_state=seed,
        stratify=df["label"],
    )
    return sampled.reset_index(drop=True)


def apply_speed_preset(args: argparse.Namespace, device: torch.device) -> None:
    use_quick = bool(args.quick)
    auto_quick = device.type == "cpu" and not args.no_auto_quick
    if not use_quick and not auto_quick:
        return

    # Keep explicit user choices when possible, but make defaults practical on CPU.
    if args.epochs == DEFAULT_EPOCHS:
        args.epochs = 2
    args.max_length = min(int(args.max_length), 128)
    if args.max_train_samples <= 0:
        args.max_train_samples = 3000
    if args.max_val_samples <= 0:
        args.max_val_samples = 1000
    args.freeze_base_model = True


def evaluate_model(
    model: AutoModelForSequenceClassification,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Validation", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            losses.append(float(loss.detach().cpu().item()))
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(batch["labels"].detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = float(np.mean(losses)) if losses else math.nan
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": all_labels,
        "preds": all_preds,
    }


def save_reports(
    report_dir: Path,
    metrics: dict[str, Any],
    id2label: dict[int, str],
    model_name: str,
    args: argparse.Namespace,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    target_names = [id2label[index] for index in sorted(id2label)]
    labels_sorted = list(sorted(id2label))

    report_dict = classification_report(
        metrics["labels"],
        metrics["preds"],
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_dir / "classification_report.csv")

    cm = confusion_matrix(metrics["labels"], metrics["preds"], labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(report_dir / "confusion_matrix.csv")

    summary = {
        "model_name": model_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "loss": metrics["loss"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "test_size": args.test_size,
        "freeze_base_model": bool(args.freeze_base_model),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    apply_speed_preset(args, device)

    df = load_dataframe(args.dataset)
    label2id, id2label = build_label_maps(df["label"])
    df["label_id"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    train_df = stratified_sample(train_df, int(args.max_train_samples), RANDOM_SEED)
    val_df = stratified_sample(val_df, int(args.max_val_samples), RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    # Also store integer-key mapping for our own metadata.
    runtime_id2label = {int(key): value for key, value in id2label.items()}

    if args.freeze_base_model:
        freeze_backbone(model)

    model.to(device)

    train_dataset = SmartSortDataset(
        texts=train_df["text"].astype(str).tolist(),
        labels=train_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_dataset = SmartSortDataset(
        texts=val_df["text"].astype(str).tolist(),
        labels=val_df["label_id"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    collator = BatchCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_train_steps = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)) * args.epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_metric = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model_name}")
    print(f"[INFO] Train samples: {len(train_df)} | Validation samples: {len(val_df)}")
    print(f"[INFO] max_length={args.max_length} batch_size={args.batch_size} epochs={args.epochs}")
    print(f"[INFO] freeze_base_model={args.freeze_base_model} fp16={use_fp16}")
    print(f"[INFO] Labels: {label2id}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{args.epochs}",
            leave=False,
        )
        for step, batch in progress:
            batch = move_batch_to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = outputs.loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().cpu().item())

            if step % max(1, args.grad_accum_steps) == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % max(1, int(args.log_interval)) == 0 or step == len(train_loader):
                current_lr = float(scheduler.get_last_lr()[0]) if scheduler.get_last_lr() else float(args.learning_rate)
                progress.set_postfix(loss=f"{running_loss / step:.4f}", lr=f"{current_lr:.2e}")

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start

        epoch_info = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
        }
        history.append(epoch_info)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"time={epoch_seconds:.1f}s"
        )

        current_metric = val_metrics["macro_f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0

            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            metadata = {
                "model_name": args.model_name,
                "label2id": label2id,
                "id2label": runtime_id2label,
                "max_length": args.max_length,
                "text_format": "[FILE] <normalized filename> [CONTENT] <content>",
            }
            (args.output_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            pd.DataFrame(history).to_csv(args.report_dir / "history.csv", index=False)
            save_reports(args.report_dir, val_metrics, runtime_id2label, args.model_name, args)
            print(f"[OK] Saved new best model to: {args.output_dir}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("[INFO] Early stopping triggered.")
                break

    print(f"[OK] Training finished. Best validation macro F1: {best_metric:.4f}")
    print(f"[OK] Model dir: {args.output_dir}")
    print(f"[OK] Report dir: {args.report_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Transformer training failed: {exc}")
        raise
