from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.train_model import RANDOM_SEED, featurize_name


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "smartsort_model.pkl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports" / "model_eval" / "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SmartSort model and save reproducible quality artifacts."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to dataset CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to model .pkl file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifacts output directory.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for split reproducibility.")
    parser.add_argument("--top-errors", type=int, default=25, help="How many misclassified rows to save.")
    return parser.parse_args()


def ensure_inputs(dataset_path: Path, model_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")


def save_text_report(path: Path, report_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_inputs(args.dataset, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    dataset = pd.read_csv(args.dataset).fillna("")
    required_columns = {"filename", "content", "label"}
    missing = required_columns.difference(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    prepared = dataset.copy()
    prepared["filename"] = prepared["filename"].apply(featurize_name)

    X = prepared[["filename", "content"]]
    y = prepared["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    classes = list(model.classes_)
    class_to_index = {name: index for index, name in enumerate(classes)}

    accuracy = float(accuracy_score(y_val, y_pred))
    report_dict = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_val, y_pred, zero_division=0)
    labels = sorted(y.unique().tolist())
    matrix = confusion_matrix(y_val, y_pred, labels=labels)

    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "true_label"
    confusion_df.to_csv(args.output_dir / "confusion_matrix.csv")

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(args.output_dir / "classification_report.csv", index=True)
    (args.output_dir / "classification_report.json").write_text(
        json.dumps(report_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_text_report(args.output_dir / "classification_report.txt", report_text)

    mistakes: list[dict[str, object]] = []
    y_val_reset = y_val.reset_index(drop=True)
    X_val_reset = X_val.reset_index(drop=True)
    for row_index, (true_label, pred_label) in enumerate(zip(y_val_reset.tolist(), y_pred.tolist())):
        if true_label == pred_label:
            continue

        pred_idx = class_to_index[str(pred_label)]
        true_idx = class_to_index[str(true_label)]
        confidence = float(y_proba[row_index][pred_idx])
        true_prob = float(y_proba[row_index][true_idx])

        mistakes.append(
            {
                "row_index": int(row_index),
                "filename": str(X_val_reset.iloc[row_index]["filename"]),
                "true_label": str(true_label),
                "predicted_label": str(pred_label),
                "predicted_confidence": round(confidence, 6),
                "true_label_probability": round(true_prob, 6),
                "margin_pred_minus_true": round(confidence - true_prob, 6),
                "content_preview": str(X_val_reset.iloc[row_index]["content"])[:220].replace("\n", " "),
            }
        )

    mistakes_sorted = sorted(mistakes, key=lambda item: item["predicted_confidence"], reverse=True)
    pd.DataFrame(mistakes_sorted[: max(0, int(args.top_errors))]).to_csv(
        args.output_dir / "top_errors.csv",
        index=False,
    )

    summary = {
        "dataset_path": str(args.dataset.resolve()),
        "model_path": str(args.model.resolve()),
        "validation_rows": int(len(X_val)),
        "train_rows": int(len(X_train)),
        "accuracy": round(accuracy, 6),
        "labels": labels,
        "total_errors": int((y_val != y_pred).sum()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Accuracy: {accuracy:.2%}")
    print(f"[OK] Evaluation artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        raise
