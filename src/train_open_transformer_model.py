from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "open_dataset_manifest.ru_extended.json"
DEFAULT_MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "smartsort_transformer"
DEFAULT_REPORT_DIR = BASE_DIR / "reports" / "model_eval" / "transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SmartSort dataset and fine-tune a transformer classifier."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--source-retries", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-auto-quick", action="store_true")
    parser.add_argument("--freeze-base-model", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def run_module(module_name: str, *extra_args: str) -> None:
    print(f"[RUN] {module_name} {' '.join(extra_args)}".strip(), flush=True)
    subprocess.run(
        [sys.executable, "-m", module_name, *extra_args],
        cwd=BASE_DIR,
        check=True,
    )


def main() -> None:
    args = parse_args()

    run_module("src.generate_ru_seed_corpus")
    run_module(
        "src.build_open_dataset",
        "--manifest",
        str(args.manifest),
        "--source-retries",
        str(args.source_retries),
    )

    train_args = [
        "--model-name", args.model_name,
        "--output-dir", str(args.output_dir),
        "--report-dir", str(args.report_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--max-length", str(args.max_length),
        "--learning-rate", str(args.learning_rate),
        "--max-train-samples", str(args.max_train_samples),
        "--max-val-samples", str(args.max_val_samples),
        "--log-interval", str(args.log_interval),
        "--num-workers", str(args.num_workers),
    ]
    if args.quick:
        train_args.append("--quick")
    if args.no_auto_quick:
        train_args.append("--no-auto-quick")
    if args.freeze_base_model:
        train_args.append("--freeze-base-model")
    if args.fp16:
        train_args.append("--fp16")

    run_module("src.train_transformer_model", *train_args)


if __name__ == "__main__":
    main()
