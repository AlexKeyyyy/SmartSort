from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "open_dataset_manifest.example.json"
DEFAULT_EVAL_OUTPUT_DIR = BASE_DIR / "reports" / "model_eval" / "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download open datasets, build SmartSort dataset.csv, and train the model."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest file with the selected datasets.",
    )
    parser.add_argument(
        "--source-retries",
        type=int,
        default=2,
        help="Retries per source while building open dataset.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run model evaluation and save reproducible artifacts after training.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Directory for evaluation artifacts (used with --evaluate).",
    )
    parser.add_argument(
        "--smoke-checks",
        action="store_true",
        help="Run smoke checks after training/evaluation.",
    )
    parser.add_argument(
        "--smoke-training-check",
        action="store_true",
        help="Run full train command inside smoke checks (longer).",
    )
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
    run_module("src.train_model")
    if args.evaluate:
        run_module("src.evaluate_model", "--output-dir", str(args.eval_output_dir))
    if args.smoke_checks:
        smoke_args: list[str] = []
        if args.smoke_training_check:
            smoke_args.append("--run-training-check")
        run_module("src.smoke_checks", *smoke_args)


if __name__ == "__main__":
    main()
