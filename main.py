from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.app import load_config, run_app, save_config


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "smartsort_transformer" / "metadata.json"


def run_module(module_name: str) -> None:
    print(f"[RUN] {module_name}", flush=True)
    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=BASE_DIR,
        check=True,
    )


def ensure_assets() -> None:
    (BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "models").mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        save_config(CONFIG_PATH, load_config(CONFIG_PATH))

    if not MODEL_PATH.exists():
        if not DATASET_PATH.exists():
            run_module("src.generate_dataset")
        run_module("src.train_transformer_model")


def main() -> None:
    ensure_assets()
    config = load_config(CONFIG_PATH)
    run_app(CONFIG_PATH, config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Failed to start SmartSort: {exc}", flush=True)
        raise
