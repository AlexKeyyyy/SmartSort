from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from src.app import DEFAULT_CONFIG, load_config, save_config
from src.mover import FileMover
from src.predictor import SmartSortPredictor
from src.storage import Storage
from src.watcher import FolderWatcher


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "smartsort_transformer"
DEFAULT_OUTPUT_PATH = BASE_DIR / "reports" / "smoke" / "latest" / "smoke_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmartSort smoke checks for final quality gate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to JSON report.",
    )
    parser.add_argument(
        "--run-training-check",
        action="store_true",
        help="Also run full training command as part of smoke checks.",
    )
    return parser.parse_args()


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_predictor_checks(predictor: SmartSortPredictor) -> dict:
    with tempfile.TemporaryDirectory(prefix="smartsort_smoke_predictor_") as tmp:
        root = Path(tmp)
        files = {
            "api_client.py": ("Code", "def fetch_data():\n    return {'ok': True}\n"),
            "meeting_notes.txt": ("Docs", "Meeting notes for weekly planning and release checklist."),
            "invoice_april.txt": ("Finance", "Invoice #2026-04 amount 24000 rub, VAT 20 percent, payment due in 5 days."),
            "lecture_nn.txt": ("Study", "Neural networks lecture: gradient descent and backpropagation examples."),
            "service.log": ("Logs", "2026-04-08 INFO worker processed batch_id=42 latency_ms=23"),
            "family_photo.jpg": ("Media", ""),
            "backup_weekly.zip": ("Archives", ""),
        }

        paths: list[Path] = []
        expected: dict[str, str] = {}
        for name, (label, content) in files.items():
            path = root / name
            expected[name] = label
            if path.suffix.lower() in {".jpg", ".zip"}:
                path.write_bytes(b"binary_demo")
            else:
                path.write_text(content, encoding="utf-8")
            paths.append(path)

        results = predictor.predict_batch(paths)
        assert_true(len(results) == len(paths), "Predictor returned unexpected number of records.")
        matches = 0
        stable_files = {"api_client.py", "service.log", "family_photo.jpg", "backup_weekly.zip"}
        stable_matches = 0
        for result in results:
            assert_true("category" in result and result["category"], "Missing category in predictor output.")
            assert_true("confidence" in result, "Missing confidence in predictor output.")
            confidence = float(result["confidence"])
            assert_true(0.0 <= confidence <= 1.0, "Predictor confidence is out of [0, 1] range.")
            if expected.get(result["filename"]) == result["category"]:
                matches += 1
                if result["filename"] in stable_files:
                    stable_matches += 1

        # Keep this as a smoke test (pipeline sanity), not a hard quality gate.
        assert_true(stable_matches >= 4, f"Rule-based categories mismatch: {stable_matches}/4")
        assert_true(matches >= 4, f"Predictor matched too few expected labels: {matches}/7")

        return {
            "files_tested": len(paths),
            "matched_expected": matches,
            "categories": sorted({item["category"] for item in results}),
        }


def run_watcher_checks(predictor: SmartSortPredictor) -> dict:
    with tempfile.TemporaryDirectory(prefix="smartsort_smoke_watcher_") as tmp:
        watch_dir = Path(tmp) / "watch"
        watch_dir.mkdir(parents=True, exist_ok=True)

        events: list[dict] = []

        def callback(payload: dict) -> None:
            events.append(payload)

        watcher = FolderWatcher(watch_dir, callback, predictor)
        watcher.start()
        assert_true(watcher.is_running(), "Watcher failed to start.")

        hidden = watch_dir / ".tmp_ignore.txt"
        hidden.write_text("hidden", encoding="utf-8")
        time.sleep(0.5)

        target = watch_dir / "manual_notes.txt"
        target.write_text("Manual notes and project checklist for the next sprint.", encoding="utf-8")

        started = time.time()
        while time.time() - started < 6.0:
            if any(item.get("filename") == target.name for item in events):
                break
            time.sleep(0.2)

        watcher.stop()
        assert_true(not watcher.is_running(), "Watcher failed to stop.")
        assert_true(any(item.get("filename") == target.name for item in events), "Watcher did not emit target file event.")
        assert_true(all(item.get("filename") != hidden.name for item in events), "Watcher did not ignore hidden file.")

        return {"events_received": len(events)}


def run_mover_storage_checks() -> dict:
    with tempfile.TemporaryDirectory(prefix="smartsort_smoke_mover_") as tmp:
        root = Path(tmp)
        incoming_dir = root / "incoming"
        sorted_dir = root / "sorted"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        sorted_dir.mkdir(parents=True, exist_ok=True)

        sample = incoming_dir / "report.txt"
        sample.write_text("Quarterly report draft", encoding="utf-8")

        storage = Storage(root / "smoke.db")
        mover = FileMover(sorted_dir, storage)

        moved_to = Path(mover.move_file(sample, "Docs", 0.88))
        assert_true(moved_to.exists(), "Moved file does not exist.")
        assert_true(not sample.exists(), "Source file still exists after move.")

        undo_ok = mover.undo_last()
        assert_true(undo_ok, "Undo for last move failed.")
        assert_true(sample.exists(), "Source file was not restored after undo.")
        assert_true(not moved_to.exists(), "Destination file still exists after undo.")

        undo_again = mover.undo_last()
        assert_true(not undo_again, "Second undo should return False when nothing to undo.")

        return {
            "moves_logged": len(storage.get_recent(limit=10)),
            "undo_second_attempt": undo_again,
        }


def run_config_checks() -> dict:
    with tempfile.TemporaryDirectory(prefix="smartsort_smoke_config_") as tmp:
        path = Path(tmp) / "config.json"
        initial = load_config(path)
        assert_true(initial == DEFAULT_CONFIG, "Default config shape mismatch.")

        initial["watch_dir"] = "/tmp/watch_demo"
        initial["output_dir"] = "/tmp/out_demo"
        initial["confidence_threshold"] = 0.63
        initial["auto_move"] = True
        save_config(path, initial)

        restored = load_config(path)
        assert_true(restored["watch_dir"] == "/tmp/watch_demo", "watch_dir was not persisted.")
        assert_true(restored["output_dir"] == "/tmp/out_demo", "output_dir was not persisted.")
        assert_true(float(restored["confidence_threshold"]) == 0.63, "confidence_threshold was not persisted.")
        assert_true(bool(restored["auto_move"]) is True, "auto_move was not persisted.")

        return {"config_path": str(path)}


def run_training_check() -> dict:
    subprocess.run(
        [sys.executable, "-m", "src.train_transformer_model", "--quick"],
        cwd=BASE_DIR,
        check=True,
    )
    assert_true((MODEL_PATH / "metadata.json").exists(), "Transformer model metadata is missing after training check.")
    return {"model_path": str(MODEL_PATH)}


def main() -> None:
    args = parse_args()
    checks: dict[str, dict] = {}

    if not (MODEL_PATH / "metadata.json").exists():
        raise FileNotFoundError(
            "Model not found: "
            f"{MODEL_PATH}. "
            "Train model first (`python -m src.train_open_transformer_model` or `python -m src.train_transformer_model`)."
        )

    predictor = SmartSortPredictor(MODEL_PATH)
    checks["predictor"] = run_predictor_checks(predictor)
    checks["watcher"] = run_watcher_checks(predictor)
    checks["mover_storage"] = run_mover_storage_checks()
    checks["config"] = run_config_checks()
    if args.run_training_check:
        checks["training"] = run_training_check()

    report = {
        "status": "ok",
        "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(BASE_DIR),
        "checks": checks,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Smoke checks passed.")
    print(f"[OK] Report saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Smoke checks failed: {exc}")
        raise
