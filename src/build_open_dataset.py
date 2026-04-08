from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.generate_dataset import (
    RANDOM_SEED,
    build_code_sample,
    build_docs_sample,
    build_finance_sample,
    build_logs_sample,
    build_archives_sample,
    build_media_sample,
    build_other_sample,
    build_study_sample,
)

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency in runtime
    load_dataset = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = BASE_DIR / "data" / "open_dataset_manifest.example.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "dataset.csv"
ALLOWED_LABELS = {"Code", "Docs", "Finance", "Study", "Media", "Archives", "Logs", "Other"}
TEXT_EXTENSIONS = {
    "Code": "txt",
    "Docs": "txt",
    "Finance": "txt",
    "Study": "txt",
    "Logs": "log",
    "Other": "txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SmartSort dataset from open datasets described in a manifest file."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to JSON manifest with dataset sources.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Target CSV path for SmartSort dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=BASE_DIR / "data" / "hf_cache",
        help="Local cache directory for downloaded Hugging Face datasets.",
    )
    parser.add_argument(
        "--source-retries",
        type=int,
        default=2,
        help="How many times to retry one source before skipping it (if not required).",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        raise ValueError("Manifest must contain a 'sources' array.")

    return manifest


def load_table(path: Path, file_format: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    fmt = (file_format or suffix.lstrip(".")).lower()

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt in {"jsonl", "ndjson"}:
        return pd.read_json(path, lines=True)
    if fmt == "json":
        return pd.read_json(path)
    if fmt == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported source format for {path}: {fmt}")


def normalize_text(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(normalize_text(item) for item in value if normalize_text(item))
    text = str(value or "")
    return " ".join(text.replace("\x00", " ").split())


def ensure_filename(value: str, label: str, row_index: int) -> str:
    cleaned = value.strip()
    if cleaned:
        return cleaned[:180]

    extension = TEXT_EXTENSIONS.get(label, "txt")
    return f"{label.lower()}_sample_{row_index}.{extension}"


def first_present(row: dict, candidates: list[str]) -> str:
    lowered = {str(key).lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is None:
            continue
        value = normalize_text(row.get(key, ""))
        if value:
            return value
    return ""


def collect_present(row: dict, candidates: list[str]) -> list[str]:
    lowered = {str(key).lower(): key for key in row.keys()}
    values: list[str] = []
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is None:
            continue
        value = normalize_text(row.get(key, ""))
        if value:
            values.append(value)
    return values


def row_matches_filters(row: dict, filters: dict[str, object]) -> bool:
    lowered = {str(key).lower(): key for key in row.keys()}
    for expected_key, expected_value in filters.items():
        actual_key = lowered.get(expected_key.lower())
        if actual_key is None:
            return False
        actual_value = normalize_text(row.get(actual_key, ""))
        if actual_value.lower() != normalize_text(expected_value).lower():
            return False
    return True


def iter_huggingface_rows(source: dict, cache_dir: Path) -> Iterable[dict]:
    if load_dataset is None:
        raise RuntimeError(
            "Hugging Face datasets support requires the 'datasets' package. Install requirements first."
        )

    dataset_id = source["dataset"]
    config_name = source.get("config")
    split_name = source.get("split", "train")

    try:
        dataset = load_dataset(
            dataset_id,
            name=config_name,
            split=split_name,
            streaming=True,
            cache_dir=str(cache_dir),
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message:
            raise RuntimeError(
                f"Dataset '{dataset_id}' uses legacy loading scripts and is not supported by the installed "
                "datasets version. Replace this source in the manifest with a non-script dataset "
                "(or downgrade datasets to <4.0 if you really need this source)."
            ) from exc
        raise

    sample_limit = int(source.get("sample_limit", 0) or 0)
    if sample_limit > 0:
        dataset = dataset.shuffle(seed=RANDOM_SEED, buffer_size=max(1000, sample_limit * 3))

    count = 0
    filters = source.get("filters", {})
    for row in dataset:
        if filters and not row_matches_filters(row, filters):
            continue
        yield dict(row)
        count += 1
        if sample_limit > 0 and count >= sample_limit:
            break


def iter_local_rows(source: dict) -> Iterable[dict]:
    source_path = Path(source["path"])
    if not source_path.is_absolute():
        source_path = (BASE_DIR / source_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    frame = load_table(source_path, source.get("format")).fillna("")
    if source.get("drop_duplicates", True):
        frame = frame.drop_duplicates()

    sample_limit = int(source.get("sample_limit", 0) or 0)
    if sample_limit > 0 and len(frame) > sample_limit:
        frame = frame.sample(n=sample_limit, random_state=RANDOM_SEED)

    for row in frame.to_dict(orient="records"):
        yield {str(key): value for key, value in row.items()}


def iter_source_rows(source: dict, cache_dir: Path) -> Iterable[dict]:
    kind = source.get("kind", "local_file")
    if kind == "huggingface":
        return iter_huggingface_rows(source, cache_dir)
    if kind == "local_file":
        return iter_local_rows(source)
    raise ValueError(f"Unsupported source kind: {kind}")


def build_rows_from_source(source: dict, cache_dir: Path) -> list[dict[str, str]]:
    label = source.get("label")
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Unsupported label: {label}")

    filename_candidates = source.get("filename_columns", [])
    content_candidates = source.get("content_columns", [])
    if not content_candidates:
        raise ValueError(f"Source for label {label} must define non-empty 'content_columns'.")

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    min_content_length = int(source.get("min_content_length", 20))

    for row_index, row in enumerate(iter_source_rows(source, cache_dir)):
        filename = first_present(row, filename_candidates)
        if not filename and source.get("filename_from_content_prefix", False):
            filename = first_present(row, content_candidates)[:40]
        filename = ensure_filename(filename, label, row_index)

        content_values = collect_present(row, content_candidates)
        content = normalize_text(" ".join(content_values))
        if len(content) < min_content_length:
            continue

        item = {
            "filename": filename,
            "content": content[:4000],
            "label": label,
        }
        dedupe_key = (item["filename"], item["content"], item["label"])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(item)

    return rows


def append_synthetic_rows(rows: list[dict[str, str]], counts: dict[str, int]) -> None:
    builders = {
        "Code": build_code_sample,
        "Docs": build_docs_sample,
        "Finance": build_finance_sample,
        "Study": build_study_sample,
        "Logs": build_logs_sample,
        "Media": build_media_sample,
        "Archives": build_archives_sample,
        "Other": build_other_sample,
    }

    for label, count in counts.items():
        if label not in builders or count <= 0:
            continue

        builder = builders[label]
        for index in range(count):
            filename, content = builder(index)
            rows.append(
                {
                    "filename": filename,
                    "content": content,
                    "label": label,
                }
            )


def rebalance_rows(rows: list[dict[str, str]], target_counts: dict[str, int]) -> list[dict[str, str]]:
    if not target_counts:
        return rows

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    balanced: list[dict[str, str]] = []
    for label, label_rows in grouped.items():
        target_count = int(target_counts.get(label, len(label_rows)))
        if target_count <= 0:
            continue

        if len(label_rows) > target_count:
            sampled = random.Random(RANDOM_SEED).sample(label_rows, target_count)
            balanced.extend(sampled)
            continue

        balanced.extend(label_rows)
        if len(label_rows) < target_count:
            print(
                f"[WARN] Label {label} has only {len(label_rows)} rows, "
                f"below target {target_count}. Keeping all available rows."
            )

    return balanced


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    rows: list[dict[str, str]] = []
    loaded_labels: set[str] = set()
    source_retries_default = max(0, int(args.source_retries))
    for source in manifest["sources"]:
        source_name = source.get("dataset") or source.get("path") or source.get("label")
        source_label = source.get("label", "unknown")
        source_required = bool(source.get("required", False))
        source_retries = max(0, int(source.get("retries", source_retries_default)))

        source_rows: list[dict[str, str]] = []
        last_error: Exception | None = None
        for attempt in range(source_retries + 1):
            try:
                source_rows = build_rows_from_source(source, args.cache_dir)
                break
            except Exception as exc:
                last_error = exc
                if attempt < source_retries:
                    print(
                        f"[WARN] Failed source {source_name} (attempt {attempt + 1}/{source_retries + 1}): {exc}"
                    )
                    continue

        if last_error is not None and not source_rows:
            if source_required:
                raise RuntimeError(
                    f"Required source failed after {source_retries + 1} attempts: {source_name}"
                ) from last_error
            print(f"[WARN] Skipped source {source_name} after failures: {last_error}")
            continue

        print(f"[INFO] Loaded {len(source_rows)} rows for {source_label} from {source_name}")
        if source_rows:
            loaded_labels.add(source_label)
            rows.extend(source_rows)

    append_synthetic_rows(rows, manifest.get("synthetic_counts", {}))
    rows = rebalance_rows(rows, manifest.get("label_target_counts", {}))

    if not rows:
        raise ValueError("No rows were collected. Check dataset settings and content columns.")

    missing_labels = ALLOWED_LABELS.difference(loaded_labels).difference({"Media", "Archives", "Other"})
    if missing_labels:
        print(
            f"[WARN] No direct sources loaded for labels: {sorted(missing_labels)}. "
            "Check source availability/network if quality drops."
        )

    dataset = pd.DataFrame(rows, columns=["filename", "content", "label"])
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"[OK] Dataset created: {len(dataset)} rows")
    print(dataset["label"].value_counts().sort_index())
    print(f"[OK] Saved to: {args.output}")


if __name__ == "__main__":
    main()
