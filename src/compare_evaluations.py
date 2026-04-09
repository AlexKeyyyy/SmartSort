from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = BASE_DIR / "reports" / "model_eval" / "baseline"
DEFAULT_CANDIDATE_DIR = BASE_DIR / "reports" / "model_eval" / "ru_extended"
DEFAULT_OUTPUT = BASE_DIR / "reports" / "model_eval" / "comparison_ru_extended.md"
FOCUS_LABELS = ["Docs", "Finance", "Study"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two SmartSort evaluation runs and create a markdown summary."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    file = path / "summary.json"
    if not file.exists():
        raise FileNotFoundError(f"Missing summary file: {file}")
    return json.loads(file.read_text(encoding="utf-8"))


def load_report(path: Path) -> pd.DataFrame:
    file = path / "classification_report.csv"
    if not file.exists():
        raise FileNotFoundError(f"Missing classification report file: {file}")
    return pd.read_csv(file).rename(columns={"Unnamed: 0": "label"}).set_index("label")


def get_metric(df: pd.DataFrame, label: str, metric: str) -> float | None:
    if label not in df.index or metric not in df.columns:
        return None
    return float(df.loc[label, metric])


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def main() -> None:
    args = parse_args()
    base_summary = load_summary(args.baseline_dir)
    cand_summary = load_summary(args.candidate_dir)
    base_report = load_report(args.baseline_dir)
    cand_report = load_report(args.candidate_dir)

    lines: list[str] = []
    lines.append("# SmartSort Evaluation Comparison")
    lines.append("")
    lines.append(f"- baseline: `{args.baseline_dir}`")
    lines.append(f"- candidate: `{args.candidate_dir}`")
    lines.append("")

    base_acc = float(base_summary.get("accuracy", 0.0))
    cand_acc = float(cand_summary.get("accuracy", 0.0))
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- baseline accuracy: {base_acc:.6f}")
    lines.append(f"- candidate accuracy: {cand_acc:.6f}")
    lines.append(f"- delta accuracy: {fmt_delta(cand_acc - base_acc)}")
    lines.append("")

    for aggregate_label in ["macro avg", "weighted avg"]:
        lines.append(f"## {aggregate_label}")
        lines.append("")
        for metric in ["precision", "recall", "f1-score"]:
            base_val = get_metric(base_report, aggregate_label, metric)
            cand_val = get_metric(cand_report, aggregate_label, metric)
            delta = None if base_val is None or cand_val is None else cand_val - base_val
            lines.append(
                f"- {metric}: baseline={base_val:.4f} candidate={cand_val:.4f} delta={fmt_delta(delta)}"
                if base_val is not None and cand_val is not None
                else f"- {metric}: n/a"
            )
        lines.append("")

    lines.append("## Focus Classes (Docs / Finance / Study)")
    lines.append("")
    for label in FOCUS_LABELS:
        lines.append(f"### {label}")
        for metric in ["precision", "recall", "f1-score"]:
            base_val = get_metric(base_report, label, metric)
            cand_val = get_metric(cand_report, label, metric)
            delta = None if base_val is None or cand_val is None else cand_val - base_val
            lines.append(
                f"- {metric}: baseline={base_val:.4f} candidate={cand_val:.4f} delta={fmt_delta(delta)}"
                if base_val is not None and cand_val is not None
                else f"- {metric}: n/a"
            )
        lines.append("")

    lines.append("## Decision Hint")
    lines.append("")
    lines.append(
        "- Candidate is acceptable if overall stability is maintained and focus-class metrics "
        "improve or stay neutral without smoke regressions."
    )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Comparison report saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Comparison failed: {exc}")
        raise
