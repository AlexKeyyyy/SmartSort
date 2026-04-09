from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DEMO_DIR = BASE_DIR / "data" / "demo_pack"
INPUT_DIR = DEMO_DIR / "input"
MANIFEST_PATH = DEMO_DIR / "expected_labels.json"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[str, str, str]] = [
        ("api_router.py", "Code", "def route(request):\n    return {'status': 'ok', 'path': request.path}\n"),
        ("meeting_protocol.txt", "Docs", "Protocol: weekly sync, roadmap update, task owners and deadlines."),
        ("invoice_march_2026.txt", "Finance", "Invoice 2026-03 amount 54100 rub. VAT 20 percent. Payment due 5 days."),
        ("lecture_backprop.txt", "Study", "Lecture notes about backpropagation, gradients and model regularization."),
        ("server_worker.log", "Logs", "2026-04-08 INFO queue worker started request_id=req-889"),
    ]

    # Binary placeholders for extension-driven categories.
    binary_samples: list[tuple[str, str, bytes]] = [
        ("promo_banner.jpg", "Media", b"\xff\xd8\xff\xe0\x00\x10JFIF"),
        ("backup_snapshot.zip", "Archives", b"PK\x03\x04"),
    ]

    # Intentionally ambiguous file for "Other".
    samples.append(("unsorted_blob.tmp", "Other", "random token stream xyz qwe alpha beta"))

    expected: dict[str, str] = {}
    for filename, label, content in samples:
        write_text(INPUT_DIR / filename, content)
        expected[filename] = label

    for filename, label, content in binary_samples:
        write_binary(INPUT_DIR / filename, content)
        expected[filename] = label

    MANIFEST_PATH.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text(
        DEMO_DIR / "README.txt",
        (
            "SmartSort demo pack\n"
            "1) Put files from ./input into your watch folder.\n"
            "2) Start watcher and show predicted categories.\n"
            "3) Compare with expected_labels.json during demo.\n"
        ),
    )

    print(f"[OK] Demo files prepared: {len(expected)}")
    print(f"[OK] Input dir: {INPUT_DIR}")
    print(f"[OK] Expected labels: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
