from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "smartsort_transformer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict SmartSort class with a fine-tuned transformer.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def featurize_name(filename: str) -> str:
    path = Path(str(filename).lower())
    stem = path.stem.replace("_", " ").replace("-", " ")
    ext = path.suffix.lower().lstrip(".")
    return f"{stem} EXT_{ext}" if ext else stem


def build_text_input(filename: str, content: str) -> str:
    safe_filename = featurize_name(filename)
    safe_content = " ".join(str(content or "").split())
    return f"[FILE] {safe_filename} [CONTENT] {safe_content}".strip()


def main() -> None:
    args = parse_args()
    metadata_path = args.model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in model dir: {args.model_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in metadata["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = build_text_input(args.filename, args.content)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=args.max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu()
        pred_id = int(torch.argmax(probs).item())

    result = {
        "predicted_label": id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "probabilities": {id2label[index]: float(probs[index].item()) for index in range(len(probs))},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Prediction failed: {exc}")
        raise
