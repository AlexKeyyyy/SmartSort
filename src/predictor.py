from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.extractor import extract_text

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:  # pragma: no cover - depends on environment
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None


CATEGORY_ICONS = {
    "Code": "💻",
    "Docs": "📄",
    "Finance": "💰",
    "Study": "📚",
    "Media": "🎬",
    "Archives": "📦",
    "Logs": "📋",
    "Other": "📂",
}

MEDIA_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mp3",
    ".wav",
    ".flac",
}
ARCHIVE_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".go",
    ".sql",
    ".sh",
}
LOG_EXTENSIONS = {
    ".log",
}


class SmartSortPredictor:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.backend = "legacy"
        self.model = None
        self.tokenizer = None
        self.device = None
        self.id2label: dict[int, str] = {}
        self.max_length = 256
        self.threshold = 0.55

        if self.model_path.is_dir():
            self._load_transformer_model()
        else:
            self._load_legacy_model()

    def _load_legacy_model(self) -> None:
        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load model: {exc}") from exc

    def _load_transformer_model(self) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError(
                "Transformer dependencies are missing. "
                "Install torch/transformers in the active environment. "
                f"Original import error: {TRANSFORMERS_IMPORT_ERROR}"
            )

        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in model dir: {self.model_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        id2label_raw = metadata.get("id2label", {})
        if not id2label_raw:
            raise ValueError(f"metadata.json does not contain id2label mapping: {metadata_path}")

        self.id2label = {int(key): str(value) for key, value in id2label_raw.items()}
        self.max_length = int(metadata.get("max_length", self.max_length))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.backend = "transformer"

    @staticmethod
    def _featurize_name(filename: str) -> str:
        path = Path(filename.lower())
        stem = path.stem.replace("_", " ").replace("-", " ")
        ext = path.suffix.lower().lstrip(".")
        return f"{stem} EXT_{ext}" if ext else stem

    @staticmethod
    def _build_text_input(filename: str, content: str) -> str:
        safe_filename = SmartSortPredictor._featurize_name(filename)
        safe_content = " ".join(str(content or "").split())
        return f"[FILE] {safe_filename} [CONTENT] {safe_content}".strip()

    def _predict_transformer_text(self, filename: str, content: str) -> tuple[str, float]:
        text = self._build_text_input(filename, content)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu()
            pred_id = int(torch.argmax(probs).item())

        category = self.id2label.get(pred_id, "Other")
        confidence = float(probs[pred_id].item())
        return category, confidence

    def predict_file(self, filepath: str | Path) -> dict:
        path = Path(filepath)
        suffixes = {suffix.lower() for suffix in path.suffixes}

        if suffixes & MEDIA_EXTENSIONS or path.suffix.lower() in MEDIA_EXTENSIONS:
            return {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "category": "Media",
                "icon": CATEGORY_ICONS["Media"],
                "confidence": 0.99,
                "content_preview": "",
            }

        if suffixes & ARCHIVE_EXTENSIONS or path.suffix.lower() in ARCHIVE_EXTENSIONS:
            return {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "category": "Archives",
                "icon": CATEGORY_ICONS["Archives"],
                "confidence": 0.99,
                "content_preview": "",
            }

        if suffixes & CODE_EXTENSIONS or path.suffix.lower() in CODE_EXTENSIONS:
            return {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "category": "Code",
                "icon": CATEGORY_ICONS["Code"],
                "confidence": 0.98,
                "content_preview": "",
            }

        if suffixes & LOG_EXTENSIONS or path.suffix.lower() in LOG_EXTENSIONS:
            return {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "category": "Logs",
                "icon": CATEGORY_ICONS["Logs"],
                "confidence": 0.98,
                "content_preview": "",
            }

        try:
            content = extract_text(path)
            preview = content[:120].strip()
            if self.backend == "transformer":
                category, confidence = self._predict_transformer_text(path.name, content)
            else:
                features = pd.DataFrame(
                    [
                        {
                            "filename": self._featurize_name(path.name),
                            "content": content,
                        }
                    ]
                )

                probabilities = self.model.predict_proba(features)[0]
                classes = self.model.classes_
                best_index = int(probabilities.argmax())
                confidence = float(probabilities[best_index])
                category = str(classes[best_index])

            if confidence < self.threshold:
                category = "Other"

            return {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "category": category,
                "icon": CATEGORY_ICONS.get(category, CATEGORY_ICONS["Other"]),
                "confidence": round(confidence, 3),
                "content_preview": preview,
            }
        except Exception:
            return {
                "filepath": str(path),
                "filename": path.name,
                "category": "Other",
                "icon": CATEGORY_ICONS["Other"],
                "confidence": 0.0,
                "content_preview": "",
            }

    def predict_batch(self, filepaths: list[str | Path]) -> list[dict]:
        return [self.predict_file(filepath) for filepath in filepaths]
