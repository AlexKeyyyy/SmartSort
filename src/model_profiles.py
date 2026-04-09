from __future__ import annotations

import json
from pathlib import Path


MODEL_PRIORITY = [
    "transformer_rubert",
    "transformer_distilbert",
    "legacy_mlp",
]


def build_model_profiles(base_dir: Path) -> dict[str, dict]:
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports" / "model_eval"

    return {
        "legacy_mlp": {
            "title": "Legacy (TF-IDF + MLP)",
            "description": "Classical baseline for fast CPU inference.",
            "backend": "legacy",
            "model_path": models_dir / "smartsort_model.pkl",
            "report_dir": reports_dir / "latest",
            "model_name": "legacy_tfidf_mlp",
        },
        "transformer_distilbert": {
            "title": "Transformer DistilBERT (multilingual)",
            "description": "Fast multilingual transformer baseline (RU + EN).",
            "backend": "transformer",
            "model_name": "distilbert/distilbert-base-multilingual-cased",
            "model_path": models_dir / "smartsort_transformer_distilbert",
            "fallback_path": models_dir / "smartsort_transformer",
            "fallback_model_name_contains": "distilbert-base-multilingual-cased",
            "report_dir": reports_dir / "transformer_distilbert",
        },
        "transformer_rubert": {
            "title": "Transformer ruBERT (DeepPavlov)",
            "description": "Russian-focused transformer for maximum RU quality.",
            "backend": "transformer",
            "model_name": "DeepPavlov/rubert-base-cased",
            "model_path": models_dir / "smartsort_transformer_rubert",
            "fallback_path": models_dir / "smartsort_transformer",
            "fallback_model_name_contains": "rubert-base-cased",
            "report_dir": reports_dir / "transformer_rubert",
        },
    }


def read_transformer_model_name(model_dir: Path) -> str | None:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return str(metadata.get("model_name", "")).strip() or None


def resolve_profile_model_path(profile: dict) -> Path | None:
    direct_path = Path(profile["model_path"])
    if direct_path.exists():
        return direct_path

    fallback_path_raw = profile.get("fallback_path")
    expected_fragment = str(profile.get("fallback_model_name_contains", "")).lower()
    if not fallback_path_raw or not expected_fragment:
        return None

    fallback_path = Path(fallback_path_raw)
    if not fallback_path.exists() or not fallback_path.is_dir():
        return None

    model_name = read_transformer_model_name(fallback_path)
    if model_name and expected_fragment in model_name.lower():
        return fallback_path
    return None


def is_profile_available(profile: dict) -> bool:
    return resolve_profile_model_path(profile) is not None


def first_available_model_key(profiles: dict[str, dict], preferred_key: str | None = None) -> str | None:
    if preferred_key in profiles and is_profile_available(profiles[preferred_key]):
        return preferred_key

    for key in MODEL_PRIORITY:
        if key in profiles and is_profile_available(profiles[key]):
            return key
    return None
