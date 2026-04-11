from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "smartsort_model.pkl"
RANDOM_SEED = 42


def featurize_name(filename: str) -> str:
    path = Path(str(filename).lower())
    stem = path.stem.replace("_", " ").replace("-", " ")
    ext = path.suffix.lower().lstrip(".")
    return f"{stem} EXT_{ext}" if ext else stem


def build_pipeline() -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "filename_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    max_features=5000,
                    lowercase=True,
                ),
                "filename",
            ),
            (
                "content_word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=10000,
                    lowercase=True,
                ),
                "content",
            ),
            (
                "content_char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=8000,
                    lowercase=True,
                ),
                "content",
            ),
        ]
    )

    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        # NOTE:
        # In some sklearn/numpy combos with string targets, early_stopping may crash
        # inside internal validation scoring (np.isnan on non-numeric class outputs).
        # Keep training deterministic and stable for course delivery.
        early_stopping=False,
        max_iter=250,
        random_state=RANDOM_SEED,
        verbose=False,
    )

    return Pipeline(
        [
            ("features", features),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH).fillna("")
    required_columns = {"filename", "content", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    prepared = df.copy()
    prepared["filename"] = prepared["filename"].apply(featurize_name)

    X = prepared[["filename", "content"]]
    y = prepared["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = build_pipeline()
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"\n[OK] Accuracy: {accuracy:.2%}\n")
    print(classification_report(y_val, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[OK] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        raise
