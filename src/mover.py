from __future__ import annotations

import shutil
from pathlib import Path

from src.storage import Storage


class FileMover:
    def __init__(self, base_dst_dir: str | Path, storage: Storage):
        self.base_dst_dir = Path(base_dst_dir)
        self.storage = storage

    def _build_destination(self, src_path: Path, category: str) -> Path:
        target_dir = self.base_dst_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)

        candidate = target_dir / src_path.name
        if not candidate.exists():
            return candidate

        stem = src_path.stem
        suffix = src_path.suffix
        counter = 1
        while True:
            candidate = target_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def move_file(self, src_path: str | Path, category: str, confidence: float = 0.0) -> str:
        source = Path(src_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        destination = self._build_destination(source, category)

        try:
            shutil.move(str(source), str(destination))
            self.storage.log_move(str(source), str(destination), category, confidence)
            return str(destination)
        except Exception as exc:
            raise RuntimeError(f"Failed to move file '{source}' to '{destination}': {exc}") from exc

    def undo_last(self) -> bool:
        record = self.storage.undo_last()
        if record is None:
            return False

        source = Path(record["src"])
        destination = Path(record["dst"])
        source.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not destination.exists():
                raise FileNotFoundError(f"Moved file not found: {destination}")

            shutil.move(str(destination), str(source))
            return True
        except Exception:
            self.storage.set_undone(record["id"], False)
            return False
