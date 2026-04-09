from __future__ import annotations

from pathlib import Path
from typing import Callable

from watchdog.events import FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer


class _SmartSortEventHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[dict], None], predictor):
        self.callback = callback
        self.predictor = predictor

    @staticmethod
    def _should_ignore(path: str | Path) -> bool:
        name = Path(path).name
        return not name or name.startswith(".") or name.startswith("~")

    def _process_path(self, path: str | Path) -> None:
        if self._should_ignore(path):
            return

        candidate = Path(path)
        if not candidate.exists() or not candidate.is_file():
            return

        try:
            result = self.predictor.predict_file(candidate)
            self.callback(result)
        except Exception:
            return

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._process_path(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if event.is_directory:
            return
        self._process_path(event.dest_path)


class FolderWatcher:
    def __init__(self, watch_dir: str | Path, callback: Callable[[dict], None], predictor):
        self.watch_dir = Path(watch_dir)
        self.callback = callback
        self.predictor = predictor
        self.observer: Observer | None = None
        self.event_handler = _SmartSortEventHandler(callback, predictor)

    def start(self) -> None:
        if self.is_running():
            return

        if not self.watch_dir.exists() or not self.watch_dir.is_dir():
            raise FileNotFoundError(f"Watch directory not found: {self.watch_dir}")

        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(self.watch_dir), recursive=False)
        self.observer.start()

    def stop(self) -> None:
        if not self.observer:
            return

        self.observer.stop()
        self.observer.join(timeout=5)
        self.observer = None

    def is_running(self) -> bool:
        return self.observer is not None and self.observer.is_alive()
