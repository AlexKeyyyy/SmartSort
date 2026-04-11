from __future__ import annotations

import sqlite3
from pathlib import Path


class Storage:
    def __init__(self, db_path: str | Path = "smartsort.db"):
        self.db_path = Path(db_path)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS moves (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        src TEXT NOT NULL,
                        dst TEXT NOT NULL,
                        category TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        undone INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                connection.commit()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize database: {exc}") from exc

    def log_move(self, src: str, dst: str, category: str, confidence: float) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO moves (src, dst, category, confidence, undone)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (src, dst, category, float(confidence)),
                )
                connection.commit()
        except Exception as exc:
            raise RuntimeError(f"Failed to log move: {exc}") from exc

    def set_undone(self, move_id: int, undone: bool) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE moves
                    SET undone = ?
                    WHERE id = ?
                    """,
                    (1 if undone else 0, int(move_id)),
                )
                connection.commit()
        except Exception as exc:
            raise RuntimeError(f"Failed to update undo state: {exc}") from exc

    def get_recent(self, limit: int = 50) -> list[dict]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id, src, dst, category, confidence, ts, undone
                    FROM moves
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch recent moves: {exc}") from exc

    def undo_last(self) -> dict | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT id, src, dst, category, confidence, ts, undone
                    FROM moves
                    WHERE undone = 0
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()

                if row is None:
                    return None

                connection.execute(
                    """
                    UPDATE moves
                    SET undone = 1
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
                connection.commit()
            return dict(row)
        except Exception as exc:
            raise RuntimeError(f"Failed to undo last move: {exc}") from exc

    def get_stats(self) -> dict:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT category, COUNT(*) AS count
                    FROM moves
                    WHERE undone = 0
                    GROUP BY category
                    """
                ).fetchall()
            return {row["category"]: row["count"] for row in rows}
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch stats: {exc}") from exc
