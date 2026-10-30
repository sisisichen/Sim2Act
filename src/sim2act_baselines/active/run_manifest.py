from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActiveRunManifest:
    """Transactional, resumable work-unit ledger for active smoke trials."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize(self, work_units: list[tuple[str, str, int, str]], protocol_hash: str) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS work_units(
                    dataset TEXT NOT NULL, scene_id TEXT NOT NULL, seed INTEGER NOT NULL,
                    selector TEXT NOT NULL, protocol_hash TEXT NOT NULL, status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0, started_utc TEXT, completed_utc TEXT,
                    artifact_path TEXT, artifact_sha256 TEXT, error TEXT, metadata_json TEXT,
                    PRIMARY KEY(dataset,scene_id,seed,selector,protocol_hash));
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
                    dataset TEXT, scene_id TEXT, seed INTEGER, selector TEXT,
                    old_status TEXT, new_status TEXT, detail TEXT);
            """)
            connection.executemany(
                "INSERT OR IGNORE INTO work_units(dataset,scene_id,seed,selector,protocol_hash,status) VALUES(?,?,?,?,?,'PENDING')",
                [(*unit, protocol_hash) for unit in work_units],
            )

    def claim(self, dataset: str, scene_id: str, seed: int, selector: str, protocol_hash: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            key = (dataset, scene_id, seed, selector, protocol_hash)
            row = connection.execute(
                "SELECT status FROM work_units WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?", key
            ).fetchone()
            if row is None:
                raise KeyError(key)
            if row["status"] not in {"PENDING", "RETRY_REQUIRED"}:
                connection.rollback(); return False
            old = row["status"]
            connection.execute(
                "UPDATE work_units SET status='RUNNING',attempt=attempt+1,started_utc=?,completed_utc=NULL,error=NULL WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                (utc_now(), *key),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, seed, selector, old, "RUNNING", "atomic claim"),
            )
            connection.commit(); return True
        finally:
            connection.close()

    def finish(self, dataset: str, scene_id: str, seed: int, selector: str, protocol_hash: str, *, artifact_path: str, artifact_sha256: str, metadata: dict | None = None) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            key = (dataset, scene_id, seed, selector, protocol_hash)
            row = connection.execute(
                "SELECT status FROM work_units WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?", key
            ).fetchone()
            if row is None or row["status"] != "RUNNING":
                raise RuntimeError(f"invalid manifest completion: {key} {None if row is None else row['status']}")
            connection.execute(
                "UPDATE work_units SET status='COMPLETE',completed_utc=?,artifact_path=?,artifact_sha256=?,metadata_json=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                (utc_now(), artifact_path, artifact_sha256, json.dumps(metadata or {}, sort_keys=True), *key),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, seed, selector, "RUNNING", "COMPLETE", artifact_path),
            )
            connection.commit()
        finally:
            connection.close()

    def fail(self, dataset: str, scene_id: str, seed: int, selector: str, protocol_hash: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE work_units SET status='FAILED',completed_utc=?,error=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                (utc_now(), error, dataset, scene_id, seed, selector, protocol_hash),
            )

    def reset_running(self, detail: str = "resume after interruption") -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT dataset,scene_id,seed,selector FROM work_units WHERE status='RUNNING'").fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=?",
                    (detail, row["dataset"], row["scene_id"], row["seed"], row["selector"]),
                )
                connection.execute(
                    "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                    (utc_now(), row["dataset"], row["scene_id"], row["seed"], row["selector"], "RUNNING", "RETRY_REQUIRED", detail),
                )
            connection.commit(); return len(rows)
        finally:
            connection.close()

    def rows(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM work_units ORDER BY dataset,scene_id,seed,selector")]

    def retry_failed(self, detail: str = "explicit retry") -> int:
        with self._connect() as connection:
            rows = connection.execute("SELECT COUNT(*) FROM work_units WHERE status='FAILED'").fetchone()[0]
            connection.execute("UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE status='FAILED'", (detail,))
            return int(rows)
