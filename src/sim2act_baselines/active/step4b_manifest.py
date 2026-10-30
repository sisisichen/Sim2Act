from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterable


VALID_STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "FAILED_RUNTIME",
    "FAILED_RESOURCE",
    "FAILED_DATA",
    "FAILED_LEAKAGE",
    "RETRY_REQUIRED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Step4BRunManifest:
    """Transactional ledger for the frozen STEP 4B trial key."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=120)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=120000")
        return connection

    def initialize(self, work_units: Iterable[tuple[str, str, int, str]], protocol_hash: str) -> None:
        units = list(work_units)
        if len(units) != len(set(units)):
            raise ValueError("duplicate work-unit key before manifest initialization")
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS work_units(
                    dataset TEXT NOT NULL,
                    scene_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    selector TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    started_utc TEXT,
                    completed_utc TEXT,
                    artifact_path TEXT,
                    artifact_sha256 TEXT,
                    audit_path TEXT,
                    audit_sha256 TEXT,
                    error TEXT,
                    metadata_json TEXT,
                    PRIMARY KEY(dataset, scene_id, seed, selector, protocol_hash)
                );
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    dataset TEXT,
                    scene_id TEXT,
                    seed INTEGER,
                    selector TEXT,
                    old_status TEXT,
                    new_status TEXT,
                    worker_id TEXT,
                    detail TEXT
                );
                """
            )
            connection.executemany(
                "INSERT OR IGNORE INTO work_units(dataset,scene_id,seed,selector,protocol_hash,status) VALUES(?,?,?,?,?,'PENDING')",
                [(*unit, protocol_hash) for unit in units],
            )
            actual = connection.execute(
                "SELECT COUNT(*) FROM work_units WHERE protocol_hash=?", (protocol_hash,)
            ).fetchone()[0]
            if actual != len(units):
                raise RuntimeError(f"manifest cardinality mismatch: expected {len(units)}, found {actual}")

    def claim(self, dataset: str, scene_id: str, seed: int, selector: str, protocol_hash: str, worker_id: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            key = (dataset, scene_id, seed, selector, protocol_hash)
            row = connection.execute(
                "SELECT status FROM work_units WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                key,
            ).fetchone()
            if row is None:
                raise KeyError(key)
            if row["status"] not in {"PENDING", "RETRY_REQUIRED"}:
                connection.rollback()
                return False
            old_status = row["status"]
            connection.execute(
                "UPDATE work_units SET status='RUNNING',attempt=attempt+1,worker_id=?,started_utc=?,completed_utc=NULL,error=NULL WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                (worker_id, utc_now(), *key),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,worker_id,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, seed, selector, old_status, "RUNNING", worker_id, "atomic claim"),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def finish(
        self,
        dataset: str,
        scene_id: str,
        seed: int,
        selector: str,
        protocol_hash: str,
        *,
        worker_id: str,
        artifact_path: str,
        artifact_sha256: str,
        audit_path: str,
        audit_sha256: str,
        metadata: dict,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            key = (dataset, scene_id, seed, selector, protocol_hash)
            row = connection.execute(
                "SELECT status,worker_id FROM work_units WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                key,
            ).fetchone()
            if row is None or row["status"] != "RUNNING" or row["worker_id"] != worker_id:
                raise RuntimeError(f"invalid completion transition: {key} {None if row is None else dict(row)}")
            connection.execute(
                "UPDATE work_units SET status='COMPLETE',completed_utc=?,artifact_path=?,artifact_sha256=?,audit_path=?,audit_sha256=?,metadata_json=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                (
                    utc_now(),
                    artifact_path,
                    artifact_sha256,
                    audit_path,
                    audit_sha256,
                    json.dumps(metadata, sort_keys=True),
                    *key,
                ),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,worker_id,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, seed, selector, "RUNNING", "COMPLETE", worker_id, artifact_path),
            )
            connection.commit()
        finally:
            connection.close()

    def fail(self, dataset: str, scene_id: str, seed: int, selector: str, protocol_hash: str, *, worker_id: str, status: str, error: str) -> None:
        if status not in VALID_STATUSES or not status.startswith("FAILED_"):
            raise ValueError(status)
        with self._connect() as connection:
            key = (dataset, scene_id, seed, selector, protocol_hash)
            connection.execute(
                "UPDATE work_units SET status=?,completed_utc=?,error=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=? AND status='RUNNING' AND worker_id=?",
                (status, utc_now(), error, *key, worker_id),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,worker_id,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, seed, selector, "RUNNING", status, worker_id, error[:4000]),
            )

    def reset_stale_running(self, active_worker_ids: set[str] | None = None, detail: str = "documented resume after interrupted worker") -> int:
        active = active_worker_ids or set()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT dataset,scene_id,seed,selector,protocol_hash,worker_id FROM work_units WHERE status='RUNNING'").fetchall()
            stale = [row for row in rows if row["worker_id"] not in active]
            for row in stale:
                connection.execute(
                    "UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE dataset=? AND scene_id=? AND seed=? AND selector=? AND protocol_hash=?",
                    (detail, row["dataset"], row["scene_id"], row["seed"], row["selector"], row["protocol_hash"]),
                )
                connection.execute(
                    "INSERT INTO events(created_utc,dataset,scene_id,seed,selector,old_status,new_status,worker_id,detail) VALUES(?,?,?,?,?,?,?,?,?)",
                    (utc_now(), row["dataset"], row["scene_id"], row["seed"], row["selector"], "RUNNING", "RETRY_REQUIRED", row["worker_id"], detail),
                )
            connection.commit()
            return len(stale)
        finally:
            connection.close()

    def retry_failures(self, statuses: set[str], detail: str) -> int:
        if not statuses or any(status not in VALID_STATUSES or not status.startswith("FAILED_") for status in statuses):
            raise ValueError(statuses)
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as connection:
            count = connection.execute(
                f"SELECT COUNT(*) FROM work_units WHERE status IN ({placeholders})", tuple(sorted(statuses))
            ).fetchone()[0]
            connection.execute(
                f"UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE status IN ({placeholders})",
                (detail, *sorted(statuses)),
            )
            return int(count)

    def rows(self) -> list[dict]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute("SELECT * FROM work_units ORDER BY dataset,scene_id,seed,selector")]

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {str(row[0]): int(row[1]) for row in connection.execute("SELECT status,COUNT(*) FROM work_units GROUP BY status")}
