from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


STATUSES = {
    "PENDING",
    "RUNNING",
    "COMPLETE",
    "VALID_EMPTY",
    "FAILED_DATA",
    "FAILED_RUNTIME",
    "FAILED_RESOURCE",
    "SKIPPED_NEGATIVE_CONTROL",
    "RETRY_REQUIRED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunManifest:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def claim(self, dataset: str, scene_id: str, method: str, stage: str, protocol_hash: str) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM work_units WHERE dataset=? AND scene_id=? AND method=? AND stage=? AND protocol_hash=?",
                (dataset, scene_id, method, stage, protocol_hash),
            ).fetchone()
            if row is None:
                raise KeyError((dataset, scene_id, method, stage, protocol_hash))
            if row["status"] not in {"PENDING", "RETRY_REQUIRED"}:
                connection.rollback()
                return False
            old = row["status"]
            connection.execute(
                "UPDATE work_units SET status='RUNNING',attempt=attempt+1,started_utc=?,completed_utc=NULL,error=NULL WHERE dataset=? AND scene_id=? AND method=? AND stage=? AND protocol_hash=?",
                (utc_now(), dataset, scene_id, method, stage, protocol_hash),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,method,stage,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, method, stage, old, "RUNNING", "atomic claim"),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def finish(
        self,
        dataset: str,
        scene_id: str,
        method: str,
        stage: str,
        protocol_hash: str,
        status: str,
        *,
        artifact_path: str | None = None,
        artifact_sha256: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if status not in STATUSES or status == "RUNNING":
            raise ValueError(status)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM work_units WHERE dataset=? AND scene_id=? AND method=? AND stage=? AND protocol_hash=?",
                (dataset, scene_id, method, stage, protocol_hash),
            ).fetchone()
            if row is None:
                raise KeyError((dataset, scene_id, method, stage, protocol_hash))
            old = row["status"]
            if old != "RUNNING" and not (old == "RETRY_REQUIRED" and status == "RETRY_REQUIRED"):
                raise RuntimeError(f"invalid manifest transition {old}->{status}")
            connection.execute(
                "UPDATE work_units SET status=?,completed_utc=?,artifact_path=?,artifact_sha256=?,error=?,metadata_json=? WHERE dataset=? AND scene_id=? AND method=? AND stage=? AND protocol_hash=?",
                (
                    status,
                    utc_now(),
                    artifact_path,
                    artifact_sha256,
                    error,
                    json.dumps(metadata, sort_keys=True) if metadata is not None else None,
                    dataset,
                    scene_id,
                    method,
                    stage,
                    protocol_hash,
                ),
            )
            connection.execute(
                "INSERT INTO events(created_utc,dataset,scene_id,method,stage,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                (utc_now(), dataset, scene_id, method, stage, old, status, error or artifact_path),
            )
            connection.commit()
        finally:
            connection.close()

    def reset_running(self, detail: str = "resume after interruption") -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT dataset,scene_id,method,stage,status FROM work_units WHERE status='RUNNING'").fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE dataset=? AND scene_id=? AND method=? AND stage=?",
                    (detail, row["dataset"], row["scene_id"], row["method"], row["stage"]),
                )
                connection.execute(
                    "INSERT INTO events(created_utc,dataset,scene_id,method,stage,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                    (utc_now(), row["dataset"], row["scene_id"], row["method"], row["stage"], "RUNNING", "RETRY_REQUIRED", detail),
                )
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    def rows(self, *, dataset: str | None = None, stage: str | None = None, statuses: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        clauses, values = [], []
        if dataset:
            clauses.append("dataset=?"); values.append(dataset)
        if stage:
            clauses.append("stage=?"); values.append(stage)
        if statuses:
            clauses.append("status IN (" + ",".join("?" for _ in statuses) + ")"); values.extend(statuses)
        query = "SELECT * FROM work_units" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY dataset,scene_id,stage,method"
        connection = self._connect()
        try:
            return [dict(row) for row in connection.execute(query, values).fetchall()]
        finally:
            connection.close()

    def export_csv(self, path: Path) -> None:
        rows = self.rows()
        if not rows:
            raise ValueError("empty run manifest")
        temporary = Path(str(path) + f".tmp.{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
            handle.flush()
        temporary.replace(path)

    def retry_failed(self, *, dataset: str, scene_ids: tuple[str, ...], stage: str, detail: str) -> int:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            changed = 0
            for scene_id in scene_ids:
                rows = connection.execute(
                    "SELECT method,status FROM work_units WHERE dataset=? AND scene_id=? AND stage=? AND status IN ('FAILED_DATA','FAILED_RUNTIME','FAILED_RESOURCE')",
                    (dataset, scene_id, stage),
                ).fetchall()
                for row in rows:
                    connection.execute(
                        "UPDATE work_units SET status='RETRY_REQUIRED',error=? WHERE dataset=? AND scene_id=? AND method=? AND stage=?",
                        (detail, dataset, scene_id, row["method"], stage),
                    )
                    connection.execute(
                        "INSERT INTO events(created_utc,dataset,scene_id,method,stage,old_status,new_status,detail) VALUES(?,?,?,?,?,?,?,?)",
                        (utc_now(), dataset, scene_id, row["method"], stage, row["status"], "RETRY_REQUIRED", detail),
                    )
                    changed += 1
            connection.commit()
            return changed
        finally:
            connection.close()
