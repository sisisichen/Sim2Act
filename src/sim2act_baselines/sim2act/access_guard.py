from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Iterator

import h5py


ACCESS_LOG = Path(os.environ.get("SIM2ACT_ACCESS_LOG", "sim2act_access.jsonl"))
TEST_IDS = frozenset(str(value) for value in range(1218, 1429))


class SealedEvaluationAccessError(RuntimeError):
    pass


def _sample_id(path: Path) -> str | None:
    for part in path.parts:
        if re.fullmatch(r"\d{4}", part):
            return part
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", path.name)
    return match.group(1) if match else None


def _record(path: Path, purpose: str, mode: str, decision: str) -> None:
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "path": str(path.resolve(strict=False)),
        "purpose": purpose,
        "mode": mode,
        "decision": decision,
    }
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def assert_model_access(path: str | Path, purpose: str, mode: str = "read") -> Path:
    value = Path(path).resolve(strict=False)
    lowered = str(value).lower().replace("\\", "/")
    sample_id = _sample_id(value)
    forbidden = (
        "/datasets/raw/crackensembles/test/" in lowered
        or ("/cache/passive_evidence_v1/crackensembles/" in lowered and sample_id in TEST_IDS)
        or "/cache/passive_evidence_v1/crackstructures/" in lowered
        or "/datasets/raw/crackstructures/" in lowered
        or "/trial_shards/crackstructures/" in lowered
        or "/audit_shards/crackstructures/" in lowered
    )
    decision = "DENY" if forbidden else "ALLOW"
    _record(value, purpose, mode, decision)
    if forbidden:
        raise SealedEvaluationAccessError(f"sealed final-evaluation access denied: {value}")
    return value


def guarded_read_text(path: str | Path, purpose: str) -> str:
    value = assert_model_access(path, purpose, "read_text")
    return value.read_text(encoding="utf-8")


@contextmanager
def guarded_h5(path: str | Path, purpose: str) -> Iterator[h5py.File]:
    value = assert_model_access(path, purpose, "h5_read")
    with h5py.File(value, "r") as handle:
        yield handle


def assert_scaler_source(split: str) -> None:
    if split != "development_train":
        raise SealedEvaluationAccessError(f"scaler fit is restricted to development_train, got {split}")


def assert_checkpoint_selection_source(split: str) -> None:
    if split != "development_validation":
        raise SealedEvaluationAccessError(f"checkpoint selection is restricted to development_validation, got {split}")
