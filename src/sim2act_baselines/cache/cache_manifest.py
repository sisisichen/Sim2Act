from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix() if root else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

