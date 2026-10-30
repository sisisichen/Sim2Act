from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


def git_snapshot(path: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=path, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        ).stdout
    diff = git("diff", "--binary", "HEAD")
    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD").strip(),
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "tracked_clean": diff == "",
    }

