from __future__ import annotations

import csv
from pathlib import Path
import threading
import time

import psutil


class ResourceMonitor:
    def __init__(self, log_path: Path | None = None, interval_seconds: float = 0.1):
        self.log_path = Path(log_path) if log_path is not None else None
        self.interval_seconds = float(interval_seconds)
        self.peak_rss_bytes = 0
        self.samples: list[tuple[float, int]] = []
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ResourceMonitor":
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _sample_loop(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            rss = int(process.memory_info().rss)
            elapsed = time.perf_counter() - self._started
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.samples.append((elapsed, rss))
            self._stop.wait(self.interval_seconds)

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["elapsed_seconds", "rss_bytes"])
                writer.writerows(self.samples)
