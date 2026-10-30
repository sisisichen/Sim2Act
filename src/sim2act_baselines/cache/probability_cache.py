from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ProbabilityCache:
    def __init__(self, bundle: Path):
        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "probability_manifest.json").read_text(encoding="utf-8"))
        self.view_ids = tuple(self.manifest["view_ids"])
        self.point_probabilities = np.load(self.bundle / "point_probabilities.npy", mmap_mode="r")

    def image_probability(self, view_id: str, mmap_mode: str | None = "r") -> np.ndarray:
        if view_id not in self.view_ids:
            raise KeyError(view_id)
        return np.load(self.bundle / "image_probabilities" / f"{view_id}.npy", mmap_mode=mmap_mode)

    def validate(self) -> None:
        if self.point_probabilities.dtype != np.float16:
            raise TypeError("point probabilities must reproduce upstream float16 Mapper storage")
        if self.point_probabilities.ndim != 3 or self.point_probabilities.shape[2] != 2:
            raise ValueError(self.point_probabilities.shape)
        if self.point_probabilities.shape[1] != len(self.view_ids):
            raise ValueError("point probability/view alignment mismatch")
        finite_or_nan = np.isfinite(self.point_probabilities) | np.isnan(self.point_probabilities)
        if not bool(np.all(finite_or_nan)):
            raise ValueError("point cache contains infinity")

