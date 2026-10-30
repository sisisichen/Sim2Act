from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GeometryCache:
    ARRAY_NAMES = (
        "point_ids",
        "points",
        "normals",
        "colors",
        "camera_centers",
        "camera_rotations",
        "camera_translations",
        "projected_xy",
        "valid_image_mask",
        "visibility_mask",
        "view_angles",
        "view_distances",
    )

    def __init__(self, bundle: Path):
        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "geometry_manifest.json").read_text(encoding="utf-8"))
        self.view_ids = tuple(self.manifest["view_ids"])
        for name in self.ARRAY_NAMES:
            setattr(self, name, np.load(self.bundle / f"{name}.npy", mmap_mode="r"))

    def validate(self) -> None:
        count = len(self.point_ids)
        views = len(self.view_ids)
        expected = int(self.manifest.get("num_points", count))
        if count != expected:
            raise ValueError(f"expected {expected} points, got {count}")
        if not np.array_equal(self.point_ids, np.arange(count, dtype=self.point_ids.dtype)):
            raise ValueError("point IDs are not stable and contiguous")
        for name in ("points", "normals", "colors", "camera_centers", "camera_rotations", "camera_translations", "view_angles", "view_distances"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains NaN/Inf")
        if self.visibility_mask.shape != (count, views):
            raise ValueError("visibility shape mismatch")
        if self.valid_image_mask.shape != (count, views):
            raise ValueError("valid-image shape mismatch")
        if self.projected_xy.shape != (count, views, 2):
            raise ValueError("projection shape mismatch")
        if np.any(self.visibility_mask.astype(bool) & ~self.valid_image_mask.astype(bool)):
            raise ValueError("visibility cannot exceed image validity")
