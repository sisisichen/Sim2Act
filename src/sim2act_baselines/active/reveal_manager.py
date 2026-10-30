from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .base import CandidateViewGeometry, readonly


@dataclass(frozen=True)
class RevealedViewEvidence:
    view_id: str
    visible_surface_indices: np.ndarray
    crack_probabilities: np.ndarray
    angles: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        indices = readonly(self.visible_surface_indices, np.int32).reshape(-1)
        probabilities = readonly(self.crack_probabilities, np.float32)
        angles = readonly(self.angles, np.float32).reshape(-1)
        distances = readonly(self.distances, np.float32).reshape(-1)
        if probabilities.shape != (len(indices), 2) or len(angles) != len(indices) or len(distances) != len(indices):
            raise ValueError("revealed evidence shape mismatch")
        object.__setattr__(self, "visible_surface_indices", indices)
        object.__setattr__(self, "crack_probabilities", probabilities)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "distances", distances)


class RevealManager:
    """Read-only evidence gate. Probabilities are read only after an explicit reveal."""

    def __init__(self, cache_path: Path):
        import h5py

        self.cache_path = Path(cache_path)
        self._revealed: set[str] = set()
        self._evidence_cache: dict[str, RevealedViewEvidence] = {}
        with h5py.File(self.cache_path, "r") as handle:
            self.view_ids = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in handle["view_ids"][:])
            self.surface_count = len(handle["surface_ids"])
            metadata = json.loads(handle.attrs["metadata_json"])
            self.surface_area = float(metadata["surface_area_m2"])
            camera_centers = np.asarray(handle["camera_centers"][:], dtype=np.float32)
            camera_records = {}
            cameras_path = Path(metadata.get("cameras_path", ""))
            if cameras_path.is_file():
                camera_records = json.loads(cameras_path.read_text())
            candidates = []
            for stable_index, view_id in enumerate(self.view_ids):
                indices = np.asarray(handle["views"][view_id]["surface_indices"][:], dtype=np.int32)
                camera_pose = np.eye(4, dtype=np.float32)
                record = camera_records.get(view_id, {})
                if "R" in record:
                    rotation = np.asarray(record["R"], dtype=np.float32).reshape(3, 3)
                    camera_pose[:3, :3] = rotation.T
                camera_pose[:3, 3] = camera_centers[stable_index]
                candidates.append(CandidateViewGeometry(
                    view_id=view_id,
                    stable_index=stable_index,
                    camera_pose=camera_pose,
                    visible_surface_indices=indices,
                    visible_surface_area=float(len(indices) * self.surface_area / self.surface_count),
                ))
        self._candidates = tuple(candidates)

    @property
    def candidates(self) -> tuple[CandidateViewGeometry, ...]:
        return self._candidates

    @property
    def revealed_view_ids(self) -> tuple[str, ...]:
        return tuple(view_id for view_id in self.view_ids if view_id in self._revealed)

    def _read_evidence(self, view_id: str) -> RevealedViewEvidence:
        import h5py

        if view_id not in self.view_ids:
            raise KeyError(view_id)
        cached = self._evidence_cache.get(view_id)
        if cached is not None:
            return cached
        with h5py.File(self.cache_path, "r") as handle:
            group = handle["views"][view_id]
            evidence = RevealedViewEvidence(
                view_id=view_id,
                visible_surface_indices=group["surface_indices"][:],
                crack_probabilities=group["probabilities"][:],
                angles=group["angles"][:],
                distances=group["distances"][:],
            )
        self._evidence_cache[view_id] = evidence
        return evidence

    def reveal(self, view_id: str) -> RevealedViewEvidence:
        if view_id in self._revealed:
            raise ValueError(f"view already revealed: {view_id}")
        evidence = self._read_evidence(view_id)
        self._revealed.add(view_id)
        return evidence

    def oracle_loader(self) -> Callable[[str], RevealedViewEvidence]:
        """Capability handed only to OracleContext by the active runner."""
        return self._read_evidence

    def fork(self) -> "RevealManager":
        """Create an independent reveal session without rereading candidate geometry."""
        result = object.__new__(RevealManager)
        result.cache_path = self.cache_path
        result._revealed = set()
        result.view_ids = self.view_ids
        result.surface_count = self.surface_count
        result.surface_area = self.surface_area
        result._candidates = self._candidates
        result._evidence_cache = self._evidence_cache
        return result
