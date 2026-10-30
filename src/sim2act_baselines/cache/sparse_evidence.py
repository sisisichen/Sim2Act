from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(group: h5py.Group, name: str, values: np.ndarray, *, chunks=True) -> h5py.Dataset:
    array = np.asarray(values)
    return group.create_dataset(
        name,
        data=array,
        chunks=chunks,
        compression="gzip",
        compression_opts=1,
        shuffle=True,
        fletcher32=True,
    )


@dataclass(frozen=True)
class DenseEvidence:
    surface_ids: np.ndarray
    surface_coordinates: np.ndarray
    surface_normals: np.ndarray
    surface_colors: np.ndarray
    view_ids: tuple[str, ...]
    probabilities: np.ndarray
    visibility: np.ndarray
    angles: np.ndarray
    distances: np.ndarray
    camera_centers: np.ndarray


class SparseEvidenceWriter:
    """Transactional writer for lossless per-view projected evidence."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: dict[str, Any],
        surface_coordinates: np.ndarray,
        surface_normals: np.ndarray,
        surface_colors: np.ndarray,
        camera_centers: np.ndarray,
        view_ids: Iterable[str],
    ):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        self.sidecar = self.path.with_suffix(self.path.suffix + ".json")
        if self.path.exists() or self.partial.exists() or self.sidecar.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.view_ids = tuple(str(value) for value in view_ids)
        self.handle = h5py.File(self.partial, "x", libver="latest")
        self.handle.attrs["complete"] = False
        self.handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        string_type = h5py.string_dtype(encoding="utf-8")
        self.handle.create_dataset("view_ids", data=np.asarray(self.view_ids, dtype=object), dtype=string_type)
        coordinates = np.asarray(surface_coordinates, dtype=np.float32)
        normals = np.asarray(surface_normals, dtype=np.float32)
        colors = np.asarray(surface_colors, dtype=np.float32)
        if coordinates.shape != normals.shape or coordinates.shape != colors.shape or coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("surface array shape mismatch")
        self.count = len(coordinates)
        _dataset(self.handle, "surface_ids", np.arange(self.count, dtype=np.int32))
        _dataset(self.handle, "surface_coordinates", coordinates)
        _dataset(self.handle, "surface_normals", normals)
        _dataset(self.handle, "surface_colors", colors)
        _dataset(self.handle, "camera_centers", np.asarray(camera_centers, dtype=np.float32))
        self.views_group = self.handle.create_group("views", track_order=True)
        self.records: list[dict[str, Any]] = []
        self._closed = False

    def add_view(
        self,
        view_id: str,
        *,
        visible_indices: np.ndarray,
        probabilities: np.ndarray,
        visible_angles: np.ndarray,
        visible_distances: np.ndarray,
        all_angles: np.ndarray,
        all_distances: np.ndarray,
        projected_xy: np.ndarray,
        record: dict[str, Any],
    ) -> None:
        key = str(view_id)
        if key not in self.view_ids or key in self.views_group:
            raise ValueError(key)
        indices = np.asarray(visible_indices, dtype=np.int32).reshape(-1)
        if len(indices) and (indices.min() < 0 or indices.max() >= self.count):
            raise IndexError("visible surface index outside cache")
        if len(indices) and np.any(indices[1:] <= indices[:-1]):
            raise ValueError("visible indices must be strictly increasing")
        values = np.asarray(probabilities, dtype=np.float32)
        angles = np.asarray(visible_angles, dtype=np.float32).reshape(-1)
        distances = np.asarray(visible_distances, dtype=np.float32).reshape(-1)
        xy = np.asarray(projected_xy, dtype=np.int32).reshape(-1, 2)
        if values.shape != (len(indices), 2) or len(angles) != len(indices) or len(distances) != len(indices) or len(xy) != len(indices):
            raise ValueError("sparse view array shape mismatch")
        all_a = np.asarray(all_angles, dtype=np.float32).reshape(-1)
        all_d = np.asarray(all_distances, dtype=np.float32).reshape(-1)
        if len(all_a) != self.count or len(all_d) != self.count:
            raise ValueError("dense condition length mismatch")
        if not all(np.isfinite(array).all() for array in (values, angles, distances, all_a, all_d)):
            raise ValueError("non-finite sparse evidence")
        group = self.views_group.create_group(key)
        _dataset(group, "surface_indices", indices)
        _dataset(group, "probabilities", values)
        _dataset(group, "angles", angles)
        _dataset(group, "distances", distances)
        _dataset(group, "all_surface_angles", all_a)
        _dataset(group, "all_surface_distances", all_d)
        _dataset(group, "projected_xy", xy)
        _dataset(group, "valid_projection", np.ones(len(indices), dtype=np.uint8))
        group.attrs["record_json"] = json.dumps(record, sort_keys=True)
        self.records.append({"view_id": key, **record, "visible_surface_count": len(indices)})
        self.handle.flush()

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("writer already closed")
        missing = [value for value in self.view_ids if value not in self.views_group]
        if missing:
            raise RuntimeError(f"incomplete view set: {missing}")
        self.handle.attrs.modify("complete", True)
        self.handle.flush()
        self.handle.close()
        self._closed = True
        self.partial.replace(self.path)
        digest = sha256(self.path)
        payload = {
            "path": str(self.path),
            "sha256": digest,
            "bytes": self.path.stat().st_size,
            "view_count": len(self.view_ids),
            "surface_count": self.count,
            "probability_dtype": "float32",
            "angle_dtype": "float32",
            "distance_dtype": "float32",
            "compression": "gzip_level_1_shuffle_fletcher32_lossless",
            "records": self.records,
            "complete": True,
        }
        with self.sidecar.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return payload

    def abort(self) -> None:
        if not self._closed:
            self.handle.close()
            self._closed = True


class SparseEvidenceCache:
    def __init__(self, path: Path, *, validate_checksum: bool = True):
        self.path = Path(path)
        self.sidecar_path = self.path.with_suffix(self.path.suffix + ".json")
        self.sidecar = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        if validate_checksum and sha256(self.path) != self.sidecar["sha256"]:
            raise ValueError(f"sparse evidence checksum mismatch: {self.path}")
        with h5py.File(self.path, "r") as handle:
            if not bool(handle.attrs["complete"]):
                raise ValueError("incomplete sparse cache")
            self.metadata = json.loads(handle.attrs["metadata_json"])
            self.view_ids = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in handle["view_ids"][:])
            self.surface_count = len(handle["surface_ids"])

    def validate(self) -> None:
        with h5py.File(self.path, "r") as handle:
            ids = handle["surface_ids"][:]
            if ids.dtype != np.int32 or not np.array_equal(ids, np.arange(len(ids), dtype=np.int32)):
                raise ValueError("surface IDs are not stable contiguous int32")
            for name in ("surface_coordinates", "surface_normals", "surface_colors", "camera_centers"):
                values = handle[name]
                if values.dtype != np.float32 or not np.isfinite(values[:]).all():
                    raise ValueError(f"invalid {name}")
            if tuple(handle["views"].keys()) != self.view_ids:
                raise ValueError("view order mismatch")
            for view_id in self.view_ids:
                group = handle["views"][view_id]
                if group["surface_indices"].dtype != np.int32:
                    raise TypeError("surface indices must be int32")
                for name in ("probabilities", "angles", "distances", "all_surface_angles", "all_surface_distances"):
                    if group[name].dtype != np.float32:
                        raise TypeError(f"{name} must be float32")
                    if not np.isfinite(group[name][:]).all():
                        raise ValueError(f"{view_id}/{name} is non-finite")

    def dense_evidence(self) -> DenseEvidence:
        with h5py.File(self.path, "r") as handle:
            count = len(handle["surface_ids"])
            views = len(self.view_ids)
            probabilities = np.full((count, views, 2), np.nan, dtype=np.float32)
            visibility = np.zeros((count, views), dtype=bool)
            angles = np.empty((count, views), dtype=np.float32)
            distances = np.empty((count, views), dtype=np.float32)
            for view_index, view_id in enumerate(self.view_ids):
                group = handle["views"][view_id]
                indices = group["surface_indices"][:]
                probabilities[indices, view_index] = group["probabilities"][:]
                visibility[indices, view_index] = True
                angles[:, view_index] = group["all_surface_angles"][:]
                distances[:, view_index] = group["all_surface_distances"][:]
            return DenseEvidence(
                surface_ids=handle["surface_ids"][:],
                surface_coordinates=handle["surface_coordinates"][:],
                surface_normals=handle["surface_normals"][:],
                surface_colors=handle["surface_colors"][:],
                view_ids=self.view_ids,
                probabilities=probabilities,
                visibility=visibility,
                angles=angles,
                distances=distances,
                camera_centers=handle["camera_centers"][:],
            )

    def visible_count(self) -> int:
        with h5py.File(self.path, "r") as handle:
            return sum(len(handle["views"][view_id]["surface_indices"]) for view_id in self.view_ids)

