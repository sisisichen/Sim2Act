from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim2act_baselines.active.reveal_manager import RevealManager

from .access_guard import guarded_h5
from .feature_schema import CANDIDATE_FEATURES, GLOBAL_FEATURES, validate_schema
from .state_sampler import CausalState


@dataclass(frozen=True)
class GeometryFeatureCache:
    view_ids: tuple[str, ...]
    indices: dict[str, np.ndarray]
    angles: dict[str, np.ndarray]
    distances: dict[str, np.ndarray]
    camera_centers: np.ndarray
    scene_distance_scale: float
    camera_distance_scale: float
    surface_count: int

    @classmethod
    def load(cls, path: Path) -> "GeometryFeatureCache":
        with guarded_h5(path, "load candidate geometry only; probabilities are not read") as handle:
            view_ids = tuple(value.decode() if isinstance(value, bytes) else str(value) for value in handle["view_ids"][:])
            indices = {view_id: np.asarray(handle["views"][view_id]["surface_indices"][:], dtype=np.int32) for view_id in view_ids}
            angles = {view_id: np.asarray(handle["views"][view_id]["angles"][:], dtype=np.float32) for view_id in view_ids}
            distances = {view_id: np.asarray(handle["views"][view_id]["distances"][:], dtype=np.float32) for view_id in view_ids}
            camera = np.asarray(handle["camera_centers"][:], dtype=np.float32)
            all_distance = np.concatenate([value for value in distances.values() if len(value)])
            scene_scale = float(np.median(all_distance)) if len(all_distance) else 1.0
            deltas = camera[:, None, :] - camera[None, :, :]
            pairwise = np.linalg.norm(deltas, axis=2)
            camera_scale = float(np.max(pairwise)) if len(camera) else 1.0
            surface_count = len(handle["surface_ids"])
        return cls(view_ids, indices, angles, distances, camera, max(scene_scale, 1e-12), max(camera_scale, 1e-12), surface_count)


def _summary(values: np.ndarray, count: int) -> tuple[float, ...]:
    if count == 0:
        return (0.0,) * 4
    value = np.asarray(values, dtype=np.float64)
    return float(value.mean()), float(value.std()), float(np.quantile(value, 0.75)), float(value.max())


def _camera_features(candidate_index: int, selected_indices: list[int], camera: np.ndarray, distance_scale: float) -> tuple[float, float, float]:
    if not selected_indices:
        return 0.0, 0.0, 0.0
    candidate = camera[candidate_index]
    selected = camera[selected_indices]
    origin = camera.mean(axis=0)
    cvec = candidate - origin
    svec = selected - origin
    cnorm = max(float(np.linalg.norm(cvec)), 1e-12)
    snorm = np.maximum(np.linalg.norm(svec, axis=1), 1e-12)
    cosine = np.clip((svec @ cvec) / (snorm * cnorm), -1.0, 1.0)
    separation = np.arccos(cosine) / np.pi
    distances = np.linalg.norm(selected - candidate, axis=1) / distance_scale
    return float(separation.mean()), float(separation.min()), float(distances.mean())


class CausalFeatureExtractor:
    def __init__(self, manager: RevealManager, geometry: GeometryFeatureCache):
        validate_schema()
        if manager.view_ids != geometry.view_ids or manager.surface_count != geometry.surface_count:
            raise ValueError("manager/geometry cache mismatch")
        self.manager = manager
        self.geometry = geometry

    def global_features(self, state: CausalState) -> np.ndarray:
        cs = state.csmax
        observed = cs.observed_mask
        probability = cs.fused_crack_probability[observed]
        entropy = cs.observed_surface_entropy[observed]
        count = cs.visible_view_count[observed].astype(np.float64)
        disagreement = state.disagreement[observed]
        if len(probability):
            p_mean, p_std = float(probability.mean()), float(probability.std())
            p25, p50, p75 = (float(value) for value in np.quantile(probability, (0.25, 0.50, 0.75)))
            e_mean, e_std = float(entropy.mean()), float(entropy.std())
            c_mean, c_std = float(count.mean() / len(self.manager.view_ids)), float(count.std() / len(self.manager.view_ids))
            d_mean, d_std, d_max = float(disagreement.mean()), float(disagreement.std()), float(disagreement.max())
        else:
            p_mean = p_std = p25 = p50 = p75 = e_mean = e_std = c_mean = c_std = d_mean = d_std = d_max = 0.0
        selected_fraction = cs.selected_view_count / len(self.manager.view_ids)
        values = np.asarray([
            selected_fraction, 1.0 - selected_fraction, float(observed.mean()),
            float(np.mean(cs.hard_semantic_label == 1)), p_mean, p_std, p25, p50, p75,
            e_mean, e_std, c_mean, c_std, d_mean, d_std, d_max,
        ], dtype=np.float32)
        if values.shape != (len(GLOBAL_FEATURES),) or not np.isfinite(values).all():
            raise ValueError("invalid global causal features")
        return values

    def candidate_features(self, state: CausalState, candidate_view_ids: tuple[str, ...]) -> np.ndarray:
        cs = state.csmax
        disagreement = state.disagreement
        index_by_id = {view_id: index for index, view_id in enumerate(self.geometry.view_ids)}
        selected_indices = [index_by_id[value] for value in cs.selected_view_ids]
        rows = []
        for view_id in candidate_view_ids:
            indices = self.geometry.indices[view_id]
            observed_local = cs.observed_mask[indices]
            observed_indices = indices[observed_local]
            visible_fraction = len(indices) / self.geometry.surface_count
            newly_visible_fraction = int(np.count_nonzero(~observed_local)) / self.geometry.surface_count
            overlap = float(observed_local.mean()) if len(indices) else 0.0
            probability = _summary(cs.fused_crack_probability[observed_indices], len(observed_indices))
            entropy = _summary(cs.observed_surface_entropy[observed_indices], len(observed_indices))
            disagree = _summary(disagreement[observed_indices], len(observed_indices))
            angles = self.geometry.angles[view_id].astype(np.float64) / 180.0
            distances = self.geometry.distances[view_id].astype(np.float64) / self.geometry.scene_distance_scale
            if len(angles):
                angle_values = (float(angles.mean()), float(angles.std()), float(np.quantile(angles, 0.25)), float(np.quantile(angles, 0.75)))
                distance_values = (float(distances.mean()), float(distances.std()))
            else:
                angle_values = (0.0,) * 4
                distance_values = (0.0, 0.0)
            camera_values = _camera_features(index_by_id[view_id], selected_indices, self.geometry.camera_centers, self.geometry.camera_distance_scale)
            rows.append((visible_fraction, newly_visible_fraction, overlap, *probability, *entropy, *disagree, *angle_values, *distance_values, *camera_values))
        values = np.asarray(rows, dtype=np.float32)
        if values.shape != (len(candidate_view_ids), len(CANDIDATE_FEATURES)) or not np.isfinite(values).all():
            raise ValueError(f"invalid candidate causal feature shape: {values.shape}")
        return values

    def extract(self, state: CausalState) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        selected = set(state.csmax.selected_view_ids)
        candidates = tuple(view_id for view_id in self.manager.view_ids if view_id not in selected)
        if not candidates:
            raise ValueError("terminal state has no candidates")
        return candidates, self.global_features(state), self.candidate_features(state, candidates)
