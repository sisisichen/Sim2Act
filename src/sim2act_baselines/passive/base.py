from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
import time
from typing import Any, Sequence

import numpy as np
import torch

from sim2act_baselines.upstream.crackstructures_adapter import CLASS_WEIGHT


@dataclass(frozen=True)
class FusionResult:
    fused_probability: np.ndarray
    fused_label: np.ndarray
    visible_view_count: np.ndarray
    selected_view_ids: tuple[int, ...]
    valid_surface_mask: np.ndarray
    method_name: str
    method_parameters: dict[str, Any]
    provenance: dict[str, Any]
    runtime_seconds: float


class PassiveFuser(ABC):
    name: str

    @abstractmethod
    def fuse(
        self,
        point_or_surface_ids,
        per_view_probabilities,
        visibility_mask,
        view_angles,
        view_distances,
        selected_view_ids,
    ) -> FusionResult:
        raise NotImplementedError


class UpstreamFuserAdapter(PassiveFuser):
    upstream_class = None
    method_parameters: dict[str, Any] = {}

    def method_view_mask(
        self,
        visibility: np.ndarray,
        angles: np.ndarray,
        distances: np.ndarray,
    ) -> np.ndarray:
        return visibility

    @staticmethod
    def _selection(selected: Sequence[int], total_views: int) -> tuple[int, ...]:
        result = tuple(int(value) for value in selected)
        if not result:
            raise ValueError("selected_view_ids is mandatory and cannot be empty")
        if len(set(result)) != len(result):
            raise ValueError("a view cannot be selected twice")
        if min(result) < 0 or max(result) >= total_views:
            raise IndexError(result)
        return result

    def fuse(
        self,
        point_or_surface_ids,
        per_view_probabilities,
        visibility_mask,
        view_angles,
        view_distances,
        selected_view_ids,
    ) -> FusionResult:
        ids = np.asarray(point_or_surface_ids)
        probabilities = np.asarray(per_view_probabilities)
        visibility = np.asarray(visibility_mask, dtype=bool)
        angles = np.asarray(view_angles)
        distances = np.asarray(view_distances)
        if probabilities.shape[:2] != visibility.shape or probabilities.shape[2] != 2:
            raise ValueError("probability/visibility shape mismatch")
        if angles.shape != visibility.shape or distances.shape != visibility.shape:
            raise ValueError("view-condition shape mismatch")
        if len(ids) != probabilities.shape[0]:
            raise ValueError("surface ID alignment mismatch")
        selected = self._selection(selected_view_ids, probabilities.shape[1])
        index = np.asarray(selected, dtype=np.int64)
        selected_probabilities = probabilities[:, index, :].copy()
        selected_visibility = visibility[:, index]
        selected_angles = angles[:, index]
        selected_distances = distances[:, index]
        selected_probabilities[~selected_visibility] = np.nan
        tensor = torch.as_tensor(selected_probabilities, dtype=torch.float16).clone()
        conditions = SimpleNamespace(
            visible=selected_visibility.astype(np.float16),
            angles=selected_angles.astype(np.float16),
            distances=selected_distances.astype(np.float16),
        )
        fuser = self.upstream_class(class_weight=torch.tensor(CLASS_WEIGHT, dtype=torch.float16))
        started = time.perf_counter()
        aggregated, labels = fuser(tensor, conditions)
        runtime = time.perf_counter() - started
        fused = aggregated.cpu().numpy().astype(np.float32, copy=False)
        hard = labels.cpu().numpy().astype(np.uint8, copy=False)
        method_mask = self.method_view_mask(selected_visibility, selected_angles, selected_distances)
        counts = method_mask.sum(axis=1).astype(np.uint16)
        valid = counts > 0
        if not np.isfinite(fused).all():
            raise ValueError(f"{self.name} produced NaN/Inf")
        return FusionResult(
            fused_probability=fused,
            fused_label=hard,
            visible_view_count=counts,
            selected_view_ids=selected,
            valid_surface_mask=valid,
            method_name=self.name,
            method_parameters=dict(self.method_parameters),
            provenance={
                "adapter": f"{type(self).__module__}.{type(self).__name__}",
                "upstream_class": f"{self.upstream_class.__module__}.{self.upstream_class.__name__}",
                "class_weight": CLASS_WEIGHT,
                "selected_view_enforcement": "array slice before upstream call",
            },
            runtime_seconds=runtime,
        )

