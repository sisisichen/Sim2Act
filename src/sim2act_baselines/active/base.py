from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Sequence

import numpy as np

from .active_state import CurrentCrackState


def readonly(array: np.ndarray, dtype=None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def stable_view_key(view_id: str) -> tuple[Any, ...]:
    """Natural, deterministic ordering for numeric and mixed view IDs."""
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(view_id)))


@dataclass(frozen=True)
class CandidateViewGeometry:
    """Geometry-only candidate record; it intentionally has no image/evidence fields."""

    view_id: str
    stable_index: int
    camera_pose: np.ndarray
    visible_surface_indices: np.ndarray
    visible_surface_area: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "view_id", str(self.view_id))
        object.__setattr__(self, "camera_pose", readonly(self.camera_pose, np.float32))
        indices = readonly(self.visible_surface_indices, np.int32).reshape(-1)
        if len(indices) and np.any(indices[1:] <= indices[:-1]):
            raise ValueError("candidate visibility indices must be strictly increasing")
        object.__setattr__(self, "visible_surface_indices", indices)
        if self.visible_surface_area < 0:
            raise ValueError("visible surface area cannot be negative")

    @property
    def area_per_visible_element(self) -> float:
        return self.visible_surface_area / len(self.visible_surface_indices) if len(self.visible_surface_indices) else 0.0


@dataclass(frozen=True)
class SelectionDecision:
    view_id: str
    score: float
    selector_runtime_seconds: float
    newly_visible_area: float
    view_redundancy: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class ViewSelector(ABC):
    """Deployable selector API: only geometry and already-revealed state are accepted."""

    name: str
    deployable: bool = True

    @abstractmethod
    def select(
        self,
        candidates: Sequence[CandidateViewGeometry],
        state: CurrentCrackState,
        rng: np.random.Generator,
    ) -> SelectionDecision:
        raise NotImplementedError


def candidate_geometry_stats(candidate: CandidateViewGeometry, state: CurrentCrackState) -> tuple[float, float]:
    indices = candidate.visible_surface_indices
    if not len(indices):
        return 0.0, 0.0
    observed = state.observed_mask[indices]
    area = candidate.area_per_visible_element
    new_area = float(np.count_nonzero(~observed) * area)
    redundancy = float(np.count_nonzero(observed) / len(indices))
    return new_area, redundancy


def require_candidates(candidates: Sequence[CandidateViewGeometry], state: CurrentCrackState) -> tuple[CandidateViewGeometry, ...]:
    selected = set(state.selected_view_ids)
    result = tuple(candidate for candidate in candidates if candidate.view_id not in selected)
    if not result:
        raise ValueError("no unselected candidate views remain")
    if len({candidate.view_id for candidate in result}) != len(result):
        raise ValueError("duplicate candidate view IDs")
    return result
