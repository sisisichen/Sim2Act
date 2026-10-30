from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ro(array: np.ndarray, dtype=None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CurrentCrackState:
    """A snapshot derived exclusively from views that have already been revealed."""

    observed_mask: np.ndarray
    fused_crack_probability: np.ndarray
    visible_view_count: np.ndarray
    hard_semantic_label: np.ndarray
    observed_surface_entropy: np.ndarray
    selected_view_ids: tuple[str, ...]
    official_weighted_probability: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.observed_mask)
        arrays = {
            "observed_mask": _ro(self.observed_mask, bool),
            "fused_crack_probability": _ro(self.fused_crack_probability, np.float32),
            "visible_view_count": _ro(self.visible_view_count, np.uint16),
            "hard_semantic_label": _ro(self.hard_semantic_label, np.uint8),
            "observed_surface_entropy": _ro(self.observed_surface_entropy, np.float32),
            "official_weighted_probability": _ro(self.official_weighted_probability, np.float32),
        }
        if any(len(value) != count for value in arrays.values()):
            raise ValueError("active state length mismatch")
        if arrays["official_weighted_probability"].shape != (count, 2):
            raise ValueError("official weighted probability must have shape (N,2)")
        observed = arrays["observed_mask"]
        probability = arrays["fused_crack_probability"]
        entropy = arrays["observed_surface_entropy"]
        if not np.isnan(probability[~observed]).all() or not np.isnan(entropy[~observed]).all():
            raise ValueError("unobserved probability and entropy must be NaN")
        if np.isnan(probability[observed]).any() or np.isnan(entropy[observed]).any():
            raise ValueError("observed probability and entropy must be finite")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "selected_view_ids", tuple(str(value) for value in self.selected_view_ids))

    @property
    def surface_count(self) -> int:
        return len(self.observed_mask)

    @property
    def selected_view_count(self) -> int:
        return len(self.selected_view_ids)
