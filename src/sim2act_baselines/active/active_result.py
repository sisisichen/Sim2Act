from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveStepResult:
    dataset: str
    scene_id: str
    seed: int
    selector: str
    budget: int
    total_valid_views: int
    selected_view_id: str
    candidate_score: float | None
    selector_runtime_seconds: float
    cumulative_runtime_seconds: float
    view_redundancy: float
    observed_surface_coverage: float
    mean_observed_surface_entropy: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    iou: float | None
    predicted_positive_count: int
    all_view_reference_iou: float | None
    all_view_reference_f1: float
    normalized_iou_to_all_view: float | None
    normalized_f1_to_all_view: float | None
    exceeds_all_view_reference: bool
    maximum_fused_crack_probability: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
