from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


VALID_STATUSES = {
    "VALID_NONEMPTY",
    "VALID_EMPTY_GT_NONEMPTY",
    "VALID_EMPTY_GT_EMPTY",
    "PIPELINE_ERROR",
    "INVALID_GEOMETRY",
    "INVALID_DATA",
    "RESOURCE_LIMIT_ORIGINAL_EXTRACTOR",
    "MEMORY_SAFE_EXTRACTION_SUCCESS",
    "MEMORY_SAFE_EXTRACTION_NOT_EQUIVALENT",
}


@dataclass(frozen=True)
class StructuredRunResult:
    status: str
    is_empty: bool
    empty_reason: str | None
    ground_truth_status: str | None
    semantic_positive_count: int
    final_vertex_count: int
    final_polyline_count: int
    error: str | None = None
    semantic_metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(self.status)
        if min(self.semantic_positive_count, self.final_vertex_count, self.final_polyline_count) < 0:
            raise ValueError("counts cannot be negative")
        expected_empty = self.final_vertex_count == 0 or self.final_polyline_count == 0
        if self.is_empty != expected_empty and self.status not in {
            "PIPELINE_ERROR",
            "RESOURCE_LIMIT_ORIGINAL_EXTRACTOR",
            "MEMORY_SAFE_EXTRACTION_NOT_EQUIVALENT",
        }:
            raise ValueError("is_empty is inconsistent with final geometry counts")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structured_result(
    *,
    ground_truth_positive_count: int,
    semantic_positive_count: int,
    final_vertex_count: int,
    final_polyline_count: int,
    ground_truth_status: str | None,
    first_empty_stage: str | None,
    error: BaseException | None = None,
) -> StructuredRunResult:
    if error is not None:
        return StructuredRunResult(
            status="PIPELINE_ERROR",
            is_empty=True,
            empty_reason=first_empty_stage or "runtime_exception",
            ground_truth_status=ground_truth_status,
            semantic_positive_count=max(0, int(semantic_positive_count)),
            final_vertex_count=max(0, int(final_vertex_count)),
            final_polyline_count=max(0, int(final_polyline_count)),
            error=f"{type(error).__name__}: {error}",
        )
    empty = final_vertex_count == 0 or final_polyline_count == 0
    if not empty:
        status = "VALID_NONEMPTY"
        reason = None
    elif ground_truth_positive_count > 0:
        status = "VALID_EMPTY_GT_NONEMPTY"
        reason = first_empty_stage or "final_geometry_empty"
    else:
        status = "VALID_EMPTY_GT_EMPTY"
        reason = first_empty_stage or "correct_negative_scene"
    return StructuredRunResult(
        status=status,
        is_empty=empty,
        empty_reason=reason,
        ground_truth_status=ground_truth_status,
        semantic_positive_count=int(semantic_positive_count),
        final_vertex_count=int(final_vertex_count),
        final_polyline_count=int(final_polyline_count),
    )


def resource_limited_original_extractor_result(
    *,
    semantic_positive_count: int,
    ground_truth_status: str | None = None,
    detail: str = "original extractor exceeded the resource limit",
    semantic_metrics: dict[str, Any] | None = None,
) -> StructuredRunResult:
    """Preserve a valid semantic result when original geometry extraction OOMs."""
    return StructuredRunResult(
        status="RESOURCE_LIMIT_ORIGINAL_EXTRACTOR",
        is_empty=True,
        empty_reason=None,
        ground_truth_status=ground_truth_status,
        semantic_positive_count=max(0, int(semantic_positive_count)),
        final_vertex_count=0,
        final_polyline_count=0,
        error=detail,
        semantic_metrics=semantic_metrics,
    )


def binary_crack_metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    true = np.asarray(ground_truth, dtype=bool).reshape(-1)
    pred = np.asarray(prediction, dtype=bool).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("ground-truth/prediction shape mismatch")
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    gt_positive = int(true.sum())
    pred_positive = int(pred.sum())
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    if gt_positive:
        iou: float | None = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
    else:
        iou = None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "ground_truth_positive_count": gt_positive,
        "predicted_positive_count": pred_positive,
        "correct_negative_scene": gt_positive == 0 and pred_positive == 0,
        "include_in_positive_scene_iou_mean": gt_positive > 0,
    }


def length_metrics(ground_truth_length: float, predicted_length: float) -> dict[str, float | None]:
    gt = float(ground_truth_length)
    pred = float(predicted_length)
    if gt < 0 or pred < 0:
        raise ValueError("lengths cannot be negative")
    return {
        "ground_truth_length": gt,
        "predicted_length": pred,
        "relative_length_error": abs(pred - gt) / gt if gt > 0 else None,
        "false_positive_length": pred if gt == 0 else 0.0,
    }
