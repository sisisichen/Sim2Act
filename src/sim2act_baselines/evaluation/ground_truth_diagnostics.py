from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from enstrect.evaluation.metrics import interpolate_G
from enstrect.evaluation.utils.lines import obj_to_G

from .official_tolerance_metrics import graph_length


GT_SURFACE_TOLERANCE_M = 0.01


def obj_counts(path: Path) -> tuple[int, int]:
    vertices = 0
    polylines = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            vertices += 1
        elif fields and fields[0] == "l" and len(fields) >= 3:
            polylines += 1
    return vertices, polylines


def centerline_graph_and_points(path: Path, gap: float = 0.001) -> tuple[nx.Graph, np.ndarray]:
    graph = obj_to_G(path)
    if len(graph.nodes) == 0:
        return graph, np.empty((0, 3), dtype=np.float64)
    interpolated = interpolate_G(graph, gap=gap)
    points = np.asarray([data["pos"] for _, data in interpolated.nodes(data=True)], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"invalid ground-truth centerline geometry: {path}")
    return graph, points


def ground_truth_surface_mask(
    surface_points: np.ndarray,
    centerline_points: np.ndarray,
    tolerance_m: float = GT_SURFACE_TOLERANCE_M,
) -> np.ndarray:
    surface = np.asarray(surface_points, dtype=np.float64)
    centerline = np.asarray(centerline_points, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != 3 or not np.isfinite(surface).all():
        raise ValueError("invalid surface points")
    if centerline.size == 0:
        return np.zeros(len(surface), dtype=bool)
    if centerline.ndim != 2 or centerline.shape[1] != 3 or not np.isfinite(centerline).all():
        raise ValueError("invalid centerline points")
    distance = cKDTree(centerline).query(surface, workers=-1)[0]
    return distance <= float(tolerance_m)


def visibility_audit(ground_truth_mask: np.ndarray, visibility_mask: np.ndarray) -> dict[str, Any]:
    gt = np.asarray(ground_truth_mask, dtype=bool)
    visibility = np.asarray(visibility_mask, dtype=bool)
    if visibility.ndim != 2 or visibility.shape[0] != len(gt):
        raise ValueError("ground-truth/visibility alignment mismatch")
    gt_count = int(gt.sum())
    if gt_count == 0:
        counts = np.zeros(0, dtype=np.int64)
    else:
        counts = visibility[gt].sum(axis=1)
    return {
        "ground_truth_positive_surface_count": gt_count,
        "views_with_any_visible_ground_truth": int(np.count_nonzero(visibility[gt].any(axis=0))) if gt_count else 0,
        "fraction_visible_at_least_1": float(np.mean(counts >= 1)) if gt_count else 0.0,
        "fraction_visible_at_least_2": float(np.mean(counts >= 2)) if gt_count else 0.0,
        "fraction_visible_at_least_3": float(np.mean(counts >= 3)) if gt_count else 0.0,
        "fraction_visible_at_least_5": float(np.mean(counts >= 5)) if gt_count else 0.0,
    }


def audit_annotation(path: Path, surface_points: np.ndarray, visibility_mask: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    try:
        vertices, polylines = obj_counts(path)
        graph, centerline = centerline_graph_and_points(path)
        length = graph_length(graph)
        mask = ground_truth_surface_mask(surface_points, centerline)
        visible = visibility_audit(mask, visibility_mask)
        if vertices == 0 or polylines == 0 or length <= 0:
            classification = "A. GT_EMPTY"
        elif visible["ground_truth_positive_surface_count"] == 0 or visible["views_with_any_visible_ground_truth"] == 0:
            classification = "B. GT_NONEMPTY_BUT_NOT_VISIBLE"
        else:
            classification = "C. GT_NONEMPTY_AND_VISIBLE"
        return {
            "ground_truth_vertex_count": vertices,
            "ground_truth_polyline_count": polylines,
            "ground_truth_length_m": length,
            "interpolated_ground_truth_point_count": len(centerline),
            **visible,
            "ground_truth_classification": classification,
            "diagnostic_surface_definition": f"sampled surface within {GT_SURFACE_TOLERANCE_M} m of distributed crack.obj centerline",
            "error": None,
        }, mask
    except Exception as error:
        return {
            "ground_truth_vertex_count": 0,
            "ground_truth_polyline_count": 0,
            "ground_truth_length_m": 0.0,
            "interpolated_ground_truth_point_count": 0,
            "ground_truth_positive_surface_count": 0,
            "views_with_any_visible_ground_truth": 0,
            "fraction_visible_at_least_1": 0.0,
            "fraction_visible_at_least_2": 0.0,
            "fraction_visible_at_least_3": 0.0,
            "fraction_visible_at_least_5": 0.0,
            "ground_truth_classification": "D. GT_INVALID",
            "diagnostic_surface_definition": f"sampled surface within {GT_SURFACE_TOLERANCE_M} m of distributed crack.obj centerline",
            "error": f"{type(error).__name__}: {error}",
        }, np.zeros(len(surface_points), dtype=bool)


def first_empty_stage(counts: dict[str, int]) -> str | None:
    order = (
        ("positive_image_pixels", "Stage 1"),
        ("visible_surface_elements", "Stage 2"),
        ("valid_evidence_elements", "Stage 3"),
        ("positive_fused_elements", "Stage 4"),
        ("clusters_before_filter", "Stage 5"),
        ("clusters_after_filter", "Stage 6"),
        ("points_before_contraction", "Stage 7"),
        ("points_after_contraction", "Stage 8"),
        ("final_vertices", "Stage 9"),
        ("final_polylines", "Stage 10"),
    )
    return next((stage for key, stage in order if int(counts[key]) == 0), None)
