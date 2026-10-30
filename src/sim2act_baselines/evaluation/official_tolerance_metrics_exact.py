from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from enstrect.evaluation.metrics import interpolate_G
from enstrect.evaluation.utils.lines import obj_to_G

from .official_tolerance_metrics import TOLERANCES, graph_length


def evaluate_obj_pair_exact_kdtree(
    ground_truth_obj: Path,
    prediction_obj: Path,
    tolerances: tuple[float, ...] = TOLERANCES,
) -> list[dict[str, float | int]]:
    """Exact official centerline metric with bounded-memory nearest queries.

    This preserves the upstream 1 mm interpolation, Euclidean nearest-distance
    definitions, and TP/FN/FP counting. It replaces only SciPy's materialized
    rectangular distance matrix with exact cKDTree nearest-neighbour queries.
    """
    true_graph = obj_to_G(ground_truth_obj) if Path(ground_truth_obj).exists() else nx.Graph()
    pred_graph = obj_to_G(prediction_obj) if Path(prediction_obj).exists() and Path(prediction_obj).stat().st_size else nx.Graph()
    true_interp = interpolate_G(true_graph, gap=0.001)
    pred_interp = interpolate_G(pred_graph, gap=0.001)
    true_pos = np.asarray([data["pos"] for _, data in true_interp.nodes(data=True)], dtype=np.float64).reshape(-1, 3)
    pred_pos = np.asarray([data["pos"] for _, data in pred_interp.nodes(data=True)], dtype=np.float64).reshape(-1, 3)
    if len(true_pos) and len(pred_pos):
        true_to_pred = cKDTree(pred_pos).query(true_pos, k=1, workers=-1)[0]
        pred_to_true = cKDTree(true_pos).query(pred_pos, k=1, workers=-1)[0]
    else:
        true_to_pred = np.full(len(true_pos), np.inf, dtype=np.float64)
        pred_to_true = np.full(len(pred_pos), np.inf, dtype=np.float64)
    gt_length = graph_length(true_graph)
    pred_length = graph_length(pred_graph)
    rows = []
    for tolerance in tolerances:
        tp = int(np.count_nonzero(true_to_pred <= tolerance))
        fn = int(len(true_to_pred) - tp)
        fp = int(np.count_nonzero(pred_to_true > tolerance))
        denominator = tp + fn + fp
        iou = float(tp / denominator) if denominator else 1.0
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({
            "tolerance_m": float(tolerance),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "predicted_centerline_length_m": pred_length,
            "ground_truth_centerline_length_m": gt_length,
            "relative_length_error": abs(pred_length - gt_length) / gt_length if gt_length else float("nan"),
            "evaluator_implementation": "official_exact_kdtree_equivalent",
        })
    return rows
