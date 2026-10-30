from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from enstrect.evaluation.metrics import clCloudIoU, interpolate_G
from enstrect.evaluation.utils.lines import obj_to_G


TOLERANCES = (0.005, 0.01, 0.02, 0.04, 0.08)


def graph_length(graph: nx.Graph) -> float:
    length = 0.0
    for left, right in graph.edges:
        a = np.asarray(graph.nodes[left]["pos"], dtype=np.float64)
        b = np.asarray(graph.nodes[right]["pos"], dtype=np.float64)
        length += float(np.linalg.norm(a - b))
    return length


def evaluate_obj_pair(
    ground_truth_obj: Path,
    prediction_obj: Path,
    tolerances: tuple[float, ...] = TOLERANCES,
) -> list[dict[str, float | int]]:
    true_graph = obj_to_G(ground_truth_obj) if ground_truth_obj.exists() else nx.Graph()
    pred_graph = obj_to_G(prediction_obj) if prediction_obj.exists() and prediction_obj.stat().st_size else nx.Graph()
    true_interp = interpolate_G(true_graph, gap=0.001)
    pred_interp = interpolate_G(pred_graph, gap=0.001)
    true_pos = np.asarray([data["pos"] for _, data in true_interp.nodes(data=True)], dtype=np.float64)
    pred_pos = np.asarray([data["pos"] for _, data in pred_interp.nodes(data=True)], dtype=np.float64)
    gt_length = graph_length(true_graph)
    pred_length = graph_length(pred_graph)
    rows: list[dict[str, float | int]] = []
    for tolerance in tolerances:
        iou, tp, fn, fp = clCloudIoU(true_pos, pred_pos, tol=tolerance)
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "tolerance_m": float(tolerance),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": float(iou),
                "predicted_centerline_length_m": pred_length,
                "ground_truth_centerline_length_m": gt_length,
                "relative_length_error": abs(pred_length - gt_length) / gt_length if gt_length else float("nan"),
            }
        )
    return rows

