from __future__ import annotations

from collections import defaultdict


def aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], float(row["tolerance_m"]))].append(row)
    output = []
    for (method, tolerance), values in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "tolerance_m": tolerance,
                "scene_count": len(values),
                "mean_precision": sum(item["precision"] for item in values) / len(values),
                "mean_recall": sum(item["recall"] for item in values) / len(values),
                "mean_f1": sum(item["f1"] for item in values) / len(values),
                "mean_iou": sum(item["iou"] for item in values) / len(values),
                "mean_relative_length_error": sum(item["relative_length_error"] for item in values) / len(values),
            }
        )
    return output

