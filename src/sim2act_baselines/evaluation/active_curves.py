from __future__ import annotations

import numpy as np


def best_so_far(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("curve must be one-dimensional")
    return np.maximum.accumulate(array)


def attach_best_so_far(rows: list[dict], field: str = "iou", output_field: str = "best_iou_so_far") -> list[dict]:
    values = best_so_far([row[field] for row in rows])
    return [{**row, output_field: float(value)} for row, value in zip(rows, values)]
