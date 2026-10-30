from __future__ import annotations

import numpy as np


def normalized_acquisition_auc(budgets, values) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or not len(x):
        raise ValueError("budget/value curve mismatch")
    if len(x) == 1:
        return float(y[0])
    if np.any(np.diff(x) <= 0):
        raise ValueError("budgets must be strictly increasing")
    return float(np.trapz(y, x=x) / (x[-1] - x[0]))
