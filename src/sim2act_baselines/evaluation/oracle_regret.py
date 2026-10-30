from __future__ import annotations

import numpy as np


def per_budget_oracle_regret(method_values, oracle_values) -> np.ndarray:
    method = np.asarray(method_values, dtype=np.float64)
    oracle = np.asarray(oracle_values, dtype=np.float64)
    if method.shape != oracle.shape:
        raise ValueError("oracle/method budget mismatch")
    return oracle - method


def integrated_oracle_regret(budgets, method_values, oracle_values) -> float:
    x = np.asarray(budgets, dtype=np.float64)
    regret = per_budget_oracle_regret(method_values, oracle_values)
    if len(x) == 1:
        return float(regret[0])
    return float(np.trapz(regret, x=x) / (x[-1] - x[0]))
