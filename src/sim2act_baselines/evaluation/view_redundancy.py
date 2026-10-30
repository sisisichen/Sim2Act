from __future__ import annotations

import numpy as np


def visible_overlap_fraction(candidate_indices, observed_mask) -> float:
    indices = np.asarray(candidate_indices, dtype=np.int64)
    observed = np.asarray(observed_mask, dtype=bool)
    return float(np.count_nonzero(observed[indices]) / len(indices)) if len(indices) else 0.0
