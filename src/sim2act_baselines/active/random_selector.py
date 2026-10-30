from __future__ import annotations

import time

import numpy as np

from .base import SelectionDecision, ViewSelector, candidate_geometry_stats, require_candidates


class RandomSelector(ViewSelector):
    name = "Random"

    def select(self, candidates, state, rng: np.random.Generator) -> SelectionDecision:
        started = time.perf_counter()
        available = require_candidates(candidates, state)
        candidate = available[int(rng.integers(0, len(available)))]
        new_area, redundancy = candidate_geometry_stats(candidate, state)
        return SelectionDecision(candidate.view_id, float("nan"), time.perf_counter() - started, new_area, redundancy)
