from __future__ import annotations

import time

import numpy as np

from .base import SelectionDecision, ViewSelector, candidate_geometry_stats, require_candidates, stable_view_key


class UADSUncertaintyNBVAdapted(ViewSelector):
    name = "UADS-Uncertainty-NBV-Adapted"

    def select(self, candidates, state, rng) -> SelectionDecision:
        started = time.perf_counter()
        ranked = []
        for candidate in require_candidates(candidates, state):
            indices = candidate.visible_surface_indices
            observed_indices = indices[state.observed_mask[indices]]
            score = float(np.sum(state.observed_surface_entropy[observed_indices], dtype=np.float64) * candidate.area_per_visible_element)
            new_area, redundancy = candidate_geometry_stats(candidate, state)
            ranked.append((candidate, score, new_area, redundancy))
        candidate, score, new_area, redundancy = min(
            ranked,
            key=lambda item: (-item[1], -item[2], stable_view_key(item[0].view_id)),
        )
        return SelectionDecision(candidate.view_id, score, time.perf_counter() - started, new_area, redundancy)
