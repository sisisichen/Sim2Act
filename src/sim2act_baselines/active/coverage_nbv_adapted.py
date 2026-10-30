from __future__ import annotations

import time

from .base import SelectionDecision, ViewSelector, candidate_geometry_stats, require_candidates, stable_view_key


class CoverageNBVAdapted(ViewSelector):
    name = "Coverage-NBV-Adapted"

    def select(self, candidates, state, rng) -> SelectionDecision:
        started = time.perf_counter()
        ranked = []
        for candidate in require_candidates(candidates, state):
            new_area, redundancy = candidate_geometry_stats(candidate, state)
            ranked.append((candidate, new_area, redundancy))
        candidate, score, redundancy = min(
            ranked,
            key=lambda item: (-item[1], -item[0].visible_surface_area, stable_view_key(item[0].view_id)),
        )
        return SelectionDecision(candidate.view_id, score, time.perf_counter() - started, score, redundancy)
