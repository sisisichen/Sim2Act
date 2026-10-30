from __future__ import annotations

import numpy as np

from .active_state import CurrentCrackState
from .reveal_manager import RevealedViewEvidence


class IncrementalCSMax:
    """Exact incremental form of the pinned float16 CS-Max-Official backend."""

    name = "CS-Max-Official"
    class_weight = np.asarray([1.0, 10.0], dtype=np.float16)

    @staticmethod
    def empty(surface_count: int) -> CurrentCrackState:
        observed = np.zeros(surface_count, dtype=bool)
        return CurrentCrackState(
            observed_mask=observed,
            fused_crack_probability=np.full(surface_count, np.nan, dtype=np.float32),
            visible_view_count=np.zeros(surface_count, dtype=np.uint16),
            hard_semantic_label=np.zeros(surface_count, dtype=np.uint8),
            observed_surface_entropy=np.full(surface_count, np.nan, dtype=np.float32),
            selected_view_ids=(),
            official_weighted_probability=np.zeros((surface_count, 2), dtype=np.float32),
        )

    @classmethod
    def update(cls, state: CurrentCrackState, evidence: RevealedViewEvidence) -> CurrentCrackState:
        if evidence.view_id in state.selected_view_ids:
            raise ValueError(f"duplicate selected view: {evidence.view_id}")
        scores = np.asarray(state.official_weighted_probability, dtype=np.float16).copy()
        counts = np.asarray(state.visible_view_count, dtype=np.uint16).copy()
        angles32 = np.asarray(evidence.angles, dtype=np.float32)
        angles16 = angles32.astype(np.float16)
        score_eligible = (angles16 >= np.float16(90.0)) & (angles16 <= np.float16(270.0))
        score_indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[score_eligible]
        if len(score_indices):
            weighted = np.asarray(evidence.crack_probabilities[score_eligible], dtype=np.float16).copy()
            weighted *= cls.class_weight
            scores[score_indices] = np.maximum(scores[score_indices], weighted)
        # The pinned adapter reports its valid mask/count from the original
        # float32 angle array, although upstream aggregation converts to float16.
        # Preserve both conventions to match the complete official result.
        count_eligible = (angles32 >= np.float32(90.0)) & (angles32 <= np.float32(270.0))
        count_indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[count_eligible]
        if len(count_indices):
            if np.any(counts[count_indices] == np.iinfo(np.uint16).max):
                raise OverflowError("visible view count overflow")
            counts[count_indices] += np.uint16(1)
        observed = counts > 0
        hard = np.argmax(scores, axis=1).astype(np.uint8)
        probability = np.full(state.surface_count, np.nan, dtype=np.float32)
        probability[observed] = scores[observed, 1].astype(np.float32) / np.float32(10.0)
        entropy = np.full(state.surface_count, np.nan, dtype=np.float32)
        if observed.any():
            p = np.clip(probability[observed], np.float32(1e-7), np.float32(1.0 - 1e-7))
            entropy[observed] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        return CurrentCrackState(
            observed_mask=observed,
            fused_crack_probability=probability,
            visible_view_count=counts,
            hard_semantic_label=hard,
            observed_surface_entropy=entropy,
            selected_view_ids=(*state.selected_view_ids, evidence.view_id),
            official_weighted_probability=scores.astype(np.float32),
        )

    @classmethod
    def replay(cls, surface_count: int, evidence_sequence) -> CurrentCrackState:
        state = cls.empty(surface_count)
        for evidence in evidence_sequence:
            state = cls.update(state, evidence)
        return state
