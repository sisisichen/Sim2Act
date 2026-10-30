from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from sim2act_baselines.evaluation.empty_policy import binary_crack_metrics

from .active_state import CurrentCrackState
from .base import CandidateViewGeometry, SelectionDecision, candidate_geometry_stats, require_candidates, stable_view_key
from .incremental_csmax import IncrementalCSMax
from .reveal_manager import RevealedViewEvidence


@dataclass(frozen=True)
class OracleContext:
    """Privileged capability; active runner constructs this only for Oracle."""

    ground_truth: np.ndarray
    unrevealed_evidence_loader: Callable[[str], RevealedViewEvidence]

    def __post_init__(self) -> None:
        truth = np.asarray(self.ground_truth, dtype=bool).copy()
        truth.setflags(write=False)
        object.__setattr__(self, "ground_truth", truth)


class OracleNBVUpperBound:
    name = "Oracle-NBV-Upper-Bound"
    deployable = False

    def select_oracle(
        self,
        candidates: tuple[CandidateViewGeometry, ...],
        state: CurrentCrackState,
        context: OracleContext,
    ) -> SelectionDecision:
        started = time.perf_counter()
        ranked = []
        current_prediction = state.hard_semantic_label == 1
        base = binary_crack_metrics(context.ground_truth, current_prediction)
        for candidate in require_candidates(candidates, state):
            evidence = context.unrevealed_evidence_loader(candidate.view_id)
            angle16 = np.asarray(evidence.angles, dtype=np.float16)
            eligible = (angle16 >= np.float16(90.0)) & (angle16 <= np.float16(270.0))
            indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[eligible]
            tp, fp, fn = int(base["tp"]), int(base["fp"]), int(base["fn"])
            if len(indices):
                weighted = np.asarray(evidence.crack_probabilities[eligible], dtype=np.float16).copy()
                weighted *= IncrementalCSMax.class_weight
                current_local = np.asarray(state.official_weighted_probability[indices], dtype=np.float16)
                new_local_prediction = np.argmax(np.maximum(current_local, weighted), axis=1) == 1
                old_local_prediction = current_prediction[indices]
                changed = new_local_prediction != old_local_prediction
                if changed.any():
                    local_truth = context.ground_truth[indices][changed]
                    old_value = old_local_prediction[changed]
                    new_value = new_local_prediction[changed]
                    tp += int(np.count_nonzero(local_truth & new_value) - np.count_nonzero(local_truth & old_value))
                    fp += int(np.count_nonzero(~local_truth & new_value) - np.count_nonzero(~local_truth & old_value))
                    fn += int(np.count_nonzero(local_truth & ~new_value) - np.count_nonzero(local_truth & ~old_value))
            precision = float(tp / (tp + fp)) if tp + fp else 0.0
            recall = float(tp / (tp + fn)) if tp + fn else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            iou = float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0
            new_area, redundancy = candidate_geometry_stats(candidate, state)
            ranked.append((candidate, iou, fp, f1, new_area, redundancy))
        candidate, iou, fp, f1, new_area, redundancy = min(
            ranked,
            key=lambda item: (-item[1], item[2], -item[3], stable_view_key(item[0].view_id)),
        )
        return SelectionDecision(
            candidate.view_id,
            iou,
            time.perf_counter() - started,
            new_area,
            redundancy,
            diagnostics={"resulting_iou": iou, "resulting_fp": fp, "resulting_f1": f1},
        )
