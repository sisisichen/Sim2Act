from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sim2act_baselines.active.base import stable_view_key
from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.reveal_manager import RevealedViewEvidence
from sim2act_baselines.evaluation.empty_policy import binary_crack_metrics

from .state_sampler import CausalState


@dataclass(frozen=True)
class CandidateGain:
    view_id: str
    iou_current: float
    iou_after: float
    delta_iou: float
    tp_after: int
    fp_after: int
    fn_after: int


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    return binary_crack_metrics(np.asarray(truth, dtype=bool), np.asarray(prediction, dtype=bool))


def brute_force_candidate_gain(state: CausalState, evidence: RevealedViewEvidence, truth: np.ndarray) -> CandidateGain:
    return brute_force_candidate_gains(state, [evidence], truth)[0]


def brute_force_candidate_gains(state: CausalState, evidences: Iterable[RevealedViewEvidence], truth: np.ndarray) -> list[CandidateGain]:
    evidence_list = list(evidences)
    if not evidence_list:
        return []
    before = _metrics(truth, state.csmax.hard_semantic_label == 1)
    # Independent full-surface reference: materialize the complete official
    # float16 score field for every candidate, apply each candidate, then
    # recompute predictions and semantic metrics over all N surfaces. Candidate
    # batching changes only execution order, not the brute-force definition.
    base_scores = np.asarray(state.csmax.official_weighted_probability, dtype=np.float16)
    scores = np.broadcast_to(base_scores, (len(evidence_list), *base_scores.shape)).copy()
    for row, evidence in enumerate(evidence_list):
        angles = np.asarray(evidence.angles, dtype=np.float16)
        valid = (angles >= np.float16(90.0)) & (angles <= np.float16(270.0))
        indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[valid]
        if len(indices):
            weighted = np.asarray(evidence.crack_probabilities[valid], dtype=np.float16).copy()
            weighted *= IncrementalCSMax.class_weight
            scores[row, indices] = np.maximum(scores[row, indices], weighted)
    predictions = scores[:, :, 1] > scores[:, :, 0]
    truth_row = np.asarray(truth, dtype=bool)[None, :]
    tp = np.count_nonzero(predictions & truth_row, axis=1)
    fp = np.count_nonzero(predictions & ~truth_row, axis=1)
    fn = np.count_nonzero(~predictions & truth_row, axis=1)
    current = float(before["iou"])
    results = []
    for evidence, tp_value, fp_value, fn_value in zip(evidence_list, tp, fp, fn):
        union = int(tp_value + fp_value + fn_value)
        resulting = float(tp_value / union) if union else 0.0
        results.append(CandidateGain(evidence.view_id, current, resulting, resulting - current, int(tp_value), int(fp_value), int(fn_value)))
    return results


def optimized_candidate_gain(state: CausalState, evidence: RevealedViewEvidence, truth: np.ndarray) -> CandidateGain:
    truth = np.asarray(truth, dtype=bool)
    current_prediction = state.csmax.hard_semantic_label == 1
    before = _metrics(truth, current_prediction)
    tp, fp, fn = int(before["tp"]), int(before["fp"]), int(before["fn"])
    angle16 = np.asarray(evidence.angles, dtype=np.float16)
    valid = (angle16 >= np.float16(90.0)) & (angle16 <= np.float16(270.0))
    indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[valid]
    if len(indices):
        weighted = np.asarray(evidence.crack_probabilities[valid], dtype=np.float16).copy()
        weighted *= IncrementalCSMax.class_weight
        current_local = np.asarray(state.csmax.official_weighted_probability[indices], dtype=np.float16)
        new_prediction = np.argmax(np.maximum(current_local, weighted), axis=1) == 1
        old_prediction = current_prediction[indices]
        changed = new_prediction != old_prediction
        if changed.any():
            local_truth = truth[indices][changed]
            old_value = old_prediction[changed]
            new_value = new_prediction[changed]
            tp += int(np.count_nonzero(local_truth & new_value) - np.count_nonzero(local_truth & old_value))
            fp += int(np.count_nonzero(~local_truth & new_value) - np.count_nonzero(~local_truth & old_value))
            fn += int(np.count_nonzero(local_truth & ~new_value) - np.count_nonzero(local_truth & ~old_value))
    union = tp + fp + fn
    after = float(tp / union) if union else 0.0
    current = float(before["iou"])
    return CandidateGain(evidence.view_id, current, after, after - current, tp, fp, fn)


def rank_gains(gains: Iterable[CandidateGain]) -> list[CandidateGain]:
    return sorted(gains, key=lambda value: (-value.delta_iou, stable_view_key(value.view_id)))
