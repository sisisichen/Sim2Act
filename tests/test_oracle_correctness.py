import numpy as np

from sim2act_baselines.active.base import CandidateViewGeometry
from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.oracle_nbv import OracleContext, OracleNBVUpperBound
from sim2act_baselines.active.reveal_manager import RevealedViewEvidence
from sim2act_baselines.evaluation.empty_policy import binary_crack_metrics


def test_oracle_matches_independent_bruteforce_and_tie_rules():
    candidates = tuple(CandidateViewGeometry(str(i), i, np.zeros(3), np.asarray([i], np.int32), 1.0) for i in range(3))
    evidence = {
        "0": RevealedViewEvidence("0", np.asarray([0], np.int32), np.asarray([[0.9, 0.1]], np.float32), np.asarray([180], np.float32), np.asarray([1], np.float32)),
        "1": RevealedViewEvidence("1", np.asarray([1], np.int32), np.asarray([[0.1, 0.9]], np.float32), np.asarray([180], np.float32), np.asarray([1], np.float32)),
        "2": RevealedViewEvidence("2", np.asarray([2], np.int32), np.asarray([[0.1, 0.9]], np.float32), np.asarray([180], np.float32), np.asarray([1], np.float32)),
    }
    truth = np.asarray([False, True, False])
    state = IncrementalCSMax.empty(3)
    oracle = OracleNBVUpperBound().select_oracle(candidates, state, OracleContext(truth, evidence.__getitem__))
    brute = []
    for candidate in candidates:
        trial = IncrementalCSMax.update(state, evidence[candidate.view_id])
        metrics = binary_crack_metrics(truth, trial.hard_semantic_label == 1)
        brute.append((candidate.view_id, metrics["iou"], metrics["fp"], metrics["f1"]))
    expected = min(brute, key=lambda item: (-item[1], item[2], -item[3], int(item[0])))
    assert oracle.view_id == expected[0] == "1"
    assert abs(oracle.score - expected[1]) <= 1e-9


def test_oracle_delta_scoring_matches_full_state_updates_on_random_pool():
    rng = np.random.default_rng(41)
    surface_count = 73
    truth = rng.random(surface_count) > 0.72
    base_indices = np.arange(surface_count, dtype=np.int32)
    base_probabilities = rng.random((surface_count, 2), dtype=np.float32)
    base_probabilities /= base_probabilities.sum(axis=1, keepdims=True)
    state = IncrementalCSMax.update(
        IncrementalCSMax.empty(surface_count),
        RevealedViewEvidence("base", base_indices, base_probabilities, np.full(surface_count, 180, np.float32), np.ones(surface_count, np.float32)),
    )
    candidates, evidence = [], {}
    for view in range(6):
        indices = np.sort(rng.choice(surface_count, size=37, replace=False)).astype(np.int32)
        probabilities = rng.random((len(indices), 2), dtype=np.float32)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        view_id = str(view)
        candidates.append(CandidateViewGeometry(view_id, view, np.zeros(3), indices, float(len(indices))))
        evidence[view_id] = RevealedViewEvidence(view_id, indices, probabilities, np.full(len(indices), 180, np.float32), np.ones(len(indices), np.float32))
    decision = OracleNBVUpperBound().select_oracle(tuple(candidates), state, OracleContext(truth, evidence.__getitem__))
    brute = []
    for candidate in candidates:
        trial = IncrementalCSMax.update(state, evidence[candidate.view_id])
        metrics = binary_crack_metrics(truth, trial.hard_semantic_label == 1)
        brute.append((candidate.view_id, metrics["iou"], metrics["fp"], metrics["f1"]))
    expected = min(brute, key=lambda item: (-item[1], item[2], -item[3], int(item[0])))
    assert decision.view_id == expected[0]
    assert abs(decision.score - expected[1]) <= 1e-9
