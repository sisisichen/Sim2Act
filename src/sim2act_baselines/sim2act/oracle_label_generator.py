from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2act_baselines.active.reveal_manager import RevealManager

from .exact_candidate_gain import CandidateGain, brute_force_candidate_gains, optimized_candidate_gain, rank_gains
from .state_sampler import CausalState


@dataclass(frozen=True)
class OracleLabelSet:
    candidate_view_ids: tuple[str, ...]
    iou_current: float
    iou_after: np.ndarray
    delta_iou: np.ndarray
    ranking: tuple[str, ...]
    top_view_id: str


def generate_exact_labels(manager: RevealManager, state: CausalState, truth: np.ndarray) -> OracleLabelSet:
    selected = set(state.csmax.selected_view_ids)
    loader = manager.oracle_loader()
    candidates = [view_id for view_id in manager.view_ids if view_id not in selected]
    gains = [optimized_candidate_gain(state, loader(view_id), truth) for view_id in candidates]
    if not gains:
        raise ValueError("terminal state has no candidate labels")
    ranked = rank_gains(gains)
    return OracleLabelSet(
        candidate_view_ids=tuple(candidates),
        iou_current=float(gains[0].iou_current),
        iou_after=np.asarray([value.iou_after for value in gains], dtype=np.float64),
        delta_iou=np.asarray([value.delta_iou for value in gains], dtype=np.float64),
        ranking=tuple(value.view_id for value in ranked),
        top_view_id=ranked[0].view_id,
    )


def equivalence_record(manager: RevealManager, state: CausalState, truth: np.ndarray) -> dict:
    selected = set(state.csmax.selected_view_ids)
    loader = manager.oracle_loader()
    optimized: list[CandidateGain] = []
    brute: list[CandidateGain] = []
    evidences = [loader(view_id) for view_id in manager.view_ids if view_id not in selected]
    optimized = [optimized_candidate_gain(state, evidence, truth) for evidence in evidences]
    brute = brute_force_candidate_gains(state, evidences, truth)
    opt_rank = rank_gains(optimized)
    brute_rank = rank_gains(brute)
    iou_error = max(abs(a.iou_after - b.iou_after) for a, b in zip(optimized, brute))
    delta_error = max(abs(a.delta_iou - b.delta_iou) for a, b in zip(optimized, brute))
    finite = bool(np.isfinite([value.iou_after for value in optimized + brute]).all())
    return {
        "candidate_count": len(optimized),
        "maximum_iou_after_difference": float(iou_error),
        "maximum_delta_iou_difference": float(delta_error),
        "ranking_identical": [value.view_id for value in opt_rank] == [value.view_id for value in brute_rank],
        "top_candidate_identical": opt_rank[0].view_id == brute_rank[0].view_id,
        "optimized_top_candidate": opt_rank[0].view_id,
        "brute_top_candidate": brute_rank[0].view_id,
        "finite": finite,
        "pass": iou_error <= 1e-12 and delta_error <= 1e-12 and finite
        and [value.view_id for value in opt_rank] == [value.view_id for value in brute_rank],
    }
