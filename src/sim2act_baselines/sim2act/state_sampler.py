from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from sim2act_baselines.active.active_state import CurrentCrackState
from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.reveal_manager import RevealManager, RevealedViewEvidence

from .access_guard import guarded_read_text


STEP4B_TRIALS = Path("/autodl-fs/data/sim2act_runtime/outputs/step4b_active_full/trial_shards/crackensembles")


@dataclass(frozen=True)
class CausalState:
    csmax: CurrentCrackState
    probability_sum: np.ndarray
    probability_sum_squares: np.ndarray
    probability_count: np.ndarray

    @classmethod
    def empty(cls, surface_count: int) -> "CausalState":
        return cls(
            csmax=IncrementalCSMax.empty(surface_count),
            probability_sum=np.zeros(surface_count, dtype=np.float32),
            probability_sum_squares=np.zeros(surface_count, dtype=np.float32),
            probability_count=np.zeros(surface_count, dtype=np.uint16),
        )

    def update(self, evidence: RevealedViewEvidence) -> "CausalState":
        csmax = IncrementalCSMax.update(self.csmax, evidence)
        total = self.probability_sum.copy()
        square = self.probability_sum_squares.copy()
        count = self.probability_count.copy()
        angles = np.asarray(evidence.angles, dtype=np.float32)
        valid = (angles >= np.float32(90.0)) & (angles <= np.float32(270.0))
        indices = np.asarray(evidence.visible_surface_indices, dtype=np.int64)[valid]
        if len(indices):
            crack = np.asarray(evidence.crack_probabilities[valid, 1], dtype=np.float32)
            total[indices] += crack
            square[indices] += crack * crack
            count[indices] += np.uint16(1)
        for array in (total, square, count):
            array.setflags(write=False)
        return CausalState(csmax=csmax, probability_sum=total, probability_sum_squares=square, probability_count=count)

    @property
    def disagreement(self) -> np.ndarray:
        result = np.full(len(self.probability_count), np.nan, dtype=np.float32)
        mask = self.probability_count > 0
        if mask.any():
            n = self.probability_count[mask].astype(np.float32)
            mean = self.probability_sum[mask] / n
            result[mask] = np.maximum(self.probability_sum_squares[mask] / n - mean * mean, 0.0)
        return result


@dataclass(frozen=True)
class StateSpec:
    sample_id: str
    split: str
    trajectory_seed: int
    selected_view_ids: tuple[str, ...]
    selected_set_hash: str
    source_policies: tuple[str, ...]
    budget: int
    total_views: int

    @property
    def uid(self) -> str:
        payload = f"{self.sample_id}|{self.trajectory_seed}|{self.selected_set_hash}"
        return hashlib.sha256(payload.encode()).hexdigest()


def normalized_budgets(total_views: int) -> tuple[int, ...]:
    values = [3]
    values.extend(max(3, int(math.ceil(total_views * fraction))) for fraction in (0.10, 0.20, 0.40, 0.60, 0.80))
    return tuple(sorted({value for value in values if value < total_views}))


def selected_set_hash(selected_view_ids: Iterable[str]) -> str:
    canonical = sorted(str(value) for value in selected_view_ids)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def trial_path(sample_id: str, seed: int, policy: str) -> Path:
    return STEP4B_TRIALS / sample_id / str(seed) / f"{policy.lower().replace('-', '_')}.json"


def collect_state_specs(sample_id: str, split: str, seed: int, policies: Iterable[str], total_views: int) -> list[StateSpec]:
    by_hash: dict[str, StateSpec] = {}
    for policy in policies:
        path = trial_path(sample_id, seed, policy)
        payload = json.loads(guarded_read_text(path, "read frozen STEP4B development trajectory"))
        if payload["dataset"] != "crackensembles" or payload["scene_id"] != sample_id or int(payload["seed"]) != seed:
            raise RuntimeError(f"trajectory provenance mismatch: {path}")
        sequence = tuple(str(value) for value in payload["selected_view_ids"])
        if len(sequence) != total_views or len(set(sequence)) != total_views:
            raise RuntimeError(f"invalid complete trajectory: {path}")
        for budget in normalized_budgets(total_views):
            selected = sequence[:budget]
            digest = selected_set_hash(selected)
            prior = by_hash.get(digest)
            if prior is None:
                by_hash[digest] = StateSpec(sample_id, split, seed, selected, digest, (policy,), budget, total_views)
            elif policy not in prior.source_policies:
                by_hash[digest] = replace(prior, source_policies=(*prior.source_policies, policy))
    return sorted(by_hash.values(), key=lambda value: (value.budget, value.selected_set_hash))


def replay_state(manager_template: RevealManager, selected_view_ids: Iterable[str]) -> CausalState:
    manager = manager_template.fork()
    state = CausalState.empty(manager.surface_count)
    for view_id in selected_view_ids:
        state = state.update(manager.reveal(str(view_id)))
    return state
