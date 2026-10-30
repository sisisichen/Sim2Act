import numpy as np

from sim2act_baselines.active.base import CandidateViewGeometry
from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.random_selector import RandomSelector
from sim2act_baselines.active.reveal_manager import RevealedViewEvidence


def test_random_selector_exhausts_pool_without_duplicates():
    candidates = tuple(CandidateViewGeometry(str(i), i, np.zeros(3), np.asarray([i], np.int32), 1.0) for i in range(7))
    state = IncrementalCSMax.empty(7)
    rng = np.random.default_rng(3)
    chosen = []
    for _ in candidates:
        decision = RandomSelector().select(candidates, state, rng)
        chosen.append(decision.view_id)
        evidence = RevealedViewEvidence(decision.view_id, np.asarray([int(decision.view_id)], np.int32), np.asarray([[0.8, 0.2]], np.float32), np.asarray([180], np.float32), np.asarray([1], np.float32))
        state = IncrementalCSMax.update(state, evidence)
    assert len(chosen) == len(set(chosen)) == 7
