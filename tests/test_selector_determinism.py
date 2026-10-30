import numpy as np

from sim2act_baselines.active.base import CandidateViewGeometry
from sim2act_baselines.active.coverage_nbv_adapted import CoverageNBVAdapted
from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.random_selector import RandomSelector


def candidates():
    return (
        CandidateViewGeometry("10", 0, np.zeros(3), np.asarray([0, 1], np.int32), 2.0),
        CandidateViewGeometry("2", 1, np.zeros(3), np.asarray([2, 3], np.int32), 2.0),
    )


def test_deterministic_tie_break_uses_lower_natural_view_id():
    state = IncrementalCSMax.empty(4)
    assert CoverageNBVAdapted().select(candidates(), state, np.random.default_rng(0)).view_id == "2"


def test_random_is_deterministic_under_fixed_seed():
    state = IncrementalCSMax.empty(4)
    first = RandomSelector().select(candidates(), state, np.random.default_rng(2025)).view_id
    second = RandomSelector().select(candidates(), state, np.random.default_rng(2025)).view_id
    assert first == second
