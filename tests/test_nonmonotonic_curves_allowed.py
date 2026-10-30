import numpy as np

from sim2act_baselines.evaluation.active_curves import best_so_far


def test_raw_curve_decrease_is_retained_and_best_curve_is_separate():
    raw = np.asarray([0.2, 0.4, 0.3, 0.5])
    best = best_so_far(raw)
    assert raw.tolist() == [0.2, 0.4, 0.3, 0.5]
    assert best.tolist() == [0.2, 0.4, 0.4, 0.5]
