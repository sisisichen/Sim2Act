import pytest

from sim2act_baselines.active.budget_protocol import clip_absolute_budget, normalized_budget


def test_absolute_budget_clipping_and_normalized_ceiling():
    assert clip_absolute_budget(30, 9) == 9
    assert clip_absolute_budget(1, 9) == 3
    assert normalized_budget(0.10, 32) == 4
    assert normalized_budget(0.20, 9) == 3
    assert normalized_budget(1.0, 9) == 9


def test_budget_rejects_invalid_scene_and_fraction():
    with pytest.raises(ValueError):
        clip_absolute_budget(3, 2)
    with pytest.raises(ValueError):
        normalized_budget(0, 9)
