import numpy as np

from sim2act_baselines.active.incremental_csmax import IncrementalCSMax
from sim2act_baselines.active.reveal_manager import RevealedViewEvidence
from sim2act_baselines.evaluation.negative_control_metrics import negative_control_active_metrics


def test_negative_control_metrics_are_separate_and_defined_without_iou():
    state = IncrementalCSMax.empty(3)
    state = IncrementalCSMax.update(state, RevealedViewEvidence("v", np.asarray([0, 1], np.int32), np.asarray([[0.1, 0.9], [0.99, 0.01]], np.float32), np.asarray([180, 180], np.float32), np.ones(2, np.float32)))
    metrics = negative_control_active_metrics(state, 3.0)
    assert metrics["predicted_positive_surface_count"] == 1
    assert metrics["false_positive_scene_indicator"] is True
    assert "iou" not in metrics
