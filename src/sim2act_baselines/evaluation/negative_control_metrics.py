from __future__ import annotations

import numpy as np


def negative_control_active_metrics(state, surface_area_m2: float) -> dict:
    prediction = np.asarray(state.hard_semantic_label) == 1
    count = int(prediction.sum())
    total = len(prediction)
    observed_probability = state.fused_crack_probability[state.observed_mask]
    maximum = float(np.max(observed_probability)) if len(observed_probability) else None
    return {
        "predicted_positive_surface_count": count,
        "predicted_positive_surface_fraction": float(count / total) if total else 0.0,
        "predicted_positive_surface_area_m2": float(count * surface_area_m2 / total) if total else 0.0,
        "false_positive_scene_indicator": bool(count > 0),
        "maximum_fused_crack_probability": maximum,
        "observed_coverage": float(state.observed_mask.mean()) if total else 0.0,
    }
