from __future__ import annotations


GLOBAL_FEATURES = (
    "selected_view_fraction",
    "remaining_view_fraction",
    "observed_surface_fraction",
    "predicted_positive_fraction",
    "fused_probability_mean",
    "fused_probability_std",
    "fused_probability_q25",
    "fused_probability_q50",
    "fused_probability_q75",
    "entropy_mean",
    "entropy_std",
    "visible_view_count_mean_normalized",
    "visible_view_count_std_normalized",
    "disagreement_mean",
    "disagreement_std",
    "disagreement_max",
)

CANDIDATE_FEATURES = (
    "visible_surface_fraction",
    "newly_visible_surface_fraction",
    "overlap_with_observed_fraction",
    "current_probability_mean",
    "current_probability_std",
    "current_probability_q75",
    "current_probability_max",
    "current_entropy_mean",
    "current_entropy_std",
    "current_entropy_q75",
    "current_entropy_max",
    "current_disagreement_mean",
    "current_disagreement_std",
    "current_disagreement_q75",
    "current_disagreement_max",
    "candidate_angle_mean_normalized",
    "candidate_angle_std_normalized",
    "candidate_angle_q25_normalized",
    "candidate_angle_q75_normalized",
    "candidate_distance_mean_normalized",
    "candidate_distance_std_normalized",
    "selected_camera_angular_separation_mean",
    "selected_camera_angular_separation_min",
    "selected_camera_centre_distance_mean_normalized",
)

FORBIDDEN_FEATURE_TOKENS = (
    "ground_truth", "gt_", "iou", "f1", "oracle", "gain", "candidate_probability",
    "rgb", "sample_id", "split", "view_id",
)


def validate_schema() -> None:
    names = GLOBAL_FEATURES + CANDIDATE_FEATURES
    if len(names) != len(set(names)):
        raise ValueError("feature names are not unique")
    violations = [name for name in names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if violations:
        raise ValueError(f"forbidden feature names: {violations}")
