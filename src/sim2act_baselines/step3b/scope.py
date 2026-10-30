from __future__ import annotations

from sim2act_baselines.datasets.crackensembles_adapter import official_split_for_id


def validate_crackensembles_validation_ids(sample_ids) -> tuple[str, ...]:
    values = tuple(f"{int(value):04d}" for value in sample_ids)
    invalid = [value for value in values if official_split_for_id(value) != "validation"]
    if invalid:
        raise ValueError(f"STEP 3B rejects non-validation CrackEnsembles IDs: {invalid}")
    if len(set(values)) != len(values):
        raise ValueError("duplicate CrackEnsembles validation ID")
    return values


def negative_control_policy(*, structure: str, ground_truth_positive_count: int) -> dict[str, bool]:
    is_negative = structure == "bridge_g"
    if is_negative and ground_truth_positive_count != 0:
        raise ValueError("bridge_g negative control cannot have positive crack GT")
    return {
        "positive_primary_set": not is_negative,
        "negative_control_set": is_negative,
        "include_in_positive_scene_mean": not is_negative,
        "include_in_false_positive_analysis": is_negative,
    }
