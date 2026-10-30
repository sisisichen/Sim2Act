from dataclasses import fields
import inspect

from sim2act_baselines.active.base import CandidateViewGeometry, ViewSelector


def test_candidate_geometry_has_no_future_image_or_prediction_fields():
    names = {field.name for field in fields(CandidateViewGeometry)}
    forbidden = {"image", "rgb", "probabilities", "prediction", "hard_prediction", "metric_improvement", "ground_truth"}
    assert names.isdisjoint(forbidden)
    assert names == {"view_id", "stable_index", "camera_pose", "visible_surface_indices", "visible_surface_area"}


def test_deployable_selector_signature_has_no_oracle_context():
    parameters = set(inspect.signature(ViewSelector.select).parameters)
    assert parameters == {"self", "candidates", "state", "rng"}
