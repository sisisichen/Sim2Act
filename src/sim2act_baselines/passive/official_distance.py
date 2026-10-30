import numpy as np

from crackstructures.mapping.fuser import DistanceRangeFuser, DistanceWeightedFuser

from .base import UpstreamFuserAdapter


class OfficialDistanceRange(UpstreamFuserAdapter):
    name = "CS-Distance-Range-Official"
    upstream_class = DistanceRangeFuser
    method_parameters = {
        "angle_range_degrees": [90.0, 270.0],
        "distance_rule": "distance < per-point median over all views (current upstream semantics)",
        "aggregation": "nanmean",
    }

    def method_view_mask(self, visibility, angles, distances):
        median = np.median(distances, axis=1)[:, None]
        return visibility & (angles >= 90) & (angles <= 270) & (distances < median)


class OfficialDistanceWeighted(UpstreamFuserAdapter):
    name = "CS-Distance-Weighted-Official"
    upstream_class = DistanceWeightedFuser
    method_parameters = {
        "angle_range_degrees": [90.0, 270.0],
        "weight": "1-(d-min(D))/(max(D)-min(D)) using current upstream all-view D",
        "aggregation": "nanmean",
    }

    def method_view_mask(self, visibility, angles, distances):
        return visibility & (angles >= 90) & (angles <= 270)

