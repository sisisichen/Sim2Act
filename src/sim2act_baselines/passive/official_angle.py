import numpy as np

from crackstructures.mapping.fuser import AngleRangeFuser, AngleWeightedFuser

from .base import UpstreamFuserAdapter


class OfficialAngleRange(UpstreamFuserAdapter):
    name = "CS-Angle-Range-Official"
    upstream_class = AngleRangeFuser
    method_parameters = {"angle_range_degrees": [130.0, 230.0], "aggregation": "nanmean"}

    def method_view_mask(self, visibility, angles, distances):
        return visibility & (angles >= 130) & (angles <= 230)


class OfficialAngleWeighted(UpstreamFuserAdapter):
    name = "CS-Angle-Weighted-Official"
    upstream_class = AngleWeightedFuser
    method_parameters = {"weight": "max(0,-cos(angle_degrees))", "aggregation": "nanmean"}

    def method_view_mask(self, visibility, angles, distances):
        return visibility & (np.maximum(0.0, -np.cos(np.deg2rad(angles))) > 0)

