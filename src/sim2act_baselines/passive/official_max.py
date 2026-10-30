import numpy as np

from crackstructures.mapping.fuser import NaiveMaxFuser

from .base import UpstreamFuserAdapter


class OfficialMax(UpstreamFuserAdapter):
    name = "CS-Max-Official"
    upstream_class = NaiveMaxFuser
    method_parameters = {"angle_range_degrees": [90.0, 270.0], "aggregation": "maximum"}

    def method_view_mask(self, visibility, angles, distances):
        return visibility & (angles >= 90) & (angles <= 270)

