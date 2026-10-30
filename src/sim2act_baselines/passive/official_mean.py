from crackstructures.mapping.fuser import NaiveMeanFuser

from .base import UpstreamFuserAdapter


class OfficialMean(UpstreamFuserAdapter):
    name = "CS-Mean-Official"
    upstream_class = NaiveMeanFuser
    method_parameters = {"angle_range_degrees": [90.0, 270.0], "aggregation": "nanmean"}

    def method_view_mask(self, visibility, angles, distances):
        return visibility & (angles >= 90) & (angles <= 270)

