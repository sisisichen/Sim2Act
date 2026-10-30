from __future__ import annotations

from crackstructures.mapping.fuser import (
    AngleBestFuser,
    AngleDistanceRangeFuser,
    AngleRangeFuser,
    AngleWeightedFuser,
    DistanceBestFuser,
    DistanceRangeFuser,
    DistanceWeightedFuser,
    NaiveAngleMaxRangeFuser,
    NaiveMaxFuser,
    NaiveMeanFuser,
    NaiveMedianFuser,
)


UPSTREAM_FUSERS = {
    "CS-Max-Official": NaiveMaxFuser,
    "CS-Mean-Official": NaiveMeanFuser,
    "CS-Median-Official": NaiveMedianFuser,
    "CS-Angle-Best-Official": AngleBestFuser,
    "CS-Angle-Range-Official": AngleRangeFuser,
    "CS-Angle-Weighted-Official": AngleWeightedFuser,
    "CS-Distance-Best-CurrentSource": DistanceBestFuser,
    "CS-Distance-Range-Official": DistanceRangeFuser,
    "CS-Distance-Weighted-Official": DistanceWeightedFuser,
    "CS-Angle-Distance-Range-Official": AngleDistanceRangeFuser,
    "CS-Angle-Range-Max-Official": NaiveAngleMaxRangeFuser,
}

CLASS_WEIGHT = [1.0, 10.0]
EXTRACTION_PARAMETERS = {
    "eps_m": 0.01,
    "min_points": 20,
    "min_samples_cluster": 10,
    "init_contraction": 100.0,
}

