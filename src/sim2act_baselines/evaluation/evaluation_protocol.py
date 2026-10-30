from __future__ import annotations

from enum import Enum


class EvaluationContext(str, Enum):
    FINAL_ALL_VIEW = "final_all_view"
    REPRESENTATIVE_BUDGET = "representative_budget"
    FINAL_METHOD_OUTPUT = "final_method_output"
    ACTIVE_CANDIDATE_SCORING = "active_candidate_scoring"


_GEOMETRY_ALLOWED = {
    EvaluationContext.FINAL_ALL_VIEW,
    EvaluationContext.REPRESENTATIVE_BUDGET,
    EvaluationContext.FINAL_METHOD_OUTPUT,
}


def assert_geometry_extraction_allowed(context: EvaluationContext | str) -> None:
    resolved = EvaluationContext(context)
    if resolved not in _GEOMETRY_ALLOWED:
        raise RuntimeError(
            "downstream geometry extraction is forbidden during active candidate scoring; "
            "use primary semantic-mapping state"
        )
