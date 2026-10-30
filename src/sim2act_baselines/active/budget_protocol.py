from __future__ import annotations

import math


ABSOLUTE_BUDGETS = (3, 5, 10, 15, 20, 30)
NORMALIZED_FRACTIONS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def clip_absolute_budget(budget: int, total_valid_views: int, initial_view_count: int = 3) -> int:
    if total_valid_views < initial_view_count:
        raise ValueError("insufficient candidate views")
    return min(total_valid_views, max(initial_view_count, int(budget)))


def normalized_budget(fraction: float, total_valid_views: int, initial_view_count: int = 3) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("normalized fraction must be in (0,1]")
    return clip_absolute_budget(math.ceil(float(fraction) * total_valid_views), total_valid_views, initial_view_count)


def canonical_budgets(total_valid_views: int, initial_view_count: int = 3) -> dict[str, int]:
    result = {f"absolute_{value}": clip_absolute_budget(value, total_valid_views, initial_view_count) for value in ABSOLUTE_BUDGETS}
    result.update({f"fraction_{fraction:.2f}": normalized_budget(fraction, total_valid_views, initial_view_count) for fraction in NORMALIZED_FRACTIONS})
    return result
