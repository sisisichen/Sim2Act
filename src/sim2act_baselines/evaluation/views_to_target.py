from __future__ import annotations


def views_to_fraction_of_reference(budgets, values, reference: float | None, fraction: float) -> int | None:
    if reference is None or reference == 0:
        return None
    target = float(reference) * float(fraction)
    return next((int(budget) for budget, value in zip(budgets, values) if float(value) >= target), None)
