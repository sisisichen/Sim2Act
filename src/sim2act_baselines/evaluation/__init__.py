"""Research components with optional dependencies loaded on demand."""

from importlib import import_module

_EXPORTS = {'evaluate_obj_pair': 'official_tolerance_metrics', 'StructuredRunResult': 'empty_policy', 'binary_crack_metrics': 'empty_policy', 'length_metrics': 'empty_policy', 'structured_result': 'empty_policy', 'ExtractionTrace': 'extraction_trace', 'ExtractionTracer': 'extraction_trace', 'GT_SURFACE_TOLERANCE_M': 'ground_truth_diagnostics', 'audit_annotation': 'ground_truth_diagnostics', 'first_empty_stage': 'ground_truth_diagnostics', 'ground_truth_surface_mask': 'ground_truth_diagnostics', 'visibility_audit': 'ground_truth_diagnostics'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
