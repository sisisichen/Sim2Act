"""Research components with optional dependencies loaded on demand."""

from importlib import import_module

_EXPORTS = {'Sim2ActViewRanker': 'model', 'Sim2ActClosedLoopSelector': 'closed_loop_selector'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
