"""Signal library for VP experiment harness.

Provides a registry of signal classes (statistical and ML) that can be
instantiated by name. Each signal class implements either the
StatisticalSignal or MLSignal interface from signals.base.

Registry population happens at import time as signal modules register
themselves via register_signal(). To add a new signal:
    1. Create a module in signals/statistical/ or signals/ml/
    2. Implement the appropriate base class
    3. Call register_signal(name, cls) at module level
    4. Import the module here
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MLSignal, StatisticalSignal

SIGNAL_REGISTRY: dict[str, type[StatisticalSignal] | type[MLSignal]] = {}


def register_signal(
    name: str,
    cls: type[StatisticalSignal] | type[MLSignal],
) -> None:
    """Register a signal class by canonical name.

    Args:
        name: Unique identifier for the signal (e.g. "ads", "svm_sp").
        cls: Signal class implementing StatisticalSignal or MLSignal.

    Raises:
        ValueError: If name is already registered to a different class.
    """
    if name in SIGNAL_REGISTRY and SIGNAL_REGISTRY[name] is not cls:
        raise ValueError(
            f"Signal name '{name}' already registered to "
            f"{SIGNAL_REGISTRY[name].__name__}, cannot re-register to "
            f"{cls.__name__}"
        )
    SIGNAL_REGISTRY[name] = cls


def get_signal_class(
    name: str,
) -> type[StatisticalSignal] | type[MLSignal]:
    """Retrieve a registered signal class by name.

    Args:
        name: Canonical signal name.

    Returns:
        The registered signal class.

    Raises:
        KeyError: If the name is not in the registry, with a list of
            available signal names for debugging.
    """
    if name not in SIGNAL_REGISTRY:
        available = sorted(SIGNAL_REGISTRY.keys())
        raise KeyError(f"Unknown signal '{name}'. Available: {available}")
    return SIGNAL_REGISTRY[name]
