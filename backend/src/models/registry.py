"""Model registry for qMachina stream producer dispatch.

Adding a new model requires only one edit: add an entry to _REGISTRY.
Unknown model_id raises ValueError with the list of valid models.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

_REGISTRY: dict[str, str] = {
    "vacuum_pressure": "src.models.vacuum_pressure.stream_pipeline",
    "ema_ensemble":    "src.models.ema_ensemble.stream_pipeline",
}


def get_async_stream_events(model_id: str) -> Callable[..., Any]:
    """Return the async_stream_events coroutine for the given model_id.

    Raises:
        ValueError: If model_id is not registered.
    """
    module_path = _REGISTRY.get(model_id)
    if module_path is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Valid: {list(_REGISTRY)}"
        )
    return getattr(importlib.import_module(module_path), "async_stream_events")


def get_build_model_config(model_id: str) -> Callable[..., dict]:
    """Return the build_model_config function for the given model_id.

    Raises:
        ValueError: If model_id is not registered.
    """
    module_path = _REGISTRY.get(model_id)
    if module_path is None:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Valid: {list(_REGISTRY)}"
        )
    return getattr(importlib.import_module(module_path), "build_model_config")
