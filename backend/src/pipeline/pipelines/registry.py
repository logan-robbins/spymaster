"""Pipeline registry.

Maps pipeline names to pipeline builders.
"""
from typing import Callable, Dict, List

from src.pipeline.core.pipeline import Pipeline


# Registry of pipeline name -> pipeline builder
_PIPELINES: Dict[str, Callable[[], Pipeline]] = {}


def _register_pipelines():
    """Register all available pipelines."""
    global _PIPELINES

    # Import here to avoid circular imports
    from src.pipeline.pipelines.es_pipeline import build_es_pipeline

    _PIPELINES = {
        'es_pipeline': build_es_pipeline,
    }


def get_pipeline(name: str = 'es_pipeline') -> Pipeline:
    """Get pipeline by name.

    Args:
        name: Pipeline name (default: 'es_pipeline')

    Returns:
        Configured Pipeline instance

    Raises:
        ValueError: If pipeline doesn't exist
    """
    if not _PIPELINES:
        _register_pipelines()

    if name not in _PIPELINES:
        available = list(_PIPELINES.keys())
        raise ValueError(
            f"No pipeline named: {name}. "
            f"Available: {available}"
        )

    return _PIPELINES[name]()


def list_available_pipelines() -> List[str]:
    """List all available pipeline names."""
    if not _PIPELINES:
        _register_pipelines()
    return list(_PIPELINES.keys())
