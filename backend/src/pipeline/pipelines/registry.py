"""Pipeline registry.

Maps pipeline names to pipeline builders.
"""
from typing import Callable, Dict, List, Set

from src.pipeline.core.pipeline import Pipeline


# Registry of pipeline name -> pipeline builder
_PIPELINES: Dict[str, Callable[[], Pipeline]] = {}

# Pipelines that require a level parameter
LEVEL_REQUIRED_PIPELINES: Set[str] = {'bronze_to_silver', 'silver_to_gold', 'pentaview'}

# Pipelines that are global (no level needed)
GLOBAL_PIPELINES: Set[str] = {'bronze_to_silver_global'}


def _register_pipelines():
    """Register all available pipelines."""
    global _PIPELINES

    # Import here to avoid circular imports
    from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline
    from src.pipeline.pipelines.silver_to_gold import build_silver_to_gold_pipeline
    from src.pipeline.pipelines.pentaview_pipeline import build_pentaview_pipeline
    from src.pipeline.pipelines.bronze_to_silver_global import BronzeToSilverGlobalPipeline

    _PIPELINES = {
        'bronze_to_silver': build_bronze_to_silver_pipeline,
        'silver_to_gold': build_silver_to_gold_pipeline,
        'pentaview': build_pentaview_pipeline,
        'bronze_to_silver_global': lambda: BronzeToSilverGlobalPipeline(),
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
