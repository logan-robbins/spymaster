"""Pipeline constructors - LAZY IMPORTS to prevent multiprocessing deadlocks."""


def build_bronze_to_silver_pipeline():
    """Build Bronze→Silver pipeline (lazy import)."""
    from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline as _build
    return _build()


def build_silver_to_gold_pipeline():
    """Build Silver→Gold pipeline (lazy import)."""
    from src.pipeline.pipelines.silver_to_gold import build_silver_to_gold_pipeline as _build
    return _build()


def build_pentaview_pipeline():
    """Build Pentaview pipeline (lazy import)."""
    from src.pipeline.pipelines.pentaview_pipeline import build_pentaview_pipeline as _build
    return _build()


def get_pipeline(name: str = 'bronze_to_silver'):
    """Get pipeline by name (lazy import)."""
    from src.pipeline.pipelines.registry import get_pipeline as _get
    return _get(name)


def list_available_pipelines():
    """List available pipelines (lazy import)."""
    from src.pipeline.pipelines.registry import list_available_pipelines as _list
    return _list()


__all__ = [
    'build_bronze_to_silver_pipeline',
    'build_silver_to_gold_pipeline',
    'build_pentaview_pipeline',
    'get_pipeline',
    'list_available_pipelines'
]
