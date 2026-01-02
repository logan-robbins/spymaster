"""
Pipeline - Feature Engineering Architecture

Modular pipelines for Bronze→Silver→Gold data transformations.

Architecture:
- core/: Base abstractions (BaseStage, Pipeline)
- stages/: Individual transformation stages
- pipelines/: Pipeline definitions (bronze_to_silver, silver_to_gold)
- utils/: Shared utilities (DuckDB reader, vectorized ops)

Usage:
    from src.pipeline import get_pipeline

    # Bronze → Silver (feature engineering)
    pipeline = get_pipeline('bronze_to_silver')
    signals_df = pipeline.run("2025-12-16", write_outputs=True)
    
    # Silver → Gold (episode construction)
    pipeline = get_pipeline('silver_to_gold')
    episodes = pipeline.run("2025-12-16", write_outputs=True)
"""
# Core abstractions
from src.pipeline.core import BaseStage, StageContext, Pipeline

# Pipeline access - LAZY IMPORTS to avoid module-level import cascade
# This prevents deadlocks in multiprocessing spawn mode on macOS


def get_pipeline(name: str = 'bronze_to_silver'):
    """
    Get pipeline by name (lazy import).
    
    Args:
        name: Pipeline name ('bronze_to_silver', 'silver_to_gold', 'pentaview')
    
    Returns:
        Configured Pipeline instance
    """
    from src.pipeline.pipelines.registry import get_pipeline as _get_pipeline
    return _get_pipeline(name)


def list_available_pipelines():
    """List all available pipeline names (lazy import)."""
    from src.pipeline.pipelines.registry import list_available_pipelines as _list
    return _list()


def build_bronze_to_silver_pipeline():
    """Build Bronze→Silver pipeline (lazy import)."""
    from src.pipeline.pipelines.bronze_to_silver import build_bronze_to_silver_pipeline as _build
    return _build()


def build_silver_to_gold_pipeline():
    """Build Silver→Gold pipeline (lazy import)."""
    from src.pipeline.pipelines.silver_to_gold import build_silver_to_gold_pipeline as _build
    return _build()


__all__ = [
    # Core
    "BaseStage",
    "StageContext",
    "Pipeline",
    # Registry
    "get_pipeline",
    "list_available_pipelines",
    # Builders
    "build_bronze_to_silver_pipeline",
    "build_silver_to_gold_pipeline",
]
