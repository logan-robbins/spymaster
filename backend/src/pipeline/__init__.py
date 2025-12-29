"""
Pipeline - Feature Engineering Architecture

Stage-based pipeline for transforming Bronze data into Silver features.

Architecture:
- core/: Base abstractions (BaseStage, Pipeline)
- stages/: Individual transformation stages
- pipelines/: Pipeline definitions
- utils/: Shared utilities (DuckDB reader, vectorized ops)

Usage:
    from src.pipeline import get_pipeline, build_es_pipeline

    # Get pipeline from registry
    pipeline = get_pipeline('es_pipeline')
    signals_df = pipeline.run("2025-12-16")

    # Or use builder directly
    from src.pipeline.pipelines import build_es_pipeline
    
    pipeline = build_es_pipeline()
    signals_df = pipeline.run("2025-12-16")
"""
# Core abstractions
from src.pipeline.core import BaseStage, StageContext, Pipeline

# Pipeline access
from src.pipeline.pipelines import get_pipeline, list_available_pipelines, build_es_pipeline

__all__ = [
    # Core
    "BaseStage",
    "StageContext",
    "Pipeline",
    # Registry
    "get_pipeline",
    "list_available_pipelines",
    # Builders
    "build_es_pipeline",
]
