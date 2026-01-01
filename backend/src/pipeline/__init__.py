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

# Pipeline access
from src.pipeline.pipelines import (
    get_pipeline, 
    list_available_pipelines,
    build_bronze_to_silver_pipeline,
    build_silver_to_gold_pipeline
)

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
