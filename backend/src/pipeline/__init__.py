"""
Pipeline - Modular Feature Engineering Architecture

Stage-based pipeline for transforming Bronze data into Silver features.
Supports versioned pipelines with different stage compositions.

Architecture:
- core/: Base abstractions (BaseStage, Pipeline)
- stages/: Individual transformation stages
- pipelines/: Versioned pipeline definitions
- utils/: Shared utilities (DuckDB reader, vectorized ops)

Usage:
    from src.pipeline import get_pipeline_for_version

    # Get pipeline for Silver version
    pipeline = get_pipeline_for_version("v1.0_mechanics_only")
    signals_df = pipeline.run("2025-12-16")

    # Or use specific pipeline builders
    from src.pipeline.pipelines import build_v1_0_pipeline, build_v2_0_pipeline

    pipeline = build_v2_0_pipeline()
    signals_df = pipeline.run("2025-12-16")
"""
# Core abstractions
from src.pipeline.core import BaseStage, StageContext, Pipeline

# Versioned pipeline access
from src.pipeline.pipelines import get_pipeline_for_version, list_available_versions

__all__ = [
    # Core
    "BaseStage",
    "StageContext",
    "Pipeline",
    # Registry
    "get_pipeline_for_version",
    "list_available_versions",
]

