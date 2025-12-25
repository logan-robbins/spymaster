"""Versioned pipeline definitions."""
from src.pipeline.pipelines.registry import get_pipeline_for_version, list_available_versions
from src.pipeline.pipelines.v1_0_mechanics_only import build_v1_0_pipeline
from src.pipeline.pipelines.v2_0_full_ensemble import build_v2_0_pipeline

__all__ = [
    "get_pipeline_for_version",
    "list_available_versions",
    "build_v1_0_pipeline",
    "build_v2_0_pipeline",
]
