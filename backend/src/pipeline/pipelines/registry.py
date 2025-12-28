"""Pipeline registry - get pipelines by version string.

Maps Silver version strings to pipeline builders.
"""
from typing import Callable, Dict, List

from src.pipeline.core.pipeline import Pipeline


# Registry of version prefix -> pipeline builder
_PIPELINES: Dict[str, Callable[[], Pipeline]] = {}


def _register_pipelines():
    """Register all available pipelines."""
    global _PIPELINES

    # Import here to avoid circular imports
    from src.pipeline.pipelines.v1_0_mechanics_only import build_v1_0_pipeline
    from src.pipeline.pipelines.v2_0_full_ensemble import build_v2_0_pipeline
    from src.pipeline.pipelines.v1_0_spx_final_call import build_v1_0_spx_final_call_pipeline

    _PIPELINES = {
        'v1.0': build_v1_0_pipeline,
        'v1.0_spx': build_v1_0_spx_final_call_pipeline,  # Final Call v1 spec
        'v2.0': build_v2_0_pipeline,
    }


def get_pipeline_for_version(version: str) -> Pipeline:
    """Get pipeline for a Silver version.

    Args:
        version: Version string like "v1.0_mechanics_only" or "v2.0"

    Returns:
        Configured Pipeline instance

    Raises:
        ValueError: If no pipeline exists for the version
    """
    if not _PIPELINES:
        _register_pipelines()

    # Extract version prefix (v1.0_mechanics_only -> v1.0)
    version_prefix = version.split('_')[0]

    if version_prefix not in _PIPELINES:
        available = list(_PIPELINES.keys())
        raise ValueError(
            f"No pipeline for version: {version}. "
            f"Available: {available}"
        )

    return _PIPELINES[version_prefix]()


def list_available_versions() -> List[str]:
    """List all available pipeline versions."""
    if not _PIPELINES:
        _register_pipelines()
    return list(_PIPELINES.keys())
