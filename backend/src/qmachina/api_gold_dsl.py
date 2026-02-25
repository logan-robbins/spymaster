"""Gold DSL API endpoints for validation, preview, lineage, and legacy compat.

Provides REST endpoints for working with Gold DSL specs:
    - POST /v1/gold/validate: validate a DSL spec
    - POST /v1/gold/preview: execute a preview against sample data
    - POST /v1/gold/lineage/compare: compare two spec hashes
    - GET  /v1/gold/from_legacy: convert VP gold config to DSL
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..gold_dsl.compat import gold_config_to_dsl
from ..gold_dsl.preview import execute_dsl_preview
from ..gold_dsl.schema import GoldDslSpec
from ..gold_dsl.validate import validate_dsl
from .gold_config import GoldFeatureConfig

logger: logging.Logger = logging.getLogger(__name__)

router = APIRouter(tags=["gold-dsl"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    """Request body for DSL validation."""

    spec: dict[str, Any]


class ValidateResponse(BaseModel):
    """Response for DSL validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)
    spec_hash: str | None = None


class PreviewRequest(BaseModel):
    """Request body for DSL preview execution."""

    spec: dict[str, Any]
    bins_parquet_path: str
    grid_parquet_path: str
    sample_bins: int = 100


class PreviewResponse(BaseModel):
    """Response for DSL preview execution."""

    output_stats: dict[str, dict[str, Any]]
    n_bins_sampled: int
    execution_time_ms: float
    spec_hash: str


class LineageCompareRequest(BaseModel):
    """Request body for lineage comparison."""

    spec_a: dict[str, Any]
    spec_b: dict[str, Any]


class LineageCompareResponse(BaseModel):
    """Response for lineage comparison."""

    hash_a: str
    hash_b: str
    hashes_match: bool
    added_nodes: list[str]
    removed_nodes: list[str]
    modified_nodes: list[str]


class LegacyConvertResponse(BaseModel):
    """Response for legacy VP config to DSL conversion."""

    spec: dict[str, Any]
    spec_hash: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/gold/validate", response_model=ValidateResponse)
async def validate_gold_dsl(request: ValidateRequest) -> ValidateResponse:
    """Validate a Gold DSL spec.

    Parses the raw dict into a GoldDslSpec and runs full structural
    and semantic validation.

    Returns:
        ValidateResponse with valid flag, errors list, and spec hash
        if the spec parsed successfully.
    """
    try:
        spec = GoldDslSpec.from_dict(request.spec)
    except Exception as exc:
        return ValidateResponse(
            valid=False,
            errors=[f"Parse error: {exc}"],
        )

    errors = validate_dsl(spec)
    return ValidateResponse(
        valid=len(errors) == 0,
        errors=errors,
        spec_hash=spec.spec_hash() if not errors else None,
    )


@router.post("/v1/gold/preview", response_model=PreviewResponse)
async def preview_gold_dsl(request: PreviewRequest) -> PreviewResponse:
    """Execute a DSL spec against sample parquet data.

    Validates the spec, then runs it against the provided parquet files
    to produce summary statistics for each output feature.

    Raises:
        HTTPException: 400 if spec is invalid, 404 if files not found.
    """
    try:
        spec = GoldDslSpec.from_dict(request.spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Parse error: {exc}") from exc

    errors = validate_dsl(spec)
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"DSL validation failed: {'; '.join(errors)}",
        )

    bins_path = Path(request.bins_parquet_path)
    grid_path = Path(request.grid_parquet_path)

    try:
        result = execute_dsl_preview(
            spec,
            bins_path,
            grid_path,
            sample_bins=request.sample_bins,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PreviewResponse(**result)


@router.post("/v1/gold/lineage/compare", response_model=LineageCompareResponse)
async def compare_lineage(request: LineageCompareRequest) -> LineageCompareResponse:
    """Compare two DSL specs by their node structure.

    Parses both specs and reports added, removed, and modified nodes
    based on node name and content comparison.

    Raises:
        HTTPException: 400 if either spec fails to parse.
    """
    try:
        spec_a = GoldDslSpec.from_dict(request.spec_a)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"spec_a parse error: {exc}"
        ) from exc

    try:
        spec_b = GoldDslSpec.from_dict(request.spec_b)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"spec_b parse error: {exc}"
        ) from exc

    hash_a = spec_a.spec_hash()
    hash_b = spec_b.spec_hash()

    names_a = set(spec_a.nodes.keys())
    names_b = set(spec_b.nodes.keys())

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)

    modified: list[str] = []
    for name in sorted(names_a & names_b):
        node_a = spec_a.nodes[name]
        node_b = spec_b.nodes[name]
        if node_a.model_dump() != node_b.model_dump():
            modified.append(name)

    return LineageCompareResponse(
        hash_a=hash_a,
        hash_b=hash_b,
        hashes_match=hash_a == hash_b,
        added_nodes=added,
        removed_nodes=removed,
        modified_nodes=modified,
    )


@router.get("/v1/gold/from_legacy", response_model=LegacyConvertResponse)
async def convert_from_legacy() -> LegacyConvertResponse:
    """Convert the default VP GoldFeatureConfig to a DSL spec.

    Uses default GoldFeatureConfig values (matching the VP baseline)
    and returns the equivalent DSL representation.
    """
    config = GoldFeatureConfig(
        c1_v_add=1.0,
        c2_v_rest_pos=0.5,
        c3_a_add=0.3,
        c4_v_pull=1.0,
        c5_v_fill=1.5,
        c6_v_rest_neg=0.5,
        c7_a_pull=0.3,
        flow_windows=[10, 30, 60],
        flow_rollup_weights=[0.5, 0.3, 0.2],
    )

    dsl_spec = gold_config_to_dsl(config)
    return LegacyConvertResponse(
        spec=dsl_spec.model_dump(mode="json"),
        spec_hash=dsl_spec.spec_hash(),
    )
