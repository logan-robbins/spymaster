"""Gold DSL v1 schema: Pydantic node types and top-level spec model.

Node types form a directed acyclic graph (DAG) that declares gold feature
computation as composable transformations of silver-stage fields.

Each node references upstream dependencies by name. The graph is validated
for acyclicity and resolved in topological order during preview execution.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class SilverRef(BaseModel, extra="forbid"):
    """Source reference to a silver-stage column.

    Attributes:
        type: Discriminator literal.
        field: Column name that must exist in SILVER_COLS.
    """

    type: Literal["silver_ref"]
    field: str


class TemporalWindow(BaseModel, extra="forbid"):
    """Temporal window aggregation over a source node's time series.

    Groups by relative tick (k), then applies a rolling or exponentially
    weighted aggregation across bins.

    Attributes:
        type: Discriminator literal.
        source: Name of the upstream node.
        window_bins: Rolling window size in bins. Must be >= 1.
        agg: Aggregation function. ``ewm`` requires ``alpha``.
        alpha: Exponential smoothing factor in (0, 1). Required for ewm.
    """

    type: Literal["temporal_window"]
    source: str
    window_bins: int
    agg: Literal["mean", "std", "min", "max", "sum", "ewm"]
    alpha: float | None = None

    @field_validator("window_bins")
    @classmethod
    def _window_bins_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"window_bins must be >= 1, got {v}")
        return v


class SpatialNeighborhood(BaseModel, extra="forbid"):
    """Spatial neighborhood aggregation across relative ticks (k axis).

    For each bin, pivots to a matrix of (bin_seq x k), applies a rolling
    window of ``radius_ticks`` along the k axis, then melts back.

    Attributes:
        type: Discriminator literal.
        source: Name of the upstream node.
        radius_ticks: Spatial radius in ticks. Must be >= 1.
        agg: Aggregation function applied over the k neighborhood.
    """

    type: Literal["spatial_neighborhood"]
    source: str
    radius_ticks: int
    agg: Literal["mean", "std", "sum", "max"]

    @field_validator("radius_ticks")
    @classmethod
    def _radius_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"radius_ticks must be >= 1, got {v}")
        return v


class ArithmeticExpr(BaseModel, extra="forbid"):
    """Binary arithmetic expression between two nodes.

    Attributes:
        type: Discriminator literal.
        op: Binary operator. ``abs_diff`` computes ``|left - right|``.
        left: Name of the left operand node.
        right: Name of the right operand node.
    """

    type: Literal["arithmetic"]
    op: Literal["add", "sub", "mul", "div", "abs_diff"]
    left: str
    right: str


class NormExpr(BaseModel, extra="forbid"):
    """Normalization transformation of a source node.

    Attributes:
        type: Discriminator literal.
        source: Name of the upstream node.
        method: Normalization method. ``zscore`` requires ``window_bins``.
        window_bins: Rolling window for z-score computation. Required
            when method is ``zscore``.
    """

    type: Literal["norm"]
    source: str
    method: Literal["zscore", "minmax", "log1p", "tanh"]
    window_bins: int | None = None


class OutputNode(BaseModel, extra="forbid"):
    """Named output terminal node.

    Each OutputNode declares a named feature that will appear in the
    final gold feature set.

    Attributes:
        type: Discriminator literal.
        source: Name of the upstream node providing the data.
        name: Output feature name. Must be unique and snake_case.
        dtype: Target data type for the output column.
    """

    type: Literal["output"]
    source: str
    name: str
    dtype: Literal["float32", "float64"] = "float32"

    @field_validator("name")
    @classmethod
    def _validate_snake_case(cls, v: str) -> str:
        if not _SNAKE_CASE_RE.match(v):
            raise ValueError(
                f"output name must be snake_case (lowercase, starting with letter), "
                f"got {v!r}"
            )
        return v


DslNode = Union[
    SilverRef,
    TemporalWindow,
    SpatialNeighborhood,
    ArithmeticExpr,
    NormExpr,
    OutputNode,
]


class GoldDslSpec(BaseModel):
    """Top-level Gold DSL specification.

    Contains a versioned dictionary of named nodes forming a DAG.
    The spec hash provides deterministic content addressing for
    lineage tracking and cache invalidation.

    Attributes:
        version: Immutable schema version. Currently always 1.
        nodes: Mapping from node name to node definition.
    """

    version: int = 1
    nodes: dict[str, DslNode] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _version_must_be_one(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Only DSL version 1 is supported, got {v}")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> GoldDslSpec:
        """Parse a raw dictionary into a validated GoldDslSpec.

        Each node in the ``nodes`` mapping is discriminated by its
        ``type`` field and deserialized into the corresponding
        Pydantic model.

        Args:
            data: Raw dictionary, typically from JSON or YAML.

        Returns:
            Validated GoldDslSpec instance.

        Raises:
            pydantic.ValidationError: If any node fails validation.
        """
        return cls.model_validate(data)

    def spec_hash(self) -> str:
        """Compute a deterministic SHA-256 hex digest of the spec content.

        The hash is computed over a canonical JSON representation with
        sorted keys and minimal separators, ensuring identical specs
        always produce the same hash regardless of field insertion order.

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
