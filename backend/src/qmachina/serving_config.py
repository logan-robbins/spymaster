"""Serving configuration models for experiment and live runtime.

This module defines two distinct artifacts:

1. ``ServingSpec`` (authoring-time):
   - Stored in ``configs/serving``.
   - Referenced by ``ExperimentSpec``.
   - Points to a ``PipelineSpec`` by name.

2. ``PublishedServingSpec`` (runtime-time, immutable):
   - Stored in ``configs/serving_versions``.
   - Fully resolved for stream parity via a single runtime snapshot.
   - Referenced by serving alias/ID from the serving registry.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

from ..shared.yaml_io import load_yaml_mapping

logger: logging.Logger = logging.getLogger(__name__)

_DERIVATIVE_PARAM_TO_RUNTIME_KEY: dict[str, str] = {
    "center_exclusion_radius": "state_model_center_exclusion_radius",
    "spatial_decay_power": "state_model_spatial_decay_power",
    "zscore_window_bins": "state_model_zscore_window_bins",
    "zscore_min_periods": "state_model_zscore_min_periods",
    "tanh_scale": "state_model_tanh_scale",
    "d1_weight": "state_model_d1_weight",
    "d2_weight": "state_model_d2_weight",
    "d3_weight": "state_model_d3_weight",
}

_DERIVATIVE_WEIGHT_TO_RUNTIME_KEY: dict[str, str] = {
    "bull_pressure": "state_model_bull_pressure_weight",
    "bull_vacuum": "state_model_bull_vacuum_weight",
    "bear_pressure": "state_model_bear_pressure_weight",
    "bear_vacuum": "state_model_bear_vacuum_weight",
    "mixed": "state_model_mixed_weight",
}


class ScoringConfig(BaseModel):
    """Parameters controlling spectrum z-score and derivative scoring."""

    zscore_window_bins: int = 300
    zscore_min_periods: int = 75
    derivative_weights: list[float] = Field(default=[0.55, 0.30, 0.15])
    tanh_scale: float = 3.0
    # NOTE: YAML configs using `threshold_neutral:` must be updated to `neutral_threshold:`.
    neutral_threshold: float = 0.15


class SignalConfig(BaseModel):
    """Named signal definition with arbitrary params and component weights."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)


class ProjectionConfig(BaseModel):
    """Forward-projection horizon and interpolation parameters."""

    horizons_ms: list[int] = Field(default=[250, 500, 1000, 2500])
    use_cubic: bool = False
    cubic_scale: float = 1.0 / 6.0
    damping_lambda: float = 0.0


class StreamFieldRole(str, Enum):
    GRID_INDEX   = "grid_index"
    FORCE        = "force"
    VELOCITY     = "velocity"
    ACCELERATION = "acceleration"
    JERK         = "jerk"
    SIGNAL       = "signal"
    INDICATOR    = "indicator"
    METADATA     = "metadata"


class StreamFieldSpec(BaseModel):
    name: str
    dtype: str  # Arrow type name: "int8", "int32", "int64", "float64"
    role: StreamFieldRole | None = None
    description: str = ""


class CellShaderConfig(BaseModel):
    """Cell color mapping configuration for heatmap display mode."""

    signal_field: str = "flow_score"
    depth_field: str = "rest_depth"
    color_scheme: Literal["pressure_vacuum", "thermal", "mono_green", "mono_red"] = "pressure_vacuum"
    normalization: Literal["adaptive_running_max", "zscore_clamp", "identity_clamp"] = "adaptive_running_max"
    gamma: float = Field(default=0.7, gt=0.0, le=5.0)


class OverlaySpec(BaseModel):
    """A single overlay layer declaration (compute or render)."""

    id: str
    type: str
    enabled: bool = True
    z_order: int = 0
    params: dict[str, Any] = Field(default_factory=dict)


class VisualizationConfig(BaseModel):
    """Top-level visualization configuration for a serving spec."""

    display_mode: Literal["heatmap", "candle"] = "heatmap"
    cell_shader: CellShaderConfig | None = None
    overlays: list[OverlaySpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "VisualizationConfig":
        if self.display_mode == "heatmap" and self.cell_shader is None:
            raise ValueError("cell_shader is required when display_mode='heatmap'")
        ids = [o.id for o in self.overlays]
        if len(ids) != len(set(ids)):
            raise ValueError("overlay IDs must be unique")
        zo = [o.z_order for o in self.overlays]
        if len(zo) != len(set(zo)):
            raise ValueError("overlay z_order values must be unique")
        for o in self.overlays:
            if o.type == "projection_bands" and "signal_field" not in o.params:
                raise ValueError(
                    f"overlay '{o.id}' of type 'projection_bands' requires params.signal_field"
                )
        return self

    @classmethod
    def default_heatmap(cls) -> "VisualizationConfig":
        """Return the canonical heatmap visualization config."""
        return cls(
            display_mode="heatmap",
            cell_shader=CellShaderConfig(),
            overlays=[
                OverlaySpec(id="vp_gold", type="vp_gold", enabled=True, z_order=0),
                OverlaySpec(id="spot_trail", type="spot_trail", enabled=True, z_order=10),
                OverlaySpec(id="bbo_spread", type="bbo_spread", enabled=True, z_order=15),
                OverlaySpec(id="bbo_trail", type="bbo_trail", enabled=True, z_order=20),
                OverlaySpec(
                    id="projection_bands",
                    type="projection_bands",
                    enabled=True,
                    z_order=30,
                    params={"signal_field": "flow_score", "composite_scale": 8.0},
                ),
            ],
        )


class ServingSpec(BaseModel):
    """Complete serving configuration for the qMachina model serving layer.

    A ServingSpec is the single artifact that parameterizes a live server.
    It names a pipeline (resolved at runtime), and carries scoring, signal,
    and projection configs as nested sub-models.
    """

    name: str
    description: str = ""
    pipeline: str
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    signal: SignalConfig | None = None
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    stream_schema: list[StreamFieldSpec] = Field(default_factory=list)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig.default_heatmap)

    @field_validator("stream_schema")
    @classmethod
    def _validate_stream_schema(cls, v: list) -> list:
        if not v:
            return v
        _VALID_DTYPES = {"int8", "int32", "int64", "float64"}
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("stream_schema field names must be unique")
        for f in v:
            if f.dtype not in _VALID_DTYPES:
                raise ValueError(f"stream_schema field '{f.name}' has invalid dtype '{f.dtype}'. Must be one of {_VALID_DTYPES}")
        if not any(f.name == "k" and f.dtype == "int32" and f.role == StreamFieldRole.GRID_INDEX for f in v):
            raise ValueError("stream_schema must contain field 'k' with dtype='int32' and role=grid_index")
        if not any(f.name == "last_event_id" and f.dtype == "int64" and f.role == StreamFieldRole.METADATA for f in v):
            raise ValueError("stream_schema must contain field 'last_event_id' with dtype='int64' and role=metadata")
        for f in v:
            if f.role == StreamFieldRole.VELOCITY and f.dtype != "float64":
                raise ValueError(f"VELOCITY field '{f.name}' must use dtype='float64'")
            if f.role == StreamFieldRole.INDICATOR and f.dtype not in ("int8", "int32"):
                raise ValueError(f"INDICATOR field '{f.name}' must use dtype 'int8' or 'int32'")
        return v

    # ------------------------------------------------------------------
    # Pipeline resolution
    # ------------------------------------------------------------------

    def resolve_pipeline(self, lake_root: Path) -> Any:
        """Resolve the named pipeline to a full PipelineSpec.

        Args:
            lake_root: Root path of the data lake (contains research/ tree).

        Returns:
            PipelineSpec loaded by name from the configs directory.

        Raises:
            FileNotFoundError: If the pipeline config YAML does not exist.
            ValueError: If the pipeline config fails validation.
        """
        from .pipeline_config import PipelineSpec

        return PipelineSpec.load_by_name(self.pipeline, lake_root)

    # ------------------------------------------------------------------
    # Runtime integration
    # ------------------------------------------------------------------

    def to_runtime_fields(
        self,
        *,
        cell_width_ms: int | None = None,
    ) -> dict[str, Any]:
        """Return runtime config fields resolved from serving config.

        The mapping is strict:
        - scoring fields always map to flow_* runtime keys
        - derivative signal params/weights map to state_model_* keys
        - projection horizons are converted from ms -> bins (requires cell_width_ms)

        Unknown derivative params/weights fail fast.
        """
        runtime_fields: dict[str, Any] = {
            "flow_derivative_weights": self.scoring.derivative_weights,
            "flow_tanh_scale": self.scoring.tanh_scale,
            "flow_neutral_threshold": self.scoring.neutral_threshold,
            "flow_zscore_window_bins": self.scoring.zscore_window_bins,
            "flow_zscore_min_periods": self.scoring.zscore_min_periods,
        }

        if self.signal is not None:
            signal_name = self.signal.name.strip().lower()
            if signal_name != "derivative":
                if self.signal.params or self.signal.weights:
                    raise ValueError(
                        "Only derivative signal params/weights are supported for "
                        f"runtime field mapping. Got signal={self.signal.name!r}."
                    )
            else:
                unknown_params = sorted(
                    key
                    for key in self.signal.params.keys()
                    if key not in _DERIVATIVE_PARAM_TO_RUNTIME_KEY
                )
                if unknown_params:
                    raise ValueError(
                        f"Unknown derivative signal params for runtime mapping: {unknown_params}"
                    )
                unknown_weights = sorted(
                    key
                    for key in self.signal.weights.keys()
                    if key not in _DERIVATIVE_WEIGHT_TO_RUNTIME_KEY
                )
                if unknown_weights:
                    raise ValueError(
                        f"Unknown derivative signal weights for runtime mapping: {unknown_weights}"
                    )

                for key, runtime_key in _DERIVATIVE_PARAM_TO_RUNTIME_KEY.items():
                    if key in self.signal.params:
                        runtime_fields[runtime_key] = self.signal.params[key]
                for key, runtime_key in _DERIVATIVE_WEIGHT_TO_RUNTIME_KEY.items():
                    if key in self.signal.weights:
                        runtime_fields[runtime_key] = self.signal.weights[key]

        if self.projection.horizons_ms:
            if cell_width_ms is None:
                raise ValueError(
                    "cell_width_ms is required to convert projection horizons "
                    "from ms to bins."
                )
            if cell_width_ms <= 0:
                raise ValueError("cell_width_ms must be > 0.")
            bins: list[int] = []
            for horizon_ms in self.projection.horizons_ms:
                if horizon_ms <= 0:
                    raise ValueError(
                        f"projection.horizons_ms must be positive, got {horizon_ms}"
                    )
                if horizon_ms % cell_width_ms != 0:
                    raise ValueError(
                        "projection.horizons_ms must be exact multiples of cell_width_ms. "
                        f"Got horizon_ms={horizon_ms}, cell_width_ms={cell_width_ms}."
                    )
                bins.append(horizon_ms // cell_width_ms)
            runtime_fields["projection_horizons_bins"] = bins

        return runtime_fields

    # ------------------------------------------------------------------
    # Persistence (YAML)
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> ServingSpec:
        """Load and validate a ServingSpec from a YAML file.

        Args:
            path: Absolute or relative path to the YAML file.

        Returns:
            Validated ServingSpec instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If the content fails model validation.
        """
        raw = load_yaml_mapping(
            Path(path),
            not_found_message="ServingSpec YAML not found: {path}",
            non_mapping_message=(
                "Expected a YAML mapping at top level, got {kind}: {path}"
            ),
        )
        logger.debug("Loading ServingSpec from %s", path)
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        """Write this ServingSpec to a YAML file.

        Creates parent directories if they do not exist.  Used by
        ``cli promote`` to persist a winning experiment configuration.

        Args:
            path: Destination file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = self.model_dump(mode="json")
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        logger.info("Wrote ServingSpec '%s' to %s", self.name, path)

    # ------------------------------------------------------------------
    # Directory conventions
    # ------------------------------------------------------------------

    @classmethod
    def configs_dir(cls, lake_root: Path) -> Path:
        """Return the canonical directory for serving config YAML files.

        Args:
            lake_root: Root path of the data lake.

        Returns:
            ``lake_root / "research" / "harness" / "configs" / "serving"``
        """
        return lake_root / "research" / "harness" / "configs" / "serving"

    @classmethod
    def load_by_name(cls, name: str, lake_root: Path) -> ServingSpec:
        """Load a ServingSpec by name from the canonical configs directory.

        Args:
            name: The spec name (used as the YAML filename stem).
            lake_root: Root path of the data lake.

        Returns:
            Validated ServingSpec instance.

        Raises:
            FileNotFoundError: If no YAML file exists for *name*.
        """
        path: Path = cls.configs_dir(lake_root) / f"{name}.yaml"
        return cls.from_yaml(path)


class PublishedServingSource(BaseModel):
    """Source provenance for a published serving version."""

    run_id: str
    experiment_name: str
    config_hash: str
    promoted_at_utc: str
    serving_spec_name: str | None = None
    signal_name: str | None = None


class PublishedServingSpec(BaseModel):
    """Immutable serving version used directly by live streaming.

    Runtime behavior is sourced from ``runtime_snapshot`` only.
    """

    serving_id: str
    description: str = ""
    runtime_snapshot: dict[str, Any]
    source: PublishedServingSource

    def stream_dt(self) -> str:
        raw = self._require_snapshot_key("stream_dt")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Published serving stream_dt must be a non-empty string.")
        return raw.strip()

    def stream_start_time(self) -> str:
        raw = self._require_snapshot_key("stream_start_time")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                "Published serving stream_start_time must be a non-empty string."
            )
        return raw.strip()

    def to_runtime_config_json(self, *, serving_name: str) -> dict[str, Any]:
        """Return serving payload attached to runtime_config stream message."""
        return {
            "serving_name": serving_name,
            "serving_id": self.serving_id,
            "source": self.source.model_dump(),
        }

    def _require_snapshot_key(self, key: str) -> Any:
        if key not in self.runtime_snapshot:
            raise ValueError(
                f"Published serving runtime snapshot is missing required key: {key}"
            )
        return self.runtime_snapshot[key]

    @classmethod
    def from_yaml(cls, path: Path) -> PublishedServingSpec:
        """Load and validate a published serving spec from YAML."""
        raw = load_yaml_mapping(
            Path(path),
            not_found_message="Published serving YAML not found: {path}",
            non_mapping_message=(
                "Expected a YAML mapping at top level, got {kind}: {path}"
            ),
        )
        logger.debug("Loading PublishedServingSpec from %s", path)
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        """Write immutable published serving YAML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = self.model_dump(mode="json")
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        logger.info("Wrote PublishedServingSpec '%s' to %s", self.serving_id, path)

    @classmethod
    def configs_dir(cls, lake_root: Path) -> Path:
        """Canonical directory for immutable serving version specs."""
        return lake_root / "research" / "harness" / "configs" / "serving_versions"
