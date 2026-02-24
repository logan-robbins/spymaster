"""PipelineSpec: three-layer configuration model for VP feature engineering.

Captures the full parameterization of the raw .dbn -> engineered feature grid
pipeline. Layers:

    1. CaptureConfig   -- what data to ingest (product, symbol, date, time window)
    2. PipelineOverrides -- optional engine/spectrum/grid coefficient overrides
    3. PipelineSpec     -- named experiment wrapping capture + overrides

Scoring, z-score, and projection parameters belong in ServingSpec (not here).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from ..shared.yaml_io import load_yaml_mapping
from .config import RuntimeConfig, build_config_with_overrides, resolve_config

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class CaptureConfig(BaseModel):
    """Data capture coordinates: what instrument, which session window."""

    product_type: str = "future_mbo"
    symbol: str
    dt: str
    start_time: str
    end_time: str

    @field_validator("dt")
    @classmethod
    def _validate_dt(cls, v: str) -> str:
        if not _DATE_RE.match(v):
            raise ValueError(f"dt must be YYYY-MM-DD format, got {v!r}")
        return v

    @field_validator("start_time", "end_time")
    @classmethod
    def _validate_time(cls, v: str) -> str:
        if not _TIME_RE.match(v):
            raise ValueError(f"time must be HH:MM format, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_time_order(self) -> CaptureConfig:
        if self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) must be before "
                f"end_time ({self.end_time})"
            )
        return self


class PipelineOverrides(BaseModel):
    """Optional engine coefficient overrides.

    Every field defaults to None, meaning the value is inherited from
    instrument.yaml at resolve time. Only non-None fields are applied
    as overrides to the base RuntimeConfig.
    """

    cell_width_ms: int | None = None
    grid_radius_ticks: int | None = None
    n_absolute_ticks: int | None = None
    bucket_size_dollars: float | None = None
    tau_velocity: float | None = None
    tau_acceleration: float | None = None
    tau_jerk: float | None = None
    tau_rest_decay: float | None = None
    c1_v_add: float | None = None
    c2_v_rest_pos: float | None = None
    c3_a_add: float | None = None
    c4_v_pull: float | None = None
    c5_v_fill: float | None = None
    c6_v_rest_neg: float | None = None
    c7_a_pull: float | None = None
    flow_windows: list[int] | None = None
    flow_rollup_weights: list[float] | None = None

    def non_none_dict(self) -> dict[str, Any]:
        """Return only fields with explicit (non-None) values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class PipelineSpec(BaseModel):
    """Named experiment specification for the VP feature engineering pipeline.

    Wraps a CaptureConfig (what data) and PipelineOverrides (how to process)
    with a human-readable name and version tag. Provides deterministic
    dataset_id generation and runtime config resolution.
    """

    name: str
    description: str = ""
    pipeline_code_version: int = 1
    capture: CaptureConfig
    pipeline: PipelineOverrides = PipelineOverrides()

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                f"name must be lowercase alphanumeric with underscores "
                f"(no leading underscore), got {v!r}"
            )
        return v

    # ------------------------------------------------------------------
    # Runtime config resolution
    # ------------------------------------------------------------------

    def resolve_runtime_config(
        self,
    ) -> RuntimeConfig:
        """Resolve full RuntimeConfig from instrument.yaml + overrides.

        Returns:
            Fully validated RuntimeConfig with pipeline overrides applied.
        """
        base_cfg = resolve_config(
            self.capture.product_type,
            self.capture.symbol,
        )
        overrides = self.pipeline.non_none_dict()
        if not overrides:
            logger.debug("No pipeline overrides; using base config as-is.")
            return base_cfg
        logger.debug("Applying %d pipeline overrides: %s", len(overrides), sorted(overrides))
        return build_config_with_overrides(base_cfg, overrides)

    # ------------------------------------------------------------------
    # Deterministic dataset ID
    # ------------------------------------------------------------------

    def dataset_id(self) -> str:
        """Compute a deterministic, human-readable dataset identifier.

        Format:
            {symbol_lower}_{dt_nodash}_{start_nodash}_{end_nodash}__{name}__{hash8}

        The hash covers pipeline_code_version, capture, and non-None
        pipeline override fields, ensuring any parameter change produces
        a different ID.

        Returns:
            Deterministic dataset ID string.
        """
        hash_payload: dict[str, Any] = {
            "pipeline_code_version": self.pipeline_code_version,
            "capture": self.capture.model_dump(),
            "pipeline": self.pipeline.non_none_dict(),
        }
        canonical_json = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
        hash8 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:8]

        symbol_lower = self.capture.symbol.lower()
        dt_nodash = self.capture.dt.replace("-", "")
        start_nodash = self.capture.start_time.replace(":", "")
        end_nodash = self.capture.end_time.replace(":", "")

        return f"{symbol_lower}_{dt_nodash}_{start_nodash}_{end_nodash}__{self.name}__{hash8}"

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineSpec:
        """Load and validate a PipelineSpec from a YAML file.

        Args:
            path: Path to a YAML file containing PipelineSpec fields.

        Returns:
            Validated PipelineSpec instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content is not a valid mapping.
            pydantic.ValidationError: If fields fail validation.
        """
        raw = load_yaml_mapping(
            Path(path),
            not_found_message="PipelineSpec YAML not found: {path}",
            non_mapping_message="PipelineSpec YAML must be a mapping, got {kind}: {path}",
            resolve_path=True,
        )
        return cls.model_validate(raw)

    @classmethod
    def configs_dir(cls, lake_root: Path) -> Path:
        """Canonical directory for pipeline config YAML files.

        Args:
            lake_root: Root of the data lake filesystem.

        Returns:
            Path to ``lake_root/research/harness/configs/pipelines``.
        """
        return lake_root / "research" / "harness" / "configs" / "pipelines"

    @classmethod
    def load_by_name(cls, name: str, lake_root: Path) -> PipelineSpec:
        """Load a named PipelineSpec from the canonical configs directory.

        Args:
            name: Config name (filename stem, without .yaml extension).
            lake_root: Root of the data lake filesystem.

        Returns:
            Validated PipelineSpec instance.

        Raises:
            FileNotFoundError: If the named config file does not exist.
        """
        path = cls.configs_dir(lake_root) / f"{name}.yaml"
        return cls.from_yaml(path)
