"""ServingSpec: single-point parameterization of the VP modeling layer.

ServingSpec is what the server loads directly. It references a pipeline by
name and defines scoring (z-score window, derivative blend weights, tanh
scale, neutral threshold), signal (name, params, weights), and projection
(horizons, cubic, damping) configuration.

No sweep axes, no eval metrics, no experiment tracking.  ``cli promote``
writes a ServingSpec from a winning experiment run.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)


class ScoringConfig(BaseModel):
    """Parameters controlling spectrum z-score and derivative scoring."""

    zscore_window_bins: int = 300
    zscore_min_periods: int = 75
    derivative_weights: list[float] = Field(default=[0.55, 0.30, 0.15])
    tanh_scale: float = 3.0
    threshold_neutral: float = 0.15


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


class ServingSpec(BaseModel):
    """Complete serving configuration for the VP modeling layer.

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

    def to_runtime_overrides(self) -> dict[str, Any]:
        """Return a dict of overrides for ``build_config_with_overrides``.

        Maps ServingSpec scoring fields to the runtime config keys consumed
        by the existing ``VPRuntimeConfig`` / ``build_config_with_overrides``
        machinery.
        """
        return {
            "flow_derivative_weights": self.scoring.derivative_weights,
            "flow_tanh_scale": self.scoring.tanh_scale,
            "flow_neutral_threshold": self.scoring.threshold_neutral,
            "flow_zscore_window_bins": self.scoring.zscore_window_bins,
            "flow_zscore_min_periods": self.scoring.zscore_min_periods,
        }

    def to_runtime_config_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for the WebSocket runtime_config message.

        This is the payload shape the frontend expects when it receives
        the current serving configuration over the wire.
        """
        return {
            "serving_name": self.name,
            "scoring": self.scoring.model_dump(),
            "signal_config": self.signal.model_dump() if self.signal else None,
            "projection": self.projection.model_dump(),
        }

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
        if not path.exists():
            raise FileNotFoundError(f"ServingSpec YAML not found: {path}")

        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected a YAML mapping at top level, got {type(raw).__name__}: {path}"
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
            ``lake_root / "research" / "vp_harness" / "configs" / "serving"``
        """
        return lake_root / "research" / "vp_harness" / "configs" / "serving"

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
