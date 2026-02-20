"""Pydantic v2 models for declarative YAML experiment configurations.

Defines the complete schema for experiment configs including grid variants,
evaluation parameters, sweep configurations, parallelism, and online simulation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class GridVariantConfig(BaseModel):
    """Grid-dependent parameters -- changing these regenerates the grid from raw .dbn.

    Each field accepts either a single value (fixed) or a list of values (sweep axis).
    """

    cell_width_ms: int | list[int] = 100
    c1_v_add: float | list[float] = 1.0
    c2_v_rest_pos: float | list[float] = 0.5
    c3_a_add: float | list[float] = 0.3
    c4_v_pull: float | list[float] = 1.0
    c5_v_fill: float | list[float] = 1.5
    c6_v_rest_neg: float | list[float] = 0.5
    c7_a_pull: float | list[float] = 0.3
    bucket_size_dollars: float | list[float] | None = None
    spectrum_windows: list[int] | None = None
    spectrum_rollup_weights: list[float] | None = None
    spectrum_derivative_weights: list[float] | None = None
    spectrum_tanh_scale: float | None = None
    tau_velocity: float | list[float] | None = None
    tau_acceleration: float | list[float] | None = None
    tau_jerk: float | list[float] | None = None
    tau_rest_decay: float | list[float] | None = None
    product_type: str = "future_mbo"
    symbol: str = "MNQH6"
    dt: str = "2026-02-06"
    start_time: str = "09:25"


class EvalConfig(BaseModel):
    """TP/SL evaluation parameters.

    Fields accepting list[int] enable threshold sweeps across TP/SL values.
    """

    tp_ticks: int | list[int] = 8
    sl_ticks: int | list[int] = 4
    max_hold_bins: int = 1200
    warmup_bins: int = 300
    tick_size: float = 0.25
    min_signals: int = 5

    @field_validator("min_signals")
    @classmethod
    def _min_signals_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_signals must be >= 1")
        return v


class SweepConfig(BaseModel):
    """Parameter sweep configuration.

    Attributes:
        universal: Sweep axes applied to all signals.
            Keys are parameter names, values are lists of values to sweep.
        per_signal: Sweep axes specific to individual signals.
            Outer key is signal name, inner dict has same structure as universal.
    """

    universal: dict[str, list[Any]] = Field(default_factory=dict)
    per_signal: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)


class ParallelConfig(BaseModel):
    """Parallelism configuration for experiment execution.

    Attributes:
        max_workers: Maximum number of concurrent workers.
        timeout_seconds: Per-worker timeout in seconds.
    """

    max_workers: int = 4
    timeout_seconds: int = 3600

    @field_validator("max_workers")
    @classmethod
    def _max_workers_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_workers must be >= 1")
        return v


class OnlineSimConfig(BaseModel):
    """Configuration for online (streaming) simulation mode.

    Attributes:
        signals: Signal names to evaluate in streaming mode.
        measure_latency: Whether to measure per-bin latency.
        measure_memory: Whether to measure peak memory usage.
        warmup_bins: Number of bins to skip before measuring.
    """

    signals: list[str] = Field(default_factory=list)
    measure_latency: bool = True
    measure_memory: bool = True
    warmup_bins: int = 300


class TrackingConfig(BaseModel):
    """Experiment tracking configuration.

    MLflow is canonical; optional W&B mirroring can be enabled per campaign.
    """

    backend: Literal["mlflow", "none"] = "mlflow"
    tracking_uri: str | None = None
    experiment_name: str | None = None
    run_name_prefix: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    wandb_mirror: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Loaded from YAML via ``ExperimentConfig.from_yaml(path)``. Defines every
    parameter needed to run an experiment: which datasets, which signals,
    grid variant parameters, evaluation settings, sweep axes, and parallelism.

    Attributes:
        name: Unique experiment name (used as directory name).
        description: Free-text description of the experiment.
        datasets: List of dataset IDs to evaluate against.
        signals: Signal names to evaluate, or ``["all"]`` for all registered.
        grid_variant: Optional grid generation parameters (None = use immutable).
        eval: TP/SL evaluation configuration.
        sweep: Parameter sweep configuration.
        parallel: Parallelism settings.
        online_sim: Optional online simulation configuration.
        tracking: Experiment tracking backend/settings.
    """

    name: str
    description: str = ""
    datasets: list[str]
    signals: list[str]
    grid_variant: GridVariantConfig | None = None
    eval: EvalConfig = Field(default_factory=EvalConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    online_sim: OnlineSimConfig | None = None
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @field_validator("datasets")
    @classmethod
    def _datasets_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("datasets must contain at least one entry")
        return v

    @field_validator("signals")
    @classmethod
    def _signals_nonempty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("signals must contain at least one entry")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load and validate an experiment config from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ExperimentConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
            pydantic.ValidationError: If the parsed data fails schema validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info("Loading experiment config from %s", path)
        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Config file is empty: {path}")

        return cls.model_validate(raw)
