"""Pydantic v2 experiment specification for the three-config-layer architecture.

ExperimentSpec references a ServingSpec by name as its defaults and adds
everything needed for offline evaluation: TP/SL parameters, cooldown/warmup,
sweep axes (over scoring + signal + projection params), parallel execution,
and MLflow tracking.

Config hierarchy:
    PipelineSpec  ->  ServingSpec  ->  ExperimentSpec
    (grid/data)       (scoring/signal)  (eval/sweep/tracking)

Many experiments can share one serving config.  The sweep explores the
neighborhood around the serving config defaults.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..vp_shared.yaml_io import load_yaml_mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ExperimentEvalConfig(BaseModel):
    """TP/SL evaluation parameters for offline backtesting.

    Fields accepting ``list[int]`` enable threshold sweeps across TP/SL
    values within a single experiment run.

    Attributes:
        tp_ticks: Take-profit distance(s) in ticks.
        sl_ticks: Stop-loss distance(s) in ticks.
        max_hold_bins: Maximum bins to hold a position before timeout.
        warmup_bins: Bins to skip at the start before generating signals.
        tick_size: Dollar value of one tick (product-dependent).
        cooldown_bins: Bins to wait after a signal before allowing the next.
        min_signals: Minimum number of signals required for a threshold
            evaluation to be considered statistically valid.
    """

    tp_ticks: list[int] | int = Field(default=[8])
    sl_ticks: list[int] | int = Field(default=[4])
    max_hold_bins: int = 1200
    warmup_bins: int = 300
    tick_size: float = 0.25
    cooldown_bins: int | list[int] = 20
    min_signals: int = 5

    @field_validator("min_signals")
    @classmethod
    def _min_signals_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_signals must be >= 1")
        return v

    @field_validator("max_hold_bins")
    @classmethod
    def _max_hold_bins_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_hold_bins must be >= 1")
        return v

    @field_validator("warmup_bins")
    @classmethod
    def _warmup_bins_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_bins must be >= 0")
        return v

    @field_validator("tick_size")
    @classmethod
    def _tick_size_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("tick_size must be > 0")
        return v


class ExperimentSweepConfig(BaseModel):
    """Parameter sweep configuration.

    Defines axes to explore around the serving config defaults.

    Attributes:
        scoring: Keys are ScoringConfig field names, values are lists
            of values to sweep (e.g. ``{"zscore_window_bins": [120, 240, 480]}``).
        per_signal: Signal-specific sweep axes.  Outer key is the signal
            name, inner dict maps parameter names to lists of values.
    """

    scoring: dict[str, Any] = Field(default_factory=dict)
    per_signal: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ExperimentParallelConfig(BaseModel):
    """Parallelism configuration for experiment execution.

    Attributes:
        max_workers: Maximum number of concurrent workers.
        timeout_seconds: Per-worker timeout in seconds.
    """

    max_workers: int = 3
    timeout_seconds: int = 7200

    @field_validator("max_workers")
    @classmethod
    def _max_workers_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_workers must be >= 1")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("timeout_seconds must be >= 1")
        return v


class ExperimentTrackingConfig(BaseModel):
    """Experiment tracking backend configuration.

    Attributes:
        backend: Tracking backend to use.  ``"mlflow"`` for MLflow,
            ``"none"`` to disable tracking.
        experiment_name: MLflow experiment name override.  Defaults to
            ``vp/{experiment.name}`` when ``None``.
        run_name_prefix: Optional prefix for individual run names.
        tags: Arbitrary key-value tags attached to every run.
    """

    backend: Literal["mlflow", "none"] = "mlflow"
    experiment_name: str | None = None
    run_name_prefix: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


class ExperimentSpec(BaseModel):
    """Top-level experiment specification.

    References a ``ServingSpec`` by name for scoring/signal defaults and
    layers on evaluation parameters, sweep axes, parallelism, and tracking.

    Loaded from YAML via ``ExperimentSpec.from_yaml(path)``.

    Attributes:
        name: Unique experiment name (used as directory name and MLflow key).
        description: Free-text description of the experiment.
        serving: Name reference to a ServingSpec YAML file.
        eval: TP/SL evaluation configuration.
        sweep: Parameter sweep configuration.
        parallel: Parallelism settings.
        tracking: Experiment tracking backend/settings.
    """

    name: str
    description: str = ""
    serving: str
    eval: ExperimentEvalConfig = Field(default_factory=ExperimentEvalConfig)
    sweep: ExperimentSweepConfig = Field(default_factory=ExperimentSweepConfig)
    parallel: ExperimentParallelConfig = Field(
        default_factory=ExperimentParallelConfig
    )
    tracking: ExperimentTrackingConfig = Field(
        default_factory=ExperimentTrackingConfig
    )

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must be non-empty")
        return v.strip()

    @field_validator("serving")
    @classmethod
    def _serving_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("serving reference must be non-empty")
        return v.strip()

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolve_serving(self, lake_root: Path) -> Any:
        """Resolve the referenced ServingSpec from disk.

        Performs a lazy import of ``ServingSpec`` to avoid circular imports
        during module initialization.

        Args:
            lake_root: Root path of the data lake.

        Returns:
            Validated ``ServingSpec`` instance.

        Raises:
            FileNotFoundError: If the named serving config does not exist.
        """
        from .serving_config import ServingSpec

        return ServingSpec.load_by_name(self.serving, lake_root)

    # ------------------------------------------------------------------
    # Harness bridge
    # ------------------------------------------------------------------

    def to_harness_config(self, lake_root: Path) -> dict[str, Any]:
        """Convert to a dict compatible with the experiment_harness ExperimentConfig schema.

        Resolves the serving config and its upstream pipeline config,
        then assembles the flat dict consumed by ``ExperimentRunner``.

        Args:
            lake_root: Root path of the data lake.

        Returns:
            Dict matching the ``ExperimentConfig`` field layout in
            ``experiment_harness.config_schema``.
        """
        serving_spec = self.resolve_serving(lake_root)
        pipeline_spec = serving_spec.resolve_pipeline(lake_root)

        signal_names: list[str] = (
            [serving_spec.signal.name]
            if serving_spec.signal
            else ["derivative"]
        )

        return {
            "name": self.name,
            "description": self.description,
            "datasets": [pipeline_spec.dataset_id()],
            "signals": signal_names,
            "eval": {
                "tp_ticks": self.eval.tp_ticks,
                "sl_ticks": self.eval.sl_ticks,
                "max_hold_bins": self.eval.max_hold_bins,
                "warmup_bins": self.eval.warmup_bins,
                "tick_size": self.eval.tick_size,
                "cooldown_bins": self.eval.cooldown_bins,
                "min_signals": self.eval.min_signals,
            },
            "sweep": {
                "universal": self._build_universal_sweep(serving_spec),
                "per_signal": self.sweep.per_signal,
            },
            "parallel": {
                "max_workers": self.parallel.max_workers,
                "timeout_seconds": self.parallel.timeout_seconds,
            },
            "tracking": {
                "backend": self.tracking.backend,
                "experiment_name": (
                    self.tracking.experiment_name or f"vp/{self.name}"
                ),
                "run_name_prefix": self.tracking.run_name_prefix,
                "tags": self.tracking.tags,
            },
        }

    def _build_universal_sweep(self, serving_spec: Any) -> dict[str, list[Any]]:
        """Build the universal sweep dict for the harness runner.

        Starts with scoring sweep axes from ``self.sweep.scoring`` and
        includes signal-level params (like ``zscore_window_bins``) when
        present.

        Args:
            serving_spec: Resolved ``ServingSpec`` (used for type context
                but not mutated).

        Returns:
            Dict mapping parameter names to lists of sweep values.
        """
        universal: dict[str, list[Any]] = {}

        for key, values in self.sweep.scoring.items():
            if isinstance(values, list):
                universal[key] = values
            else:
                universal[key] = [values]

        return universal

    # ------------------------------------------------------------------
    # Promotion: extract winning params from a completed run
    # ------------------------------------------------------------------

    def extract_winning_serving(
        self,
        run_id: str,
        results_db_root: Path,
        lake_root: Path,
    ) -> Any:
        """Create a new ServingSpec with the winning params from a completed run.

        Loads run metadata from ``ResultsDB``, parses the stored
        ``signal_params_json``, overlays the winning parameter values onto
        the base serving config, and returns a promotable ``ServingSpec``.

        Args:
            run_id: The 16-character hex run identifier.
            results_db_root: Path to the ResultsDB root directory.
            lake_root: Root path of the data lake.

        Returns:
            New ``ServingSpec`` with winning params applied and a
            descriptive name of the form ``{experiment}_promoted_{run_id[:8]}``.

        Raises:
            ValueError: If the run_id is not found in the results DB.
            KeyError: If required metadata fields are missing.
        """
        from .serving_config import ServingSpec, ScoringConfig

        from ..experiment_harness.results_db import ResultsDB

        db = ResultsDB(results_db_root)
        runs_meta = db.query_runs(run_id=run_id)

        if runs_meta.empty:
            raise ValueError(
                f"Run '{run_id}' not found in ResultsDB at {results_db_root}"
            )

        row = runs_meta.iloc[0]
        signal_params_json: str = row.get("signal_params_json", "{}")
        signal_params: dict[str, Any] = json.loads(signal_params_json)

        base_serving = self.resolve_serving(lake_root)

        # Partition signal_params into scoring fields vs signal-specific params
        scoring_field_names: set[str] = set(
            ScoringConfig.model_fields.keys()
        )
        scoring_overrides: dict[str, Any] = {}
        signal_overrides: dict[str, Any] = {}

        for param_key, param_value in signal_params.items():
            if param_key in scoring_field_names:
                scoring_overrides[param_key] = param_value
            else:
                signal_overrides[param_key] = param_value

        # Build updated scoring config
        base_scoring_data: dict[str, Any] = base_serving.scoring.model_dump()
        base_scoring_data.update(scoring_overrides)
        new_scoring = ScoringConfig.model_validate(base_scoring_data)

        # Build updated signal config
        new_signal = None
        if base_serving.signal is not None:
            signal_data: dict[str, Any] = base_serving.signal.model_dump()
            if "params" in signal_data and signal_overrides:
                signal_data["params"].update(signal_overrides)
            elif signal_overrides:
                signal_data["params"] = signal_overrides
            # Re-import the signal config class from the base model
            new_signal = base_serving.signal.model_validate(signal_data)

        promoted_name: str = f"{self.name}_promoted_{run_id[:8]}"

        # Build the promoted serving spec
        promoted_data: dict[str, Any] = base_serving.model_dump()
        promoted_data["name"] = promoted_name
        promoted_data["scoring"] = new_scoring.model_dump()
        if new_signal is not None:
            promoted_data["signal"] = new_signal.model_dump()

        return ServingSpec.model_validate(promoted_data)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentSpec:
        """Load and validate an experiment spec from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ``ExperimentSpec`` instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
            pydantic.ValidationError: If the parsed data fails schema validation.
        """
        raw = load_yaml_mapping(
            Path(path),
            not_found_message="Experiment config not found: {path}",
            empty_message="Experiment config file is empty: {path}",
            non_mapping_message="Experiment config must be a mapping, got {kind}: {path}",
            logger=logger,
            log_message="Loading experiment spec from %s",
        )
        return cls.model_validate(raw)

    @classmethod
    def configs_dir(cls, lake_root: Path) -> Path:
        """Return the canonical directory for experiment config YAML files.

        Args:
            lake_root: Root path of the data lake.

        Returns:
            ``lake_root / "research" / "vp_harness" / "configs" / "experiments"``
        """
        return Path(lake_root) / "research" / "vp_harness" / "configs" / "experiments"
