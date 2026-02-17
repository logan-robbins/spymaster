"""Experiment runner: orchestrates signal computation and TP/SL evaluation.

Loads an ExperimentConfig, expands the parameter grid (datasets x signals x
params x tp/sl x cooldown), executes each combination, and persists results
to the append-only ResultsDB.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config_schema import ExperimentConfig
from .dataset_registry import DatasetRegistry
from .eval_engine import EvalEngine
from .results_db import ResultsDB
from .signals import SIGNAL_REGISTRY, get_signal_class
from .signals.base import MLSignal, MLSignalResult, SignalResult, StatisticalSignal

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates experiment execution from config to persisted results.

    Resolves datasets once per ID, instantiates signal classes with swept
    parameters, evaluates via EvalEngine, and appends all results to a
    Parquet-backed ResultsDB.

    Args:
        lake_root: Root path of the data lake (e.g. ``backend/lake``).
    """

    def __init__(self, lake_root: Path) -> None:
        self.lake_root: Path = Path(lake_root)
        self.registry: DatasetRegistry = DatasetRegistry(self.lake_root)
        self.results_db: ResultsDB = ResultsDB(
            self.lake_root / "research" / "vp_harness" / "results"
        )

    def run(self, config: ExperimentConfig) -> list[str]:
        """Execute an experiment config and return all generated run IDs.

        Steps:
            1. Resolve signal names (expand ``["all"]`` to full registry).
            2. Expand the parameter grid into individual run specs.
            3. Group specs by dataset to load each grid only once.
            4. Execute each spec: instantiate signal, compute, evaluate.
            5. Append results to ResultsDB.
            6. Print summary ranking.

        Args:
            config: Validated experiment configuration.

        Returns:
            List of run_id strings, one per (dataset, signal, params) combo.
        """
        config_hash: str = self._compute_config_hash(config)
        specs: list[dict[str, Any]] = self._expand_param_grid(config)
        logger.info(
            "Experiment '%s': %d run specs expanded from config",
            config.name,
            len(specs),
        )

        if not specs:
            logger.warning("No run specs generated -- nothing to execute")
            return []

        # Group specs by dataset_id so we load each dataset once
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            by_dataset.setdefault(spec["dataset_id"], []).append(spec)

        run_ids: list[str] = []
        max_workers: int = config.parallel.max_workers

        if max_workers > 1:
            run_ids = self._run_parallel(
                config, config_hash, by_dataset, max_workers
            )
        else:
            run_ids = self._run_sequential(config, config_hash, by_dataset)

        self._print_summary(config)
        return run_ids

    def _run_sequential(
        self,
        config: ExperimentConfig,
        config_hash: str,
        by_dataset: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """Execute all specs sequentially, one dataset at a time.

        Args:
            config: Experiment config for metadata.
            config_hash: SHA-256 hash of the serialized config.
            by_dataset: Specs grouped by dataset_id.

        Returns:
            List of generated run_id strings.
        """
        run_ids: list[str] = []

        for dataset_id, dataset_specs in by_dataset.items():
            dataset_cache: dict[str, Any] = self._load_dataset_for_specs(
                dataset_id, dataset_specs
            )

            for spec in dataset_specs:
                try:
                    meta, results = self._run_single(
                        dataset_id=spec["dataset_id"],
                        signal_name=spec["signal_name"],
                        signal_params=spec["signal_params"],
                        eval_config=spec["eval_config"],
                        dataset_cache=dataset_cache,
                    )
                    meta["experiment_name"] = config.name
                    meta["config_hash"] = config_hash
                    meta["grid_variant_id"] = (
                        "immutable"
                        if config.grid_variant is None
                        else "variant"
                    )

                    run_id: str = self.results_db.append_run(meta, results)
                    run_ids.append(run_id)
                    logger.info(
                        "Completed run %s: signal=%s dataset=%s tp_ticks=%d sl_ticks=%d",
                        run_id,
                        spec["signal_name"],
                        spec["dataset_id"],
                        spec["eval_config"]["tp_ticks"],
                        spec["eval_config"]["sl_ticks"],
                    )
                except Exception:
                    logger.exception(
                        "Failed: signal=%s dataset=%s params=%s",
                        spec["signal_name"],
                        spec["dataset_id"],
                        spec["signal_params"],
                    )

        return run_ids

    def _run_parallel(
        self,
        config: ExperimentConfig,
        config_hash: str,
        by_dataset: dict[str, list[dict[str, Any]]],
        max_workers: int,
    ) -> list[str]:
        """Execute specs using a ThreadPoolExecutor.

        Datasets are still loaded sequentially (one per dataset_id), but
        signal evaluations within a dataset run in parallel threads.

        Args:
            config: Experiment config for metadata.
            config_hash: SHA-256 hash of the serialized config.
            by_dataset: Specs grouped by dataset_id.
            max_workers: Maximum concurrent threads.

        Returns:
            List of generated run_id strings.
        """
        run_ids: list[str] = []

        for dataset_id, dataset_specs in by_dataset.items():
            dataset_cache: dict[str, Any] = self._load_dataset_for_specs(
                dataset_id, dataset_specs
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_spec = {
                    executor.submit(
                        self._run_single,
                        dataset_id=spec["dataset_id"],
                        signal_name=spec["signal_name"],
                        signal_params=spec["signal_params"],
                        eval_config=spec["eval_config"],
                        dataset_cache=dataset_cache,
                    ): spec
                    for spec in dataset_specs
                }

                for future in as_completed(future_to_spec):
                    spec = future_to_spec[future]
                    try:
                        meta, results = future.result(
                            timeout=config.parallel.timeout_seconds
                        )
                        meta["experiment_name"] = config.name
                        meta["config_hash"] = config_hash
                        meta["grid_variant_id"] = (
                            "immutable"
                            if config.grid_variant is None
                            else "variant"
                        )

                        run_id: str = self.results_db.append_run(meta, results)
                        run_ids.append(run_id)
                        logger.info(
                            "Completed run %s: signal=%s dataset=%s",
                            run_id,
                            spec["signal_name"],
                            spec["dataset_id"],
                        )
                    except Exception:
                        logger.exception(
                            "Failed: signal=%s dataset=%s params=%s",
                            spec["signal_name"],
                            spec["dataset_id"],
                            spec["signal_params"],
                        )

        return run_ids

    def _load_dataset_for_specs(
        self,
        dataset_id: str,
        specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Load a dataset with the union of all columns needed by specs.

        Args:
            dataset_id: Dataset identifier to resolve and load.
            specs: List of run specs targeting this dataset.

        Returns:
            Loaded dataset dict (shared across all specs for this dataset).
        """
        all_columns: set[str] = set()
        for spec in specs:
            signal_cls = get_signal_class(spec["signal_name"])
            # Instantiate temporarily to read required_columns
            try:
                instance = signal_cls(**spec["signal_params"])
            except TypeError:
                instance = signal_cls()
            all_columns.update(instance.required_columns)

        columns: list[str] = sorted(all_columns)
        logger.info(
            "Loading dataset '%s' with columns: %s", dataset_id, columns
        )

        engine = EvalEngine()
        return engine.load_dataset(dataset_id, columns, self.registry)

    def _run_single(
        self,
        dataset_id: str,
        signal_name: str,
        signal_params: dict[str, Any],
        eval_config: dict[str, Any],
        dataset_cache: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Run a single signal + params combo on a cached dataset.

        Instantiates the signal, calls compute(), then dispatches to
        sweep_thresholds (statistical) or evaluate_ml_predictions (ML).

        Args:
            dataset_id: Dataset identifier string.
            signal_name: Registered signal name.
            signal_params: Keyword arguments for signal constructor.
            eval_config: Dict with tp_ticks, sl_ticks, max_hold_bins,
                warmup_bins, tick_size, cooldown_bins.
            dataset_cache: Pre-loaded dataset dict.

        Returns:
            Tuple of (meta_dict, results_list) ready for ResultsDB.append_run.
        """
        t0: float = time.monotonic()

        signal_cls = get_signal_class(signal_name)
        try:
            signal_instance = signal_cls(**signal_params)
        except TypeError:
            signal_instance = signal_cls()

        engine = EvalEngine(
            tp_ticks=eval_config["tp_ticks"],
            sl_ticks=eval_config["sl_ticks"],
            max_hold_bins=eval_config.get("max_hold_bins", 1200),
            warmup_bins=eval_config.get("warmup_bins", 300),
            tick_size=eval_config.get("tick_size", 0.25),
        )

        cooldown_bins: int = eval_config.get("cooldown_bins", 30)
        mid_price = dataset_cache["mid_price"]
        ts_ns = dataset_cache["ts_ns"]

        if isinstance(signal_instance, MLSignal):
            result: MLSignalResult = signal_instance.compute(dataset_cache)
            thresholds: list[float] = signal_instance.default_thresholds()
            results: list[dict[str, Any]] = engine.evaluate_ml_predictions(
                predictions=result.predictions,
                has_prediction=result.has_prediction,
                thresholds=thresholds,
                cooldown_bins=cooldown_bins,
                mid_price=mid_price,
                ts_ns=ts_ns,
                mode=signal_instance.prediction_mode,
            )
        elif isinstance(signal_instance, StatisticalSignal):
            result_stat: SignalResult = signal_instance.compute(dataset_cache)
            thresholds = signal_instance.default_thresholds()
            results = engine.sweep_thresholds(
                signal=result_stat.signal,
                thresholds=thresholds,
                cooldown_bins=cooldown_bins,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )
        else:
            raise TypeError(
                f"Signal '{signal_name}' is neither StatisticalSignal "
                f"nor MLSignal: {type(signal_instance)}"
            )

        elapsed: float = time.monotonic() - t0

        meta: dict[str, Any] = {
            "dataset_id": dataset_id,
            "signal_name": signal_name,
            "signal_params_json": json.dumps(signal_params, sort_keys=True),
            "eval_tp_ticks": eval_config["tp_ticks"],
            "eval_sl_ticks": eval_config["sl_ticks"],
            "eval_max_hold_bins": eval_config.get("max_hold_bins", 1200),
            "elapsed_seconds": round(elapsed, 3),
        }

        logger.info(
            "Signal '%s' on '%s': %d threshold results in %.2fs",
            signal_name,
            dataset_id,
            len(results),
            elapsed,
        )

        return meta, results

    def _expand_param_grid(
        self, config: ExperimentConfig
    ) -> list[dict[str, Any]]:
        """Expand config into a list of individual run specifications.

        Produces the cartesian product of:
            datasets x signals x signal_params x tp_ticks x sl_ticks x cooldown

        Args:
            config: Validated experiment configuration.

        Returns:
            List of run spec dicts, each with keys: dataset_id, signal_name,
            signal_params, eval_config.
        """
        # Resolve signal names
        if config.signals == ["all"]:
            signal_names: list[str] = sorted(SIGNAL_REGISTRY.keys())
        else:
            signal_names = list(config.signals)

        # Normalize tp_ticks and sl_ticks to lists
        tp_values: list[int] = (
            config.eval.tp_ticks
            if isinstance(config.eval.tp_ticks, list)
            else [config.eval.tp_ticks]
        )
        sl_values: list[int] = (
            config.eval.sl_ticks
            if isinstance(config.eval.sl_ticks, list)
            else [config.eval.sl_ticks]
        )

        cooldown_values: list[int] = config.sweep.universal.get(
            "cooldown_bins", [30]
        )
        if not isinstance(cooldown_values, list):
            cooldown_values = [cooldown_values]

        specs: list[dict[str, Any]] = []

        for dataset_id, signal_name in itertools.product(
            config.datasets, signal_names
        ):
            # Build parameter combos for this signal
            param_combos: list[dict[str, Any]] = self._expand_signal_params(
                signal_name, config
            )

            for params, tp, sl, cd in itertools.product(
                param_combos, tp_values, sl_values, cooldown_values
            ):
                specs.append(
                    {
                        "dataset_id": dataset_id,
                        "signal_name": signal_name,
                        "signal_params": dict(params),
                        "eval_config": {
                            "tp_ticks": tp,
                            "sl_ticks": sl,
                            "max_hold_bins": config.eval.max_hold_bins,
                            "warmup_bins": config.eval.warmup_bins,
                            "tick_size": config.eval.tick_size,
                            "cooldown_bins": cd,
                        },
                    }
                )

        return specs

    def _expand_signal_params(
        self,
        signal_name: str,
        config: ExperimentConfig,
    ) -> list[dict[str, Any]]:
        """Expand sweep parameters for a single signal into param combos.

        Merges universal sweep params with per-signal overrides, then
        computes the cartesian product of all sweep axes.

        Args:
            signal_name: Registered signal name.
            config: Experiment config containing sweep definitions.

        Returns:
            List of param dicts. Returns ``[{}]`` if no sweep axes exist
            (single default run).
        """
        # Start with universal params (excluding cooldown_bins, handled above)
        merged: dict[str, list[Any]] = {}
        for key, values in config.sweep.universal.items():
            if key == "cooldown_bins":
                continue
            merged[key] = values if isinstance(values, list) else [values]

        # Per-signal overrides replace universal keys
        per_signal: dict[str, list[Any]] = config.sweep.per_signal.get(
            signal_name, {}
        )
        for key, values in per_signal.items():
            merged[key] = values if isinstance(values, list) else [values]

        if not merged:
            return [{}]

        keys: list[str] = sorted(merged.keys())
        value_lists: list[list[Any]] = [merged[k] for k in keys]

        combos: list[dict[str, Any]] = []
        for combo_values in itertools.product(*value_lists):
            combos.append(dict(zip(keys, combo_values)))

        return combos

    def _print_summary(self, config: ExperimentConfig) -> None:
        """Print a ranked summary of the best result per signal.

        Queries the ResultsDB for the current experiment and logs a
        table sorted by tp_rate descending.

        Args:
            config: Experiment config (used to filter by experiment_name).
        """
        best = self.results_db.query_best(min_signals=config.eval.min_signals)
        if best.empty:
            logger.info("No results with min_signals >= %d", config.eval.min_signals)
            return

        # Filter to current experiment
        if "experiment_name" in best.columns:
            best = best[best["experiment_name"] == config.name]

        if best.empty:
            return

        logger.info("=== Experiment '%s' Summary ===", config.name)
        for _, row in best.iterrows():
            signal: str = row.get("signal_name", "?")
            tp_rate: float = row.get("tp_rate", float("nan"))
            n_sig: int = int(row.get("n_signals", 0))
            mean_pnl: float = row.get("mean_pnl_ticks", float("nan"))
            threshold: float = row.get("threshold", float("nan"))
            logger.info(
                "  %-20s tp=%.1f%%  n=%d  pnl=%.2f  thr=%.3f",
                signal,
                tp_rate * 100,
                n_sig,
                mean_pnl,
                threshold,
            )

    @staticmethod
    def _compute_config_hash(config: ExperimentConfig) -> str:
        """Compute a deterministic SHA-256 hash of the config.

        Args:
            config: Experiment configuration to hash.

        Returns:
            Hex-encoded SHA-256 hash string (64 characters).
        """
        serialized: str = json.dumps(
            config.model_dump(), sort_keys=True, default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
