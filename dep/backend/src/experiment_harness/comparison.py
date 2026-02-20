"""Cross-experiment ranking and parameter sensitivity analysis.

Provides methods for ranking experiment runs by TP rate and PnL, analyzing
parameter sensitivity, generating TP/SL heatmaps, comparing across datasets,
and producing summary tables. All queries go through ResultsDB.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .results_db import ResultsDB

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """Cross-experiment ranking and analysis over the results store.

    Queries ResultsDB to rank runs, analyze parameter sensitivity,
    and produce summary tables for comparing signal performance.

    Args:
        lake_root: Root path of the data lake (e.g. ``backend/lake``).
    """

    def __init__(self, lake_root: Path) -> None:
        self.results_db: ResultsDB = ResultsDB(
            Path(lake_root) / "research" / "vp_harness" / "results"
        )

    def _load_merged(
        self,
        signal: str | None = None,
        dataset_id: str | None = None,
        min_signals: int = 0,
    ) -> pd.DataFrame:
        """Load and merge runs + runs_meta, applying optional filters.

        Args:
            signal: If provided, filter to this signal name.
            dataset_id: If provided, filter to this dataset.
            min_signals: Minimum n_signals per row. Rows below are dropped.

        Returns:
            Merged DataFrame with both run metadata and threshold results.
            Empty DataFrame if no data is available.
        """
        meta_path: Path = self.results_db._meta_path
        runs_path: Path = self.results_db._runs_path

        if not meta_path.exists() or not runs_path.exists():
            logger.warning("Results store is empty: %s", self.results_db._root)
            return pd.DataFrame()

        meta: pd.DataFrame = pd.read_parquet(meta_path)
        runs: pd.DataFrame = pd.read_parquet(runs_path)

        if meta.empty or runs.empty:
            return pd.DataFrame()

        # Determine which metadata columns to merge (all except run_id itself,
        # which is the join key, and any columns already in runs)
        runs_cols: set[str] = set(runs.columns)
        meta_merge_cols: list[str] = ["run_id"] + [
            c for c in meta.columns if c != "run_id" and c not in runs_cols
        ]
        merged: pd.DataFrame = runs.merge(
            meta[meta_merge_cols], on="run_id", how="inner"
        )

        if signal is not None and "signal_name" in merged.columns:
            merged = merged[merged["signal_name"] == signal]

        if dataset_id is not None and "dataset_id" in merged.columns:
            merged = merged[merged["dataset_id"] == dataset_id]

        if min_signals > 0 and "n_signals" in merged.columns:
            merged = merged[merged["n_signals"] >= min_signals]

        return merged.reset_index(drop=True)

    def rank_by_tp_rate(
        self,
        signal: str | None = None,
        dataset_id: str | None = None,
        min_signals: int = 5,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Rank all runs by TP rate, selecting the best threshold per run.

        For each run_id, selects the threshold with the highest TP rate
        (subject to min_signals). Returns the top_n results sorted descending.

        Args:
            signal: If provided, filter to this signal name.
            dataset_id: If provided, filter to this dataset.
            min_signals: Minimum number of signals for a threshold to qualify.
            top_n: Maximum number of results to return.

        Returns:
            DataFrame with columns including signal_name, dataset_id,
            threshold, tp_rate, sl_rate, n_signals, mean_pnl_ticks,
            eval_tp_ticks, eval_sl_ticks. Empty DataFrame if no data.
        """
        merged: pd.DataFrame = self._load_merged(
            signal=signal, dataset_id=dataset_id, min_signals=min_signals
        )
        if merged.empty:
            return merged

        # Best threshold per run by tp_rate
        best_idx: pd.Series = merged.groupby("run_id")["tp_rate"].idxmax()
        best: pd.DataFrame = merged.loc[best_idx].copy()

        # Sort by tp_rate descending, take top_n
        best = best.sort_values("tp_rate", ascending=False).head(top_n)

        # Select display columns
        display_cols: list[str] = [
            "run_id",
            "signal_name",
            "dataset_id",
            "threshold",
            "tp_rate",
            "sl_rate",
            "timeout_rate",
            "n_signals",
            "mean_pnl_ticks",
            "events_per_hour",
            "eval_tp_ticks",
            "eval_sl_ticks",
        ]
        available: list[str] = [c for c in display_cols if c in best.columns]
        return best[available].reset_index(drop=True)

    def rank_by_pnl(
        self,
        signal: str | None = None,
        dataset_id: str | None = None,
        min_signals: int = 20,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Rank all runs by mean PnL in ticks.

        For each run_id, selects the threshold with the highest mean PnL
        (subject to min_signals). Returns the top_n results sorted descending.

        Args:
            signal: If provided, filter to this signal name.
            dataset_id: If provided, filter to this dataset.
            min_signals: Minimum number of signals for a threshold to qualify.
            top_n: Maximum number of results to return.

        Returns:
            DataFrame ranked by mean_pnl_ticks descending.
            Empty DataFrame if no data.
        """
        merged: pd.DataFrame = self._load_merged(
            signal=signal, dataset_id=dataset_id, min_signals=min_signals
        )
        if merged.empty or "mean_pnl_ticks" not in merged.columns:
            return pd.DataFrame()

        # Best threshold per run by mean_pnl_ticks
        best_idx: pd.Series = merged.groupby("run_id")["mean_pnl_ticks"].idxmax()
        best: pd.DataFrame = merged.loc[best_idx].copy()

        best = best.sort_values("mean_pnl_ticks", ascending=False).head(top_n)

        display_cols: list[str] = [
            "run_id",
            "signal_name",
            "dataset_id",
            "threshold",
            "mean_pnl_ticks",
            "tp_rate",
            "sl_rate",
            "n_signals",
            "events_per_hour",
            "eval_tp_ticks",
            "eval_sl_ticks",
        ]
        available: list[str] = [c for c in display_cols if c in best.columns]
        return best[available].reset_index(drop=True)

    def param_sensitivity(
        self,
        signal: str,
        param_name: str,
    ) -> pd.DataFrame:
        """Show how a parameter affects TP rate for a given signal.

        Extracts param_name values from signal_params_json, groups results
        by the parameter value, and reports mean/std TP rate per group.

        Args:
            signal: Signal name to analyze.
            param_name: Parameter name within signal_params_json to vary.

        Returns:
            DataFrame with columns: param_value, mean_tp_rate, std_tp_rate,
            n_runs. Sorted by param_value ascending. Empty DataFrame if
            no data or param_name not found.
        """
        merged: pd.DataFrame = self._load_merged(signal=signal, min_signals=1)
        if merged.empty:
            return pd.DataFrame()

        if "signal_params_json" not in merged.columns:
            logger.warning("No signal_params_json column in results")
            return pd.DataFrame()

        # Extract param values from JSON
        def _extract_param(params_json: Any) -> Any:
            """Parse JSON and extract the target parameter value."""
            if params_json is None or (
                isinstance(params_json, float) and np.isnan(params_json)
            ):
                return None
            try:
                if isinstance(params_json, str):
                    params: dict[str, Any] = json.loads(params_json)
                elif isinstance(params_json, dict):
                    params = params_json
                else:
                    return None
                return params.get(param_name)
            except (json.JSONDecodeError, TypeError):
                return None

        merged["_param_value"] = merged["signal_params_json"].apply(_extract_param)
        filtered: pd.DataFrame = merged.dropna(subset=["_param_value"])

        if filtered.empty:
            logger.warning(
                "Parameter '%s' not found in any run for signal '%s'",
                param_name,
                signal,
            )
            return pd.DataFrame()

        # Get best tp_rate per run (to avoid counting multiple thresholds)
        best_per_run: pd.DataFrame = (
            filtered.groupby("run_id")
            .agg(
                tp_rate=("tp_rate", "max"),
                _param_value=("_param_value", "first"),
            )
            .reset_index()
        )

        # Group by parameter value
        grouped: pd.DataFrame = (
            best_per_run.groupby("_param_value")
            .agg(
                mean_tp_rate=("tp_rate", "mean"),
                std_tp_rate=("tp_rate", "std"),
                n_runs=("tp_rate", "count"),
            )
            .reset_index()
        )
        grouped = grouped.rename(columns={"_param_value": "param_value"})
        grouped["std_tp_rate"] = grouped["std_tp_rate"].fillna(0.0)

        return grouped.sort_values("param_value").reset_index(drop=True)

    def tp_sl_heatmap(
        self,
        signal: str,
        dataset_id: str | None = None,
    ) -> pd.DataFrame:
        """Create TP/SL heatmap: rows=tp_ticks, cols=sl_ticks, values=best TP rate.

        For each (tp_ticks, sl_ticks) combination, finds the run with the
        highest TP rate and uses that as the cell value.

        Args:
            signal: Signal name to analyze.
            dataset_id: If provided, filter to this dataset.

        Returns:
            Pivot DataFrame with tp_ticks as index, sl_ticks as columns,
            and best TP rate as values. Empty DataFrame if no data.
        """
        merged: pd.DataFrame = self._load_merged(
            signal=signal, dataset_id=dataset_id, min_signals=1
        )
        if merged.empty:
            return pd.DataFrame()

        required_cols: list[str] = ["eval_tp_ticks", "eval_sl_ticks", "tp_rate"]
        if not all(c in merged.columns for c in required_cols):
            logger.warning(
                "Missing required columns for heatmap: %s",
                [c for c in required_cols if c not in merged.columns],
            )
            return pd.DataFrame()

        # Best TP rate per (tp_ticks, sl_ticks) combination
        grouped: pd.DataFrame = (
            merged.groupby(["eval_tp_ticks", "eval_sl_ticks"])["tp_rate"]
            .max()
            .reset_index()
        )

        if grouped.empty:
            return pd.DataFrame()

        # Pivot to heatmap format
        heatmap: pd.DataFrame = grouped.pivot(
            index="eval_tp_ticks",
            columns="eval_sl_ticks",
            values="tp_rate",
        )
        heatmap.index.name = "tp_ticks"
        heatmap.columns.name = "sl_ticks"

        return heatmap

    def cross_dataset(
        self,
        signal: str,
    ) -> pd.DataFrame:
        """Compare signal performance across different datasets.

        For each dataset_id, reports the best TP rate, best mean PnL,
        number of runs, and median events_per_hour.

        Args:
            signal: Signal name to compare across datasets.

        Returns:
            DataFrame with columns: dataset_id, best_tp_rate, best_pnl_ticks,
            n_runs, median_events_per_hour. Sorted by best_tp_rate descending.
            Empty DataFrame if no data.
        """
        merged: pd.DataFrame = self._load_merged(signal=signal, min_signals=1)
        if merged.empty or "dataset_id" not in merged.columns:
            return pd.DataFrame()

        # Aggregate per dataset
        agg_dict: dict[str, Any] = {"tp_rate": "max", "run_id": "nunique"}

        if "mean_pnl_ticks" in merged.columns:
            agg_dict["mean_pnl_ticks"] = "max"
        if "events_per_hour" in merged.columns:
            agg_dict["events_per_hour"] = "median"

        grouped: pd.DataFrame = (
            merged.groupby("dataset_id").agg(agg_dict).reset_index()
        )

        # Rename for clarity
        rename_map: dict[str, str] = {
            "tp_rate": "best_tp_rate",
            "run_id": "n_runs",
        }
        if "mean_pnl_ticks" in grouped.columns:
            rename_map["mean_pnl_ticks"] = "best_pnl_ticks"
        if "events_per_hour" in grouped.columns:
            rename_map["events_per_hour"] = "median_events_per_hour"

        grouped = grouped.rename(columns=rename_map)

        return grouped.sort_values("best_tp_rate", ascending=False).reset_index(
            drop=True
        )

    def summary_table(self, min_signals: int = 5) -> pd.DataFrame:
        """One-line summary per signal showing best TP rate, PnL, and n_signals.

        Provides a quick overview of all signals' best performance across
        all datasets and parameter configurations.

        Args:
            min_signals: Minimum number of signals for qualifying thresholds.

        Returns:
            DataFrame with columns: signal_name, best_tp_rate, best_pnl_ticks,
            max_n_signals, n_runs, n_datasets. Sorted by best_tp_rate
            descending. Empty DataFrame if no data.
        """
        merged: pd.DataFrame = self._load_merged(min_signals=min_signals)
        if merged.empty or "signal_name" not in merged.columns:
            return pd.DataFrame()

        agg_dict: dict[str, Any] = {
            "tp_rate": "max",
            "run_id": "nunique",
        }
        if "n_signals" in merged.columns:
            agg_dict["n_signals"] = "max"
        if "mean_pnl_ticks" in merged.columns:
            agg_dict["mean_pnl_ticks"] = "max"
        if "dataset_id" in merged.columns:
            agg_dict["dataset_id"] = "nunique"

        grouped: pd.DataFrame = (
            merged.groupby("signal_name").agg(agg_dict).reset_index()
        )

        rename_map: dict[str, str] = {
            "tp_rate": "best_tp_rate",
            "run_id": "n_runs",
        }
        if "n_signals" in grouped.columns:
            rename_map["n_signals"] = "max_n_signals"
        if "mean_pnl_ticks" in grouped.columns:
            rename_map["mean_pnl_ticks"] = "best_pnl_ticks"
        if "dataset_id" in grouped.columns:
            rename_map["dataset_id"] = "n_datasets"

        grouped = grouped.rename(columns=rename_map)

        return grouped.sort_values("best_tp_rate", ascending=False).reset_index(
            drop=True
        )
