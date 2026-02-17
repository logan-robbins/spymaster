"""Append-only Parquet results store for experiment runs.

Maintains two Parquet tables:
- ``runs_meta.parquet``: One row per experiment run with metadata.
- ``runs.parquet``: One row per threshold evaluation within a run.

Supports querying best results per run and filtering by signal/dataset.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ResultsDB:
    """Append-only Parquet results store.

    Stores experiment run metadata and per-threshold evaluation results
    in two separate Parquet files under a root directory.

    Args:
        root: Directory path for the results store. Created if it does not exist.
    """

    def __init__(self, root: Path) -> None:
        self._root: Path = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._meta_path: Path = self._root / "runs_meta.parquet"
        self._runs_path: Path = self._root / "runs.parquet"

    def append_run(self, meta: dict[str, Any], results: list[dict[str, Any]]) -> str:
        """Append one experiment run to the store.

        Args:
            meta: Run metadata dict. Expected keys:
                ``experiment_name``, ``dataset_id``, ``signal_name``,
                ``signal_params_json``, ``grid_variant_id``,
                ``eval_tp_ticks``, ``eval_sl_ticks``, ``eval_max_hold_bins``,
                ``elapsed_seconds``, ``config_hash``.
            results: List of per-threshold result dicts. Expected keys:
                ``threshold``, ``cooldown_bins``, ``n_signals``, ``tp_rate``,
                ``sl_rate``, ``timeout_rate``, ``mean_pnl_ticks``,
                ``events_per_hour``, ``median_time_to_outcome_ms``.

        Returns:
            The generated run_id (16-character hex string).
        """
        run_id: str = uuid.uuid4().hex[:16]
        meta["run_id"] = run_id
        meta["timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat()

        # Append meta row
        meta_df: pd.DataFrame = pd.DataFrame([meta])
        if self._meta_path.exists():
            existing: pd.DataFrame = pd.read_parquet(self._meta_path)
            meta_df = pd.concat([existing, meta_df], ignore_index=True)
        meta_df.to_parquet(self._meta_path, index=False)

        # Append result rows
        for r in results:
            r["run_id"] = run_id
        runs_df: pd.DataFrame = pd.DataFrame(results)
        if self._runs_path.exists():
            existing = pd.read_parquet(self._runs_path)
            runs_df = pd.concat([existing, runs_df], ignore_index=True)
        runs_df.to_parquet(self._runs_path, index=False)

        logger.info(
            "Appended run '%s': %d threshold results for signal '%s' on dataset '%s'",
            run_id,
            len(results),
            meta.get("signal_name", "unknown"),
            meta.get("dataset_id", "unknown"),
        )
        return run_id

    def query_best(
        self,
        *,
        signal: str | None = None,
        dataset_id: str | None = None,
        min_signals: int = 5,
    ) -> pd.DataFrame:
        """Query the best threshold per run, optionally filtered.

        Selects the threshold with the highest TP rate within each run,
        filtering for a minimum number of signals. Results are sorted
        by TP rate descending.

        Args:
            signal: If provided, filter to this signal name only.
            dataset_id: If provided, filter to this dataset only.
            min_signals: Minimum number of signals required for a threshold
                to be considered valid. Defaults to 5.

        Returns:
            DataFrame with one row per run (the best threshold for that run),
            sorted by ``tp_rate`` descending. Empty DataFrame if no data.
        """
        if not self._runs_path.exists() or not self._meta_path.exists():
            return pd.DataFrame()

        runs: pd.DataFrame = pd.read_parquet(self._runs_path)
        meta: pd.DataFrame = pd.read_parquet(self._meta_path)

        merge_cols: list[str] = [
            "run_id",
            "signal_name",
            "dataset_id",
            "experiment_name",
            "eval_tp_ticks",
            "eval_sl_ticks",
        ]
        # Only merge columns that exist in meta
        available_merge_cols: list[str] = [
            c for c in merge_cols if c in meta.columns
        ]
        merged: pd.DataFrame = runs.merge(
            meta[available_merge_cols], on="run_id"
        )

        if signal is not None and "signal_name" in merged.columns:
            merged = merged[merged["signal_name"] == signal]
        if dataset_id is not None and "dataset_id" in merged.columns:
            merged = merged[merged["dataset_id"] == dataset_id]

        if "n_signals" in merged.columns:
            valid: pd.DataFrame = merged[merged["n_signals"] >= min_signals]
        else:
            valid = merged

        if valid.empty:
            return valid

        best_idx: pd.Series = valid.groupby("run_id")["tp_rate"].idxmax()
        best: pd.DataFrame = valid.loc[best_idx]
        return best.sort_values("tp_rate", ascending=False).reset_index(drop=True)

    def query_runs(self, **filters: Any) -> pd.DataFrame:
        """Query run metadata with arbitrary column filters.

        Each keyword argument is matched as an equality filter against the
        corresponding column in ``runs_meta.parquet``.

        Args:
            **filters: Column name / value pairs to filter on
                (e.g. ``experiment_name="sweep_v1"``).

        Returns:
            Filtered DataFrame of run metadata rows. Empty DataFrame if no data.
        """
        if not self._meta_path.exists():
            return pd.DataFrame()

        meta: pd.DataFrame = pd.read_parquet(self._meta_path)
        for col_name, col_value in filters.items():
            if col_name in meta.columns:
                meta = meta[meta[col_name] == col_value]

        return meta
