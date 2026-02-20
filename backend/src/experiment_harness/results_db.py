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
        run_ids = self.append_runs([(meta, results)])
        return run_ids[0]

    def append_runs(
        self,
        runs: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    ) -> list[str]:
        """Append multiple experiment runs in one parquet read/write cycle.

        Args:
            runs: List of ``(meta, results)`` pairs.

        Returns:
            Generated run IDs in the same order as input.
        """
        if not runs:
            return []

        run_ids: list[str] = []
        meta_rows: list[dict[str, Any]] = []
        result_rows: list[dict[str, Any]] = []

        for meta, results in runs:
            run_id = uuid.uuid4().hex[:16]
            run_ids.append(run_id)

            meta_row = dict(meta)
            meta_row["run_id"] = run_id
            meta_row["timestamp_utc"] = datetime.now(tz=timezone.utc).isoformat()
            meta_rows.append(meta_row)

            for row in results:
                result_row = dict(row)
                result_row["run_id"] = run_id
                result_rows.append(result_row)

        new_meta_df = pd.DataFrame(meta_rows)
        if self._meta_path.exists():
            existing_meta = pd.read_parquet(self._meta_path)
            meta_df = pd.concat([existing_meta, new_meta_df], ignore_index=True)
        else:
            meta_df = new_meta_df
        meta_df.to_parquet(self._meta_path, index=False)

        new_runs_df = pd.DataFrame(result_rows)
        if self._runs_path.exists():
            existing_runs = pd.read_parquet(self._runs_path)
            runs_df = pd.concat([existing_runs, new_runs_df], ignore_index=True)
        else:
            runs_df = new_runs_df
        runs_df.to_parquet(self._runs_path, index=False)

        logger.info(
            "Appended %d runs (%d threshold rows)",
            len(runs),
            len(result_rows),
        )
        return run_ids

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
