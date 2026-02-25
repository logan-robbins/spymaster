"""REST routes for experiment browser APIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
from fastapi import FastAPI, Query

from .serving_registry import ServingRegistry


def _safe_float(v: Any) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(v: Any) -> int | None:
    """Convert to int, returning None for NaN/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else int(f)
    except (ValueError, TypeError):
        return None


_STREAM_HTML_PATH = "/stream.html"


def register_experiment_routes(
    app: FastAPI,
    *,
    lake_root: Path,
    serving_registry: ServingRegistry,
) -> None:
    """Register experiment browser REST endpoints."""

    def _build_streaming_url(run_id: str) -> tuple[str | None, bool]:
        """Build stream URL from serving alias registry only."""
        alias_hit = serving_registry.preferred_alias_for_run(run_id)
        if alias_hit is None:
            return None, False
        alias, _serving_id = alias_hit
        return f"{_STREAM_HTML_PATH}?{urlencode({'serving': alias})}", True

    def _results_db():
        from ..experiment_harness.results_db import ResultsDB

        return ResultsDB(lake_root / "research" / "harness" / "results")

    @app.get("/v1/experiments/runs")
    async def experiment_runs(
        signal: str | None = Query(None),
        dataset_id: str | None = Query(None),
        sort: str = Query("tp_rate"),
        min_signals: int = Query(5),
        top_n: int = Query(50),
    ) -> dict:
        """Return ranked experiment results with streaming URLs."""
        db = _results_db()
        best_df = db.query_best(signal=signal, dataset_id=dataset_id, min_signals=min_signals)

        if best_df.empty:
            return {"runs": [], "filters": {"signals": [], "datasets": []}}

        meta_df = db.query_runs()

        signal_params_col = "signal_params_json"
        if signal_params_col in meta_df.columns:
            params_lookup = dict(
                zip(meta_df["run_id"], meta_df[signal_params_col], strict=False)
            )
        else:
            params_lookup = {}
        timestamp_lookup: dict[str, str] = {}
        if "timestamp_utc" in meta_df.columns:
            timestamp_lookup = {
                str(run_id): str(ts)
                for run_id, ts in zip(
                    meta_df["run_id"], meta_df["timestamp_utc"], strict=False
                )
            }

        if sort in best_df.columns:
            ascending = sort not in ("tp_rate", "mean_pnl_ticks", "events_per_hour", "n_signals")
            best_df = best_df.sort_values(sort, ascending=ascending)

        best_df = best_df.head(top_n)

        runs = []
        for _, row in best_df.iterrows():
            run_id = row.get("run_id", "")
            sig_name = row.get("signal_name", "")
            raw_params = params_lookup.get(run_id, "{}")

            streaming_url, can_stream = _build_streaming_url(str(run_id))
            ds_id = row.get("dataset_id", "")

            runs.append({
                "run_id": run_id,
                "signal_name": sig_name,
                "dataset_id": ds_id,
                "experiment_name": row.get("experiment_name", ""),
                "signal_params_json": raw_params if isinstance(raw_params, str) else json.dumps(raw_params),
                "threshold": _safe_float(row.get("threshold")),
                "cooldown_bins": _safe_int(row.get("cooldown_bins")),
                "tp_rate": _safe_float(row.get("tp_rate")),
                "sl_rate": _safe_float(row.get("sl_rate")),
                "timeout_rate": _safe_float(row.get("timeout_rate")),
                "n_signals": _safe_int(row.get("n_signals")),
                "mean_pnl_ticks": _safe_float(row.get("mean_pnl_ticks")),
                "events_per_hour": _safe_float(row.get("events_per_hour")),
                "eval_tp_ticks": _safe_int(row.get("eval_tp_ticks")),
                "eval_sl_ticks": _safe_int(row.get("eval_sl_ticks")),
                "timestamp_utc": timestamp_lookup.get(str(run_id), ""),
                "streaming_url": streaming_url,
                "can_stream": can_stream,
            })

        all_signals = (
            sorted(meta_df["signal_name"].astype(str).unique().tolist())
            if "signal_name" in meta_df.columns
            else []
        )
        all_datasets = (
            sorted(meta_df["dataset_id"].astype(str).unique().tolist())
            if "dataset_id" in meta_df.columns
            else []
        )

        return {
            "runs": runs,
            "filters": {
                "signals": all_signals,
                "datasets": all_datasets,
            },
        }

    @app.get("/v1/experiments/runs/{run_id}/detail")
    async def experiment_run_detail(run_id: str) -> dict:
        """Return full detail for one experiment run."""
        db = _results_db()
        meta_df = db.query_runs(run_id=run_id)
        if meta_df.empty:
            return {"error": f"Run {run_id} not found"}

        meta_row = meta_df.iloc[0].to_dict()
        for k, v in meta_row.items():
            if isinstance(v, float) and pd.isna(v):
                meta_row[k] = None

        raw_params = meta_row.get("signal_params_json", "{}")
        try:
            sig_params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
        except (json.JSONDecodeError, TypeError):
            sig_params = {}

        runs_path = lake_root / "research" / "harness" / "results" / "runs.parquet"
        threshold_results = []
        if runs_path.exists():
            runs_df = pd.read_parquet(runs_path)
            run_rows = runs_df[runs_df["run_id"] == run_id]
            for _, r in run_rows.iterrows():
                row_dict = {}
                for c in r.index:
                    val = r[c]
                    if isinstance(val, float) and pd.isna(val):
                        row_dict[c] = None
                    elif hasattr(val, "item"):
                        row_dict[c] = val.item()
                    else:
                        row_dict[c] = val
                threshold_results.append(row_dict)

        streaming_url, can_stream = _build_streaming_url(run_id)

        meta_row["signal_params"] = sig_params
        for k, v in meta_row.items():
            if hasattr(v, "item"):
                meta_row[k] = v.item()

        return {
            "meta": meta_row,
            "threshold_results": threshold_results,
            "streaming_url": streaming_url,
            "can_stream": can_stream,
        }
