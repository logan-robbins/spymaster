"""Vacuum & Pressure Computation Engine.

Reads silver parquet data, computes vacuum / pressure metrics, and
provides both batch and streaming interfaces for downstream consumers.

This engine is **standalone** — it reads silver parquet files directly
without depending on the data_eng config system.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from .formulas import run_full_pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Silver path construction
# ──────────────────────────────────────────────────────────────────────

SILVER_PATH_TPL = (
    "silver/product_type={product_type}/symbol={symbol}"
    "/table={table}/dt={dt}"
)


def _silver_dir(
    lake_root: Path,
    product_type: str,
    symbol: str,
    table: str,
    dt: str,
) -> Path:
    """Construct silver partition directory path."""
    return lake_root / SILVER_PATH_TPL.format(
        product_type=product_type,
        symbol=symbol,
        table=table,
        dt=dt,
    )


def _read_silver_parquet(partition_dir: Path) -> pd.DataFrame:
    """Read all ``part-*.parquet`` files in a silver partition directory.

    Args:
        partition_dir: Partition directory path.

    Returns:
        Concatenated DataFrame from all parquet files.

    Raises:
        FileNotFoundError: If the directory or parquet files are missing.
    """
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Silver partition not found: {partition_dir}\n"
            f"Run the pipeline first:\n"
            f"  cd backend && uv run python -m src.data_eng.runner "
            f"--product-type equity_mbo --layer silver "
            f"--symbol QQQ --dt YYYY-MM-DD --workers 4"
        )

    parquet_files = sorted(partition_dir.glob("part-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files in partition: {partition_dir}"
        )

    dfs = [pd.read_parquet(f) for f in parquet_files]
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────


class VacuumPressureEngine:
    """Read silver data, compute vacuum / pressure metrics, serve results.

    Attributes:
        lake_root: Path to the lake directory (e.g. ``backend/lake``).
    """

    def __init__(self, lake_root: Path) -> None:
        self.lake_root = lake_root
        self._cache: Dict[
            Tuple[str, str],
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ] = {}

    # ── Data loading ──────────────────────────────────────────────

    def load_silver(
        self, symbol: str, dt: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load silver ``book_snapshot_1s`` and ``depth_and_flow_1s``.

        Args:
            symbol: Ticker symbol (e.g. ``"QQQ"``).
            dt: Date string ``YYYY-MM-DD``.

        Returns:
            ``(df_snap, df_flow)`` DataFrames.
        """
        snap_dir = _silver_dir(
            self.lake_root, "equity_mbo", symbol, "book_snapshot_1s", dt,
        )
        flow_dir = _silver_dir(
            self.lake_root, "equity_mbo", symbol, "depth_and_flow_1s", dt,
        )

        logger.info("Loading silver book_snapshot_1s from %s", snap_dir)
        df_snap = _read_silver_parquet(snap_dir)
        logger.info("  → %d rows", len(df_snap))

        logger.info("Loading silver depth_and_flow_1s from %s", flow_dir)
        df_flow = _read_silver_parquet(flow_dir)
        logger.info("  → %d rows", len(df_flow))

        return df_snap, df_flow

    # ── Batch computation ─────────────────────────────────────────

    def compute_day(
        self, symbol: str, dt: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute all vacuum / pressure metrics for a symbol / date.

        Results are cached: subsequent calls for the same key return
        instantly.

        Args:
            symbol: Ticker symbol.
            dt: Date string.

        Returns:
            ``(df_snap, df_flow_enriched, df_signals)``

            * *df_snap*: Book snapshot (1 row per window).
            * *df_flow_enriched*: Per-bucket depth / flow with scores.
            * *df_signals*: Aggregated signals with derivatives (1 per window).
        """
        key = (symbol, dt)
        if key in self._cache:
            return self._cache[key]

        df_snap, df_flow = self.load_silver(symbol, dt)
        df_signals, df_flow_enriched = run_full_pipeline(df_flow, df_snap)

        logger.info(
            "Computed %d windows of vacuum/pressure signals for %s on %s",
            len(df_signals), symbol, dt,
        )

        self._cache[key] = (df_snap, df_flow_enriched, df_signals)
        return df_snap, df_flow_enriched, df_signals

    # ── Streaming interface ───────────────────────────────────────

    def iter_windows(
        self,
        symbol: str,
        dt: str,
        start_ts_ns: int | None = None,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Iterate over windows for streaming.

        Each yield produces ``(window_end_ts_ns, surfaces_dict)`` where
        *surfaces_dict* has keys ``"snap"``, ``"flow"``, ``"signals"``.

        Args:
            symbol: Ticker symbol.
            dt: Date string.
            start_ts_ns: Optional start timestamp to skip to.

        Yields:
            ``(window_end_ts_ns, {"snap": df, "flow": df, "signals": df})``
        """
        df_snap, df_flow_enriched, df_signals = self.compute_day(symbol, dt)
        if df_signals.empty:
            return

        # Pre-group by window for O(1) lookup
        flow_by_w: Dict[int, pd.DataFrame] = {
            int(wid): grp
            for wid, grp in df_flow_enriched.groupby("window_end_ts_ns")
        }
        snap_by_w: Dict[int, pd.DataFrame] = {
            int(wid): grp
            for wid, grp in df_snap.groupby("window_end_ts_ns")
        }
        sig_by_w: Dict[int, pd.DataFrame] = {
            int(wid): grp
            for wid, grp in df_signals.groupby("window_end_ts_ns")
        }

        window_ids = sorted(
            df_signals["window_end_ts_ns"].unique().astype(int).tolist()
        )

        for wid in window_ids:
            if start_ts_ns is not None and wid < start_ts_ns:
                continue

            yield wid, {
                "snap": snap_by_w.get(wid, pd.DataFrame()),
                "flow": flow_by_w.get(wid, pd.DataFrame()),
                "signals": sig_by_w.get(wid, pd.DataFrame()),
            }

    # ── Persist to parquet ────────────────────────────────────────

    def save_signals(
        self, symbol: str, dt: str, output_dir: Path,
    ) -> Path:
        """Compute and save signals to parquet.

        Args:
            symbol: Ticker symbol.
            dt: Date string.
            output_dir: Target directory.

        Returns:
            Path to the written parquet file.
        """
        _, _, df_signals = self.compute_day(symbol, dt)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"vacuum_pressure_{symbol}_{dt}.parquet"
        df_signals.to_parquet(
            out_path, engine="pyarrow", compression="zstd", index=False,
        )
        logger.info("Saved signals to %s (%d rows)", out_path, len(df_signals))
        return out_path
