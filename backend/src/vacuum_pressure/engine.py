"""Vacuum & Pressure Computation Engine.

Reads silver parquet data, computes vacuum / pressure metrics, and
provides both batch and streaming interfaces for downstream consumers.

Supports both ``equity_mbo`` and ``future_mbo`` product types via
runtime configuration from :mod:`vacuum_pressure.config`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .config import VPRuntimeConfig
from .formulas import GoldSignalConfig, run_full_pipeline

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Silver path construction
# ──────────────────────────────────────────────────────────────────────

SILVER_PATH_TPL = (
    "silver/product_type={product_type}/symbol={symbol}"
    "/table={table}/dt={dt}"
)

REQUIRED_TABLES: List[str] = ["book_snapshot_1s", "depth_and_flow_1s"]
"""Both silver tables required for vacuum-pressure computation."""


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


def _runner_command(product_type: str, symbol: str, dt: str) -> str:
    """Build the exact runner command to produce missing silver data."""
    return (
        f"cd backend && uv run python -m src.data_eng.runner "
        f"--product-type {product_type} --layer silver "
        f"--symbol {symbol} --dt {dt} --workers 4"
    )


def _read_silver_parquet(
    partition_dir: Path,
    product_type: str,
    symbol: str,
    dt: str,
) -> pd.DataFrame:
    """Read all ``part-*.parquet`` files in a silver partition directory.

    Args:
        partition_dir: Partition directory path.
        product_type: Product type for error messaging.
        symbol: Symbol for error messaging.
        dt: Date for error messaging.

    Returns:
        Concatenated DataFrame from all parquet files.

    Raises:
        FileNotFoundError: If the directory or parquet files are missing,
            with the exact missing path and the runner command to produce it.
    """
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"Silver partition not found: {partition_dir}\n"
            f"Run the pipeline first:\n"
            f"  {_runner_command(product_type, symbol, dt)}"
        )

    parquet_files = sorted(partition_dir.glob("part-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files in partition: {partition_dir}\n"
            f"Run the pipeline first:\n"
            f"  {_runner_command(product_type, symbol, dt)}"
        )

    dfs = [pd.read_parquet(f) for f in parquet_files]
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────
# Readiness validation (4.7)
# ──────────────────────────────────────────────────────────────────────


def validate_silver_readiness(
    lake_root: Path,
    config: VPRuntimeConfig,
    dt: str,
) -> Dict[str, int]:
    """Validate that both required silver tables exist and return row counts.

    Args:
        lake_root: Lake root directory.
        config: Resolved runtime config.
        dt: Date string.

    Returns:
        Dict mapping table name to row count.

    Raises:
        FileNotFoundError: If any required silver table is missing.
    """
    row_counts: Dict[str, int] = {}
    for table in REQUIRED_TABLES:
        partition_dir = _silver_dir(
            lake_root, config.product_type, config.symbol, table, dt,
        )
        if not partition_dir.exists():
            raise FileNotFoundError(
                f"Silver table missing: {partition_dir}\n"
                f"Run the pipeline first:\n"
                f"  {_runner_command(config.product_type, config.symbol, dt)}"
            )
        parquet_files = sorted(partition_dir.glob("part-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files in: {partition_dir}\n"
                f"Run the pipeline first:\n"
                f"  {_runner_command(config.product_type, config.symbol, dt)}"
            )
        total_rows = sum(
            pd.read_parquet(f).shape[0] for f in parquet_files
        )
        row_counts[table] = total_rows

    return row_counts


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
        # Cache key: VPRuntimeConfig.cache_key(dt) -> (snap, flow_enriched, signals)
        self._cache: Dict[
            str,
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ] = {}

    @staticmethod
    def _cache_key(
        config: VPRuntimeConfig,
        dt: str,
        gold_signal_config: GoldSignalConfig | None,
    ) -> str:
        """Build cache key including optional gold-layer signal parameters."""
        base_key = config.cache_key(dt)
        if gold_signal_config is None:
            return base_key
        gold_signal_config.validate()
        return f"{base_key}:gold={gold_signal_config.cache_fragment()}"

    # ── Data loading ──────────────────────────────────────────────

    def load_silver(
        self,
        config: VPRuntimeConfig,
        dt: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load silver ``book_snapshot_1s`` and ``depth_and_flow_1s``.

        Uses runtime config to select the correct product_type path.

        Args:
            config: Resolved runtime config.
            dt: Date string ``YYYY-MM-DD``.

        Returns:
            ``(df_snap, df_flow)`` DataFrames.

        Raises:
            FileNotFoundError: If silver partitions are missing.
        """
        snap_dir = _silver_dir(
            self.lake_root, config.product_type, config.symbol,
            "book_snapshot_1s", dt,
        )
        flow_dir = _silver_dir(
            self.lake_root, config.product_type, config.symbol,
            "depth_and_flow_1s", dt,
        )

        logger.info("Loading silver book_snapshot_1s from %s", snap_dir)
        df_snap = _read_silver_parquet(
            snap_dir, config.product_type, config.symbol, dt,
        )
        logger.info("  book_snapshot_1s: %d rows", len(df_snap))

        logger.info("Loading silver depth_and_flow_1s from %s", flow_dir)
        df_flow = _read_silver_parquet(
            flow_dir, config.product_type, config.symbol, dt,
        )
        logger.info("  depth_and_flow_1s: %d rows", len(df_flow))

        return df_snap, df_flow

    # ── Batch computation ─────────────────────────────────────────

    def compute_day(
        self,
        config: VPRuntimeConfig,
        dt: str,
        gold_signal_config: GoldSignalConfig | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute all vacuum / pressure metrics for a config / date.

        Results are cached by ``config.cache_key(dt)``: subsequent calls
        for the same key return instantly.

        Args:
            config: Resolved runtime config.
            dt: Date string.

        Returns:
            ``(df_snap, df_flow_enriched, df_signals)``

            * *df_snap*: Book snapshot (1 row per window).
            * *df_flow_enriched*: Per-bucket depth / flow with scores.
            * *df_signals*: Aggregated signals with derivatives (1 per window).
        """
        key = self._cache_key(config, dt, gold_signal_config)
        if key in self._cache:
            return self._cache[key]

        df_snap, df_flow = self.load_silver(config, dt)
        df_signals, df_flow_enriched = run_full_pipeline(
            df_flow, df_snap, config, gold_signal_config=gold_signal_config,
        )

        logger.info(
            "Computed %d windows of vacuum/pressure signals for %s (%s) on %s",
            len(df_signals), config.symbol, config.product_type, dt,
        )

        self._cache[key] = (df_snap, df_flow_enriched, df_signals)
        return df_snap, df_flow_enriched, df_signals

    # ── Streaming interface ───────────────────────────────────────

    def iter_windows(
        self,
        config: VPRuntimeConfig,
        dt: str,
        start_ts_ns: int | None = None,
        gold_signal_config: GoldSignalConfig | None = None,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Iterate over windows for streaming.

        Each yield produces ``(window_end_ts_ns, surfaces_dict)`` where
        *surfaces_dict* has keys ``"snap"``, ``"flow"``, ``"signals"``.

        Args:
            config: Resolved runtime config.
            dt: Date string.
            start_ts_ns: Optional start timestamp to skip to.

        Yields:
            ``(window_end_ts_ns, {"snap": df, "flow": df, "signals": df})``
        """
        df_snap, df_flow_enriched, df_signals = self.compute_day(
            config,
            dt,
            gold_signal_config=gold_signal_config,
        )
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
        self,
        config: VPRuntimeConfig,
        dt: str,
        output_dir: Path,
        gold_signal_config: GoldSignalConfig | None = None,
    ) -> Path:
        """Compute and save signals to parquet.

        Args:
            config: Resolved runtime config.
            dt: Date string.
            output_dir: Target directory.

        Returns:
            Path to the written parquet file.
        """
        _, _, df_signals = self.compute_day(
            config,
            dt,
            gold_signal_config=gold_signal_config,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            output_dir
            / f"vacuum_pressure_{config.product_type}_{config.symbol}_{dt}.parquet"
        )
        df_signals.to_parquet(
            out_path, engine="pyarrow", compression="zstd", index=False,
        )
        logger.info("Saved signals to %s (%d rows)", out_path, len(df_signals))
        return out_path
