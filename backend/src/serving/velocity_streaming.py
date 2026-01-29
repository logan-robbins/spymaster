"""Minimal streaming service for frontend2 velocity visualization.

Serves snap + velocity surfaces. Snap always comes from futures to anchor the grid.
Velocity can come from futures or options depending on product_type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.retrieval.mbo_contract_day_selector import load_selection

WINDOW_NS = 1_000_000_000
MAX_HISTORY_WINDOWS = 1800
MAX_TICKS = 400

SNAP_COLUMNS = ["window_end_ts_ns", "mid_price", "spot_ref_price_int", "book_valid"]
VELOCITY_COLUMNS = ["window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "side", "liquidity_velocity"]


@dataclass
class VelocityStreamCache:
    snap: pd.DataFrame
    velocity: pd.DataFrame
    window_ids: List[int]
    snap_by_window: Dict[int, pd.DataFrame]
    velocity_by_window: Dict[int, pd.DataFrame]


class VelocityStreamService:
    """Minimal streaming service for velocity visualization."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, str, str], VelocityStreamCache] = {}
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cfg = load_config(self.repo_root, self.repo_root / "src" / "data_eng" / "config" / "datasets.yaml")

    def load_cache(self, symbol: str, dt: str, product_type: str = "future_mbo") -> VelocityStreamCache:
        resolved_symbol = _resolve_contract_symbol(self.repo_root, symbol, dt)
        key = (product_type, resolved_symbol, dt)
        if key in self.cache:
            return self.cache[key]

        if product_type not in {"future_mbo", "future_option_mbo"}:
            raise ValueError(f"Unsupported product_type: {product_type}")

        snap_key = "silver.future_mbo.book_snapshot_1s"
        if product_type == "future_mbo":
            velocity_key = "gold.future_mbo.physics_surface_1s"
        else:
            velocity_key = "gold.future_option_mbo.physics_surface_1s"

        def _load(ds_key: str) -> pd.DataFrame:
            ref = partition_ref(self.cfg, ds_key, resolved_symbol, dt)
            if not is_partition_complete(ref):
                raise FileNotFoundError(
                    f"Input not ready: {ds_key} symbol={resolved_symbol} dt={dt}"
                )
            contract = load_avro_contract(self.repo_root / self.cfg.dataset(ds_key).contract)
            return enforce_contract(read_partition(ref), contract)

        df_snap = _load(snap_key)
        df_velocity = _load(velocity_key)

        # Filter snap to required columns
        df_snap = df_snap.loc[:, SNAP_COLUMNS].copy()

        # Filter velocity to required columns and tick range
        df_velocity = df_velocity.loc[:, VELOCITY_COLUMNS].copy()
        if product_type == "future_option_mbo":
            df_velocity = (
                df_velocity.groupby(
                    ["window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "side"],
                    as_index=False,
                )
                .agg(liquidity_velocity=("liquidity_velocity", "sum"))
            )
        df_velocity = df_velocity.loc[df_velocity["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)]
        if product_type == "future_option_mbo":
            if (df_velocity["rel_ticks"] % 20 != 0).any():
                raise ValueError("Option velocity rel_ticks not aligned to $5 grid")

        # Get window IDs from snap (reference time series)
        window_ids = df_snap["window_end_ts_ns"].sort_values().unique().astype(int).tolist()

        # Group by window
        snap_by_window = _group_by_window(df_snap)
        velocity_by_window = _group_by_window(df_velocity)

        cache = VelocityStreamCache(
            snap=df_snap,
            velocity=df_velocity,
            window_ids=window_ids,
            snap_by_window=snap_by_window,
            velocity_by_window=velocity_by_window,
        )
        self.cache[key] = cache
        return cache

    def iter_batches(
        self,
        symbol: str,
        dt: str,
        product_type: str = "future_mbo",
        start_ts_ns: int | None = None,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Iterate over time windows yielding snap + velocity batches."""
        cache = self.load_cache(symbol, dt, product_type)

        for window_id in cache.window_ids:
            if start_ts_ns is not None and window_id < start_ts_ns:
                continue

            snap_df = cache.snap_by_window.get(window_id, pd.DataFrame(columns=SNAP_COLUMNS))
            velocity_df = cache.velocity_by_window.get(window_id, pd.DataFrame(columns=VELOCITY_COLUMNS))

            yield window_id, {
                "snap": snap_df,
                "velocity": velocity_df,
            }

    def simulate_stream(
        self,
        symbol: str,
        dt: str,
        product_type: str = "future_mbo",
        start_ts_ns: int | None = None,
        speed: float = 1.0,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Yield batches with simulated delay."""
        import time

        cache = self.load_cache(symbol, dt, product_type)
        if not cache.window_ids:
            return

        iterator = self.iter_batches(symbol, dt, product_type, start_ts_ns)
        last_window_ts = None

        for window_id, batch in iterator:
            if last_window_ts is not None:
                delta_ns = window_id - last_window_ts
                delta_sec = delta_ns / 1_000_000_000.0
                to_sleep = delta_sec / speed
                if to_sleep > 0:
                    time.sleep(to_sleep)

            last_window_ts = window_id
            yield window_id, batch


def _group_by_window(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    grouped = {}
    for window_id, group in df.groupby("window_end_ts_ns"):
        grouped[int(window_id)] = group
    return grouped


def _resolve_contract_symbol(repo_root: Path, symbol: str, dt: str) -> str:
    if symbol != "ES":
        return symbol
    selection_path = repo_root / "lake" / "selection" / "mbo_contract_day_selection.parquet"
    df = load_selection(selection_path)
    df["session_date"] = df["session_date"].astype(str)
    row = df.loc[df["session_date"] == dt]
    if row.empty:
        raise ValueError(f"No selection map entry for dt={dt}")
    selected = str(row.iloc[0]["selected_symbol"]).strip()
    if not selected:
        raise ValueError(f"Selection map has empty symbol for dt={dt}")
    return selected
