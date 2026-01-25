from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.stages.silver.future_mbo.book_engine import (
    RADAR_COLUMNS,
    SNAP_COLUMNS,
    WALL_COLUMNS,
    compute_futures_surfaces_1s_from_batches,
)
from src.data_eng.stages.silver.future_mbo.mbo_batches import iter_mbo_batches
from src.data_eng.stages.silver.future_mbo.compute_vacuum_surface_1s import SilverComputeVacuumSurface1s
from src.data_eng.stages.silver.future_mbo.compute_physics_bands_1s import SilverComputePhysicsBands1s

WINDOW_NS = 1_000_000_000
HUD_HISTORY_WINDOWS = 1800
HUD_STREAM_MAX_TICKS = 400

STREAM_COLUMNS: Dict[str, List[str]] = {
    "snap": ["window_end_ts_ns", "mid_price", "spot_ref_price_int", "book_valid"],
    "wall": ["window_end_ts_ns", "rel_ticks", "side", "depth_qty_rest"],
    "vacuum": ["window_end_ts_ns", "rel_ticks", "vacuum_score"],
    "physics": ["window_end_ts_ns", "mid_price", "above_score", "below_score"],
    "gex": [
        "window_end_ts_ns",
        "strike_points",
        "spot_ref_price_int",
        "rel_ticks",
        "underlying_spot_ref",
        "gex_abs",
        "gex",
        "gex_imbalance_ratio",
    ],
}


@dataclass
class HudStreamCache:
    snap: pd.DataFrame
    wall: pd.DataFrame
    vacuum: pd.DataFrame
    radar: pd.DataFrame
    physics: pd.DataFrame
    gex: pd.DataFrame
    window_ids: List[int]
    groups: Dict[str, Dict[int, pd.DataFrame]]
    columns: Dict[str, List[str]]


class HudRingBuffer:
    def __init__(self, max_windows: int, columns: Dict[str, List[str]]) -> None:
        self.max_windows = max_windows
        self.columns = columns
        self.snap = deque(maxlen=max_windows)
        self.wall = deque(maxlen=max_windows)
        self.vacuum = deque(maxlen=max_windows)
        self.radar = deque(maxlen=max_windows)
        self.physics = deque(maxlen=max_windows)
        self.gex = deque(maxlen=max_windows)

    def append(self, snap: pd.DataFrame, wall: pd.DataFrame, vacuum: pd.DataFrame, radar: pd.DataFrame, physics: pd.DataFrame, gex: pd.DataFrame) -> None:
        self.snap.append(_ensure_columns(snap, self.columns["snap"]))
        self.wall.append(_ensure_columns(wall, self.columns["wall"]))
        self.vacuum.append(_ensure_columns(vacuum, self.columns["vacuum"]))
        self.radar.append(_ensure_columns(radar, self.columns["radar"]))
        self.physics.append(_ensure_columns(physics, self.columns["physics"]))
        self.gex.append(_ensure_columns(gex, self.columns["gex"]))

    def frames(self) -> Dict[str, pd.DataFrame]:
        return {
            "snap": _concat(self.snap, self.columns["snap"]),
            "wall": _concat(self.wall, self.columns["wall"]),
            "vacuum": _concat(self.vacuum, self.columns["vacuum"]),
            "radar": _concat(self.radar, self.columns["radar"]),
            "physics": _concat(self.physics, self.columns["physics"]),
            "gex": _concat(self.gex, self.columns["gex"]),
        }


class HudStreamService:
    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, str], HudStreamCache] = {}
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cfg = load_config(self.repo_root, self.repo_root / "src" / "data_eng" / "config" / "datasets.yaml")

    def load_cache(self, symbol: str, dt: str) -> HudStreamCache:
        key = (symbol, dt)
        if key in self.cache:
            return self.cache[key]

        # Define dataset keys
        snap_key = "silver.future_mbo.book_snapshot_1s"
        wall_key = "silver.future_mbo.wall_surface_1s"
        vacuum_key = "silver.future_mbo.vacuum_surface_1s"
        radar_key = "silver.future_mbo.radar_vacuum_1s"
        physics_key = "silver.future_mbo.physics_bands_1s"
        gex_key = "silver.future_option_mbo.gex_surface_1s"

        # Helper to read and enforce
        def _load(ds_key: str) -> pd.DataFrame:
            ref = partition_ref(self.cfg, ds_key, symbol, dt)
            if not is_partition_complete(ref):
                # For GEX or others that might be optional in early testing, we could return empty
                # but for now we expect them to exist if pipeline ran.
                if ds_key == gex_key:  # Gracefully handle missing GEX for now if not run
                    print(f"WARN: Missing {ds_key}, returning empty.")
                    return pd.DataFrame(columns=_columns_for(self.repo_root, self.cfg, ds_key))
                raise FileNotFoundError(f"Input not ready: {ds_key} dt={dt}")
            contract = load_avro_contract(self.repo_root / self.cfg.dataset(ds_key).contract)
            return enforce_contract(read_partition(ref), contract)

        df_snap = _stream_view(_load(snap_key), "snap")
        df_wall = _stream_view(_load(wall_key), "wall")
        df_vacuum = _stream_view(_load(vacuum_key), "vacuum")
        # Radar is intentionally not streamed to the HUD frontend.
        df_radar = pd.DataFrame(columns=_columns_for(self.repo_root, self.cfg, radar_key))
        df_physics = _stream_view(_load(physics_key), "physics")
        df_gex = _stream_view(_load(gex_key), "gex")

        window_ids = df_snap["window_end_ts_ns"].sort_values().unique().astype(int).tolist() if not df_snap.empty else []

        groups = {
            "snap": _group_by_window(df_snap),
            "wall": _group_by_window(df_wall),
            "vacuum": _group_by_window(df_vacuum),
            "radar": _group_by_window(df_radar),
            "physics": _group_by_window(df_physics),
            "gex": _group_by_window(df_gex),
        }

        columns = {
            "snap": STREAM_COLUMNS["snap"],
            "wall": STREAM_COLUMNS["wall"],
            "vacuum": STREAM_COLUMNS["vacuum"],
            "radar": _columns_for(self.repo_root, self.cfg, radar_key),
            "physics": STREAM_COLUMNS["physics"],
            "gex": STREAM_COLUMNS["gex"],
        }

        cache = HudStreamCache(
            snap=df_snap,
            wall=df_wall,
            vacuum=df_vacuum,
            radar=df_radar,
            physics=df_physics,
            gex=df_gex,
            window_ids=window_ids,
            groups=groups,
            columns=columns,
        )
        self.cache[key] = cache
        return cache

    def simulate_stream(self, symbol: str, dt: str, start_ts_ns: int | None = None, speed: float = 1.0) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """
        Yields batches with a simulated delay to mimic real-time cadence.
        speed: Rate multiplier (1.0 = real time, 2.0 = 2x speed, etc.)
        """
        import time
        
        # Pre-load cache
        cache = self.load_cache(symbol, dt)
        if not cache.window_ids:
            return

        # Generator that yields frames with simulated delay
        iterator = self.iter_batches(symbol, dt, start_ts_ns)
        last_window_ts = None
        
        for window_id, batch in iterator:
            if last_window_ts is not None:
                # Calculate delta in data time
                delta_ns = window_id - last_window_ts
                delta_sec = delta_ns / 1_000_000_000.0
                
                # Sleep that amount / speed
                to_sleep = delta_sec / speed
                if to_sleep > 0:
                     time.sleep(to_sleep)
            
            last_window_ts = window_id
            yield window_id, batch

    def bootstrap_frames(self, symbol: str, dt: str, end_ts_ns: int | None = None) -> Dict[str, pd.DataFrame]:
        cache = self.load_cache(symbol, dt)
        if cache.snap.empty:
            return {
                "snap": pd.DataFrame(columns=cache.columns["snap"]),
                "wall": pd.DataFrame(columns=cache.columns["wall"]),
                "vacuum": pd.DataFrame(columns=cache.columns["vacuum"]),
                "radar": pd.DataFrame(columns=cache.columns["radar"]),
                "physics": pd.DataFrame(columns=cache.columns["physics"]),
                "gex": pd.DataFrame(columns=cache.columns["gex"]),
            }

        window_ids = cache.window_ids
        if not window_ids:
            return {
                "snap": pd.DataFrame(columns=cache.columns["snap"]),
                "wall": pd.DataFrame(columns=cache.columns["wall"]),
                "vacuum": pd.DataFrame(columns=cache.columns["vacuum"]),
                "radar": pd.DataFrame(columns=cache.columns["radar"]),
                "physics": pd.DataFrame(columns=cache.columns["physics"]),
                "gex": pd.DataFrame(columns=cache.columns["gex"]),
            }

        if end_ts_ns is None:
            end_ts_ns = window_ids[-1]
        start_ts_ns = end_ts_ns - HUD_HISTORY_WINDOWS * WINDOW_NS

        ring = HudRingBuffer(HUD_HISTORY_WINDOWS, cache.columns)
        for window_id in window_ids:
            if window_id < start_ts_ns or window_id > end_ts_ns:
                continue
            ring.append(
                cache.groups["snap"].get(window_id, pd.DataFrame(columns=cache.columns["snap"])),
                cache.groups["wall"].get(window_id, pd.DataFrame(columns=cache.columns["wall"])),
                cache.groups["vacuum"].get(window_id, pd.DataFrame(columns=cache.columns["vacuum"])),
                cache.groups["radar"].get(window_id, pd.DataFrame(columns=cache.columns["radar"])),
                cache.groups["physics"].get(window_id, pd.DataFrame(columns=cache.columns["physics"])),
                cache.groups["gex"].get(window_id, pd.DataFrame(columns=cache.columns["gex"])),
            )

        frames = ring.frames()
        return {
            "snap": _stream_view(frames["snap"], "snap"),
            "wall": _stream_view(frames["wall"], "wall"),
            "vacuum": _stream_view(frames["vacuum"], "vacuum"),
            "radar": pd.DataFrame(columns=cache.columns["radar"]),
            "physics": _stream_view(frames["physics"], "physics"),
            "gex": _stream_view(frames["gex"], "gex"),
        }

    def iter_batches(self, symbol: str, dt: str, start_ts_ns: int | None = None) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        cache = self.load_cache(symbol, dt)
        ring = HudRingBuffer(HUD_HISTORY_WINDOWS, cache.columns)
        for window_id in cache.window_ids:
            if start_ts_ns is not None and window_id < start_ts_ns:
                continue
            snap = _stream_view(
                cache.groups["snap"].get(window_id, pd.DataFrame(columns=cache.columns["snap"])),
                "snap",
            )
            wall = _stream_view(
                cache.groups["wall"].get(window_id, pd.DataFrame(columns=cache.columns["wall"])),
                "wall",
            )
            vacuum = _stream_view(
                cache.groups["vacuum"].get(window_id, pd.DataFrame(columns=cache.columns["vacuum"])),
                "vacuum",
            )
            physics = _stream_view(
                cache.groups["physics"].get(window_id, pd.DataFrame(columns=cache.columns["physics"])),
                "physics",
            )
            gex = _stream_view(
                cache.groups["gex"].get(window_id, pd.DataFrame(columns=cache.columns["gex"])),
                "gex",
            )
            radar = pd.DataFrame(columns=cache.columns["radar"])
            ring.append(snap, wall, vacuum, radar, physics, gex)
            yield window_id, {
                "snap": snap,
                "wall": wall,
                "vacuum": vacuum,
                "physics": physics,
                "gex": gex,
            }


def _group_by_window(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    grouped = {}
    for window_id, group in df.groupby("window_end_ts_ns"):
        grouped[int(window_id)] = group
    return grouped


def _columns_for(repo_root: Path, cfg, dataset_key: str) -> List[str]:
    contract = load_avro_contract(repo_root / cfg.dataset(dataset_key).contract)
    return contract.fields


def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.loc[:, columns]


def _stream_view(df: pd.DataFrame, surface: str) -> pd.DataFrame:
    columns = STREAM_COLUMNS.get(surface)
    if columns is None:
        return pd.DataFrame(columns=[])
    if df.empty:
        return pd.DataFrame(columns=columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required stream columns for {surface}: {missing}")
    df = df.loc[:, columns]
    if surface in {"wall", "vacuum"} and "rel_ticks" in df.columns:
        df = df.loc[df["rel_ticks"].between(-HUD_STREAM_MAX_TICKS, HUD_STREAM_MAX_TICKS)]
    return df


def _concat(items: deque, columns: List[str]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=columns)
    return pd.concat(list(items), ignore_index=True)
