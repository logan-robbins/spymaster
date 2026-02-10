"""Unified streaming service for frontend2 visualization.

Serves three surfaces in a single stream:
- snap: spot reference from futures (anchors the grid)
- velocity: futures liquidity velocity at $0.25 tick resolution
- options: aggregated options composite fields at $5 strike resolution

Options are aggregated across Call/Put and Bid/Ask to show NET liquidity
velocity at each strike level - this captures the aggregate market maker
positioning pressure at each strike.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from src.data_eng.config import ProductConfig, load_config
from src.data_eng.contracts import enforce_contract, load_avro_contract
from src.data_eng.io import is_partition_complete, partition_ref, read_partition
from src.data_eng.mbo_contract_day_selector import load_selection
from src.serving.config import settings
from src.serving.forecast_math import build_window_fields, run_forecast, tick_index_from_price

WINDOW_NS = 1_000_000_000
MAX_HISTORY_WINDOWS = 1800
MAX_TICKS = 400

SNAP_COLUMNS = ["window_end_ts_ns", "mid_price", "spot_ref_price_int", "book_valid"]
VELOCITY_STREAM_COLUMNS = [
    "window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "side", 
    "liquidity_velocity", "rho", "nu", "kappa", "pressure_grad", "u_wave_energy", "Omega"
]
VELOCITY_FORECAST_COLUMNS = VELOCITY_STREAM_COLUMNS + ["u_near", "u_p_slow"]
VELOCITY_COLUMNS = VELOCITY_STREAM_COLUMNS
OPTIONS_COLUMNS = [
    "window_end_ts_ns",
    "spot_ref_price_int",
    "rel_ticks",
    "liquidity_velocity",
    "pressure_grad",
    "u_wave_energy",
    "nu",
    "Omega",
]
FORECAST_COLUMNS = [
    "window_end_ts_ns", "horizon_s", "predicted_spot_tick", "predicted_tick_delta", 
    "confidence", "RunScore_up", "RunScore_down", "D_up", "D_down"
]

@dataclass
class UnifiedStreamCache:
    """Cache for unified streaming: futures snap + velocity + options + forecast."""
    snap: pd.DataFrame
    velocity: pd.DataFrame
    options: pd.DataFrame
    forecast: pd.DataFrame
    window_ids: List[int]
    snap_by_window: Dict[int, pd.DataFrame]
    velocity_by_window: Dict[int, pd.DataFrame]
    options_by_window: Dict[int, pd.DataFrame]
    forecast_by_window: Dict[int, pd.DataFrame]


class ForecastEngine:
    """Stream-time forecasting engine using deterministic physics fields."""

    def __init__(self, beta: float, gamma: float, horizon_s: int = 30) -> None:
        self.horizon_s = horizon_s
        self.beta = float(beta)
        self.gamma = float(gamma)

    def run_batch(self, df_velocity: pd.DataFrame, df_options: pd.DataFrame) -> pd.DataFrame:
        results: list[dict] = []
        if df_velocity.empty:
            return pd.DataFrame(results)

        options_by_window = {k: g for k, g in df_options.groupby("window_end_ts_ns")}
        window_ids = sorted(df_velocity["window_end_ts_ns"].unique())

        last_spot_ticks: float | None = None

        for window_id in window_ids:
            group = df_velocity.loc[df_velocity["window_end_ts_ns"] == window_id]
            if group.empty:
                continue
            spot_ref = group["spot_ref_price_int"].iloc[0]
            spot_ticks = tick_index_from_price(spot_ref)
            v0 = 0.0 if last_spot_ticks is None else (spot_ticks - last_spot_ticks)
            last_spot_ticks = spot_ticks

            opt_group = options_by_window.get(window_id, pd.DataFrame())
            fields = build_window_fields(group, opt_group)
            forecast_rows, diagnostics = run_forecast(
                fields=fields,
                spot_ticks=spot_ticks,
                v0=v0,
                beta=self.beta,
                gamma=self.gamma,
                horizon_s=self.horizon_s,
            )

            results.append(
                {
                    "window_end_ts_ns": int(window_id),
                    "horizon_s": 0,
                    "predicted_spot_tick": None,
                    "predicted_tick_delta": None,
                    "confidence": None,
                    "RunScore_up": diagnostics.run_score_up,
                    "RunScore_down": diagnostics.run_score_down,
                    "D_up": diagnostics.d_up,
                    "D_down": diagnostics.d_down,
                }
            )

            for row in forecast_rows:
                results.append(
                    {
                        "window_end_ts_ns": int(window_id),
                        "horizon_s": int(row["horizon_s"]),
                        "predicted_spot_tick": int(row["predicted_spot_tick"]),
                        "predicted_tick_delta": int(row["predicted_tick_delta"]),
                        "confidence": float(row["confidence"]),
                        "RunScore_up": None,
                        "RunScore_down": None,
                        "D_up": None,
                        "D_down": None,
                    }
                )

        df_out = pd.DataFrame(results)
        if not df_out.empty:
            df_out = df_out.sort_values(["window_end_ts_ns", "horizon_s"])
        return df_out


class VelocityStreamService:
    """Unified streaming service for futures + options visualization."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, str], UnifiedStreamCache] = {}
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cfg = load_config(self.repo_root, self.repo_root / "src" / "data_eng" / "config" / "datasets.yaml")
        self.beta, self.gamma = _load_physics_params(settings.physics_params_path)

    def product_meta(self, symbol: str) -> dict:
        """Return product metadata dict for WebSocket batch_start messages."""
        try:
            pc = self.cfg.product_for_symbol(symbol)
            return {
                "tick_size": pc.tick_size,
                "tick_int": pc.tick_int,
                "strike_ticks": pc.strike_ticks,
                "grid_max_ticks": pc.grid_max_ticks,
            }
        except (ValueError, KeyError):
            return {
                "tick_size": 0.25,
                "tick_int": 250_000_000,
                "strike_ticks": 20,
                "grid_max_ticks": 200,
            }

    def load_cache(self, symbol: str, dt: str) -> UnifiedStreamCache:
        """Load unified cache with futures snap, futures velocity, options velocity, and forecast."""
        resolved_symbol = _resolve_contract_symbol(self.repo_root, symbol, dt)
        key = (resolved_symbol, dt)
        if key in self.cache:
            return self.cache[key]

        snap_key = "silver.future_mbo.book_snapshot_1s"
        velocity_key = "gold.future_mbo.physics_surface_1s"
        options_key = "gold.future_option_mbo.physics_surface_1s"

        def _load(ds_key: str) -> pd.DataFrame:
            ref = partition_ref(self.cfg, ds_key, resolved_symbol, dt)
            if not is_partition_complete(ref):
                raise FileNotFoundError(
                    f"Input not ready: {ds_key} symbol={resolved_symbol} dt={dt}"
                )
            contract = load_avro_contract(self.repo_root / self.cfg.dataset(ds_key).contract)
            return enforce_contract(read_partition(ref), contract)

        # Load futures snap
        df_snap = _load(snap_key)
        df_snap = df_snap.loc[:, SNAP_COLUMNS].copy()

        # Load futures velocity
        df_velocity_full = _load(velocity_key)
        avail_cols = [c for c in VELOCITY_FORECAST_COLUMNS if c in df_velocity_full.columns]
        df_velocity_full = df_velocity_full.loc[:, avail_cols].copy()

        for c in VELOCITY_FORECAST_COLUMNS:
            if c not in df_velocity_full.columns:
                df_velocity_full[c] = 0.0

        df_velocity_full = df_velocity_full.loc[
            df_velocity_full["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)
        ]
        df_velocity_stream = df_velocity_full.loc[:, VELOCITY_STREAM_COLUMNS].copy()

        # Load options velocity and aggregate across C/P and A/B
        df_options_raw = _load(options_key)
        if "Omega_opt" not in df_options_raw.columns:
            df_options_raw["Omega_opt"] = 0.0
        df_options_raw = df_options_raw.loc[df_options_raw["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)]
        df_options = _aggregate_options(df_options_raw)
        df_options = df_options.loc[df_options["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)]

        # Run Forecast
        engine = ForecastEngine(beta=self.beta, gamma=self.gamma)
        df_forecast = engine.run_batch(df_velocity_full, df_options_raw)

        # Get window IDs from snap (reference time series)
        window_ids = df_snap["window_end_ts_ns"].sort_values().unique().astype(int).tolist()

        # Group by window
        snap_by_window = _group_by_window(df_snap)
        velocity_by_window = _group_by_window(df_velocity_stream)
        options_by_window = _group_by_window(df_options)
        forecast_by_window = _group_by_window(df_forecast)

        cache = UnifiedStreamCache(
            snap=df_snap,
            velocity=df_velocity_stream,
            options=df_options,
            forecast=df_forecast,
            window_ids=window_ids,
            snap_by_window=snap_by_window,
            velocity_by_window=velocity_by_window,
            options_by_window=options_by_window,
            forecast_by_window=forecast_by_window,
        )
        self.cache[key] = cache
        return cache

    def iter_batches(
        self,
        symbol: str,
        dt: str,
        start_ts_ns: int | None = None,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Iterate over time windows yielding unified snap + velocity + options + forecast batches."""
        cache = self.load_cache(symbol, dt)

        for window_id in cache.window_ids:
            if start_ts_ns is not None and window_id < start_ts_ns:
                continue

            snap_df = cache.snap_by_window.get(window_id, pd.DataFrame(columns=SNAP_COLUMNS))
            velocity_df = cache.velocity_by_window.get(window_id, pd.DataFrame(columns=VELOCITY_STREAM_COLUMNS))
            options_df = cache.options_by_window.get(window_id, pd.DataFrame(columns=OPTIONS_COLUMNS))
            forecast_df = cache.forecast_by_window.get(window_id, pd.DataFrame(columns=FORECAST_COLUMNS))

            yield window_id, {
                "snap": snap_df,
                "velocity": velocity_df,
                "options": options_df,
                "forecast": forecast_df,
            }

    def simulate_stream(
        self,
        symbol: str,
        dt: str,
        start_ts_ns: int | None = None,
        speed: float = 1.0,
    ) -> Iterable[Tuple[int, Dict[str, pd.DataFrame]]]:
        """Yield batches with simulated delay."""
        import time

        cache = self.load_cache(symbol, dt)
        if not cache.window_ids:
            return

        iterator = self.iter_batches(symbol, dt, start_ts_ns)
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


def _aggregate_options(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate options across Call/Put and Bid/Ask to get NET velocity per strike.
    
    From a quant perspective: we care about aggregate liquidity pressure at each
    strike level, not the C/P or A/B breakdown. Net positive = MM adding depth
    (strike becomes "heavier", harder for spot to pass). Net negative = MM pulling
    (strike becomes "lighter", easier to breach).
    """
    if df.empty:
        return pd.DataFrame(columns=OPTIONS_COLUMNS)

    # Ensure optional physics columns exist for backward compatibility
    for col in ("pressure_grad_opt", "u_opt_wave_energy", "nu_opt", "Omega_opt"):
        if col not in df.columns:
            df[col] = 0.0

    # Aggregate across right (C/P) and side (A/B)
    df_agg = (
        df.groupby(
            ["window_end_ts_ns", "spot_ref_price_int", "rel_ticks"],
            as_index=False,
        )
        .agg(
            liquidity_velocity=("liquidity_velocity", "sum"),
            pressure_grad=("pressure_grad_opt", "sum"),
            u_wave_energy=("u_opt_wave_energy", "sum"),
            nu=("nu_opt", "sum"),
            Omega=("Omega_opt", "max"),
        )
    )
    return df_agg[OPTIONS_COLUMNS]


def _group_by_window(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    grouped = {}
    for window_id, group in df.groupby("window_end_ts_ns"):
        grouped[int(window_id)] = group
    return grouped


def _load_physics_params(path: Path) -> Tuple[float, float]:
    if not path.exists():
        raise FileNotFoundError(f"Physics params not found: {path}")
    payload = json.loads(path.read_text())
    if "beta" not in payload or "gamma" not in payload:
        raise ValueError(f"Physics params missing beta/gamma: {path}")
    return float(payload["beta"]), float(payload["gamma"])


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
