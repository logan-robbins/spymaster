"""Unified streaming service for frontend2 visualization.

Serves three surfaces in a single stream:
- snap: spot reference from futures (anchors the grid)
- velocity: futures liquidity velocity at $0.25 tick resolution
- options: aggregated options liquidity velocity at $5 strike resolution

Options are aggregated across Call/Put and Bid/Ask to show NET liquidity
velocity at each strike level - this captures the aggregate market maker
positioning pressure at each strike.
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
VELOCITY_COLUMNS = [
    "window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "side", 
    "liquidity_velocity", "rho", "nu", "kappa", "pressure_grad", "u_wave_energy", "Omega"
]
OPTIONS_COLUMNS = ["window_end_ts_ns", "spot_ref_price_int", "rel_ticks", "liquidity_velocity"]
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
    """Stream-time forecasting engine using physics fields."""
    
    def __init__(self, horizon_s: int = 30):
        self.horizon_s = horizon_s
        self.dt = 1.0
        self.n_particles = 1000
        
    def run_batch(self, df_velocity: pd.DataFrame) -> pd.DataFrame:
        """Run forecast for all windows in the batch (vectorized by window if possible, or loop)."""
        import numpy as np
        
        results = []
        
        # We process window by window to simulate streaming behavior and easier logic
        # Optimize: could group by window and apply
        for window_id, group in df_velocity.groupby("window_end_ts_ns"):
            # group has rows for rel_ticks -200..200 (dense)
            # Need to constructing 1D arrays for physics fields
            
            # Sort by rel_ticks
            # We assume side='B' and 'A' are both present.
            # Physics fields like 'pressure_grad' might differ by side or be unified?
            # Gold compute produces pressure_grad per side.
            # Spec: "pressure_grad ... +u_p for Bid, -u_p for Ask".
            # For particle sim, we need a SINGLE vector field v_eff(x).
            # We need to aggregating Bid/Ask fields into a single field?
            # Or use the "dominant" side?
            # Usually we allow particles to move continuously.
            # Combining: take the side that corresponds to position?
            # If x < 0, use Bid physics? If x > 0 use Ask?
            # Or average?
            # Let's simple average for now or just take the Bid side which covers -200..200?
            # Wait, Gold stage output has `side` column.
            # Does Bid side cover positive ticks?
            # `u_near` was smoothed over -200..200.
            # So Bid side has values everywhere.
            # In Gold, `pressure_grad` for Bid is `+u_p`. For Ask `+u_p` (negative sign handled?). 
            # Wait, in Gold I did: `np.where(side_cond, u_p, -u_p)`.
            # So `pressure_grad` is signed relative to price move.
            # Positive means push UP. Negative means push DOWN.
            # Bid side (buys) -> pressure UP. Ask side (sells) -> pressure DOWN.
            # If we sum them: `total_pressure = pressure_grad_bid + pressure_grad_ask`.
            # If symmetric, they reinforce?
            # If Bid has `u_p=1`, pressure=+1.
            # If Ask has `u_p=1`, pressure=-1.
            # Net pressure = 0??
            # No. Bid is below spot, Ask is above spot.
            # At x=-10 (Bid side), we have Bid density. Ask density might be 0.
            # So we should sum the fields (filled with 0).
            # `pressure_total` = Sum(pressure_grad).
            
            # Pivot to accumulate sides
            # We want (rel_ticks) -> summed fields
            fields = group.groupby("rel_ticks")[["pressure_grad", "u_wave_energy", "rho", "nu"]].sum()
            # Ensure dense grid
            full_ticks = np.arange(-200, 201)
            fields = fields.reindex(full_ticks, fill_value=0.0)
            
            pg = fields["pressure_grad"].values
            we = fields["u_wave_energy"].values
            rho = fields["rho"].values
            nu = fields["nu"].values # Resistance
            
            # 1. Diagnostics (D_up, D_down, RunScore)
            # Omega? We need Omega for D_wall (Wall detection)
            # We didn't sum Omega. Let's get Omega separately.
            omega_g = group.groupby("rel_ticks")["Omega"].max() # Max of sides?
            omega_g = omega_g.reindex(full_ticks, fill_value=0.0)
            omega = omega_g.values
            
            # Wall: Omega > 3.0
            is_wall = omega > 3.0
            
            # D_up: nearest wall at x > 0
            # rel_ticks index: 0 corresponds to full_ticks search
            # center index (0 tick) is at index 200.
            center_idx = 200
            
            d_up = None
            walls_up = np.where(is_wall[center_idx+1:])[0] # relative to center+1
            if len(walls_up) > 0:
                d_up = int(walls_up[0] + 1)
            
            d_down = None
            walls_down = np.where(is_wall[:center_idx])[0]
            if len(walls_down) > 0:
                # last one (closest to center from below)
                d_down = int(center_idx - walls_down[-1])
            
            # RunScore: Avg pressure in free path
            # If D_up is None, take max (200).
            run_up_end = center_idx + (d_up if d_up else 200)
            run_down_start = center_idx - (d_down if d_down else 200)
            
            score_up = 0.0
            if run_up_end > center_idx:
                score_up = np.mean(pg[center_idx:run_up_end])
                
            score_down = 0.0
            if run_down_start < center_idx:
                score_down = np.mean(pg[run_down_start:center_idx]) # pg is signed. Down move implies negative pressure.
                # Average negative pressure?
                # "RunScore_down" usually implies "favorability for down move".
                # If pg is negative, that helps down move.
                # So we might want -1 * mean(pg).
                # Spec doesn't clarify sign of score. Assuming raw avg signed pressure.
            
            # 2. Particle Sim
            # v_eff = g_dir / (nu * rho)
            # g_dir = pg + sign(pg) * we
            g_dir = pg + np.sign(pg) * we
            
            # Avoid divide by zero
            # If rho is small (gap), resistance is 0?
            # Spec: "if rho < threshold, v = g_dir (ballistic)".
            mask_gap = rho < 0.1
            denom = nu * rho
            denom[mask_gap] = 1.0 # arbitrary, effectively replaced below
            
            v_eff = g_dir / (denom + 1e-6)
            v_eff[mask_gap] = g_dir[mask_gap] # ballistic in gaps
            
            # CRITICAL: Clamp v_eff to reasonable range
            # Pressure gradients are typically small (-1 to +1 range)
            # Without clamping, numerical issues can cause huge velocities
            v_eff = np.clip(v_eff, -5.0, 5.0)  # Max 5 ticks/sec drift
            
            # Simulation
            # Particles start at x=0 (relative to spot)
            # We simulate in "tick space".
            # v_eff represents directional drift in ticks/sec
            
            p_x = np.zeros(self.n_particles)
            sigma_brownian = 0.5 # Reduced noise for cleaner predictions
            
            for t in range(self.horizon_s):
                # Interpolate v_eff at p_x (particle positions)
                vals = np.interp(p_x, full_ticks, v_eff)
                
                # Update position: drift + noise
                noise = np.random.normal(0, sigma_brownian, self.n_particles)
                p_x += vals * self.dt + noise
                
                # Keep particles in reasonable bounds
                p_x = np.clip(p_x, -150, 150)
            
            mean_delta = np.mean(p_x)
            # Ensure delta is reasonable
            mean_delta = np.clip(mean_delta, -100, 100)
            
            results.append({
                "window_end_ts_ns": window_id,
                "horizon_s": self.horizon_s,
                "predicted_spot_tick": 0, # Relative? Contract says "predicted_spot_tick" (long). Likely absolute or relative. Doc default null.
                # Let's output relative "predicted_tick_delta".
                "predicted_tick_delta": int(mean_delta),
                "confidence": 0.5, # Placeholder
                "RunScore_up": float(score_up),
                "RunScore_down": float(score_down),
                "D_up": d_up,
                "D_down": d_down
            })
            
        return pd.DataFrame(results)


class VelocityStreamService:
    """Unified streaming service for futures + options visualization."""

    def __init__(self) -> None:
        self.cache: Dict[Tuple[str, str], UnifiedStreamCache] = {}
        self.repo_root = Path(__file__).resolve().parents[2]
        self.cfg = load_config(self.repo_root, self.repo_root / "src" / "data_eng" / "config" / "datasets.yaml")

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
        df_velocity = _load(velocity_key)
        # Ensure we have all columns, map missing if needed
        avail_cols = [c for c in VELOCITY_COLUMNS if c in df_velocity.columns]
        df_velocity = df_velocity.loc[:, avail_cols].copy()
        
        # Fill missing physics columns with 0 if gold version mismatched
        for c in VELOCITY_COLUMNS:
            if c not in df_velocity.columns:
                df_velocity[c] = 0.0

        df_velocity = df_velocity.loc[df_velocity["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)]

        # Load options velocity and aggregate across C/P and A/B
        df_options_raw = _load(options_key)
        df_options = _aggregate_options(df_options_raw)
        df_options = df_options.loc[df_options["rel_ticks"].between(-MAX_TICKS, MAX_TICKS)]

        # Validate options are on $5 grid (20 ticks)
        if not df_options.empty and (df_options["rel_ticks"] % 20 != 0).any():
            raise ValueError("Option velocity rel_ticks not aligned to $5 grid")
            
        # Run Forecast
        engine = ForecastEngine()
        df_forecast = engine.run_batch(df_velocity)

        # Get window IDs from snap (reference time series)
        window_ids = df_snap["window_end_ts_ns"].sort_values().unique().astype(int).tolist()

        # Group by window
        snap_by_window = _group_by_window(df_snap)
        velocity_by_window = _group_by_window(df_velocity)
        options_by_window = _group_by_window(df_options)
        forecast_by_window = _group_by_window(df_forecast)

        cache = UnifiedStreamCache(
            snap=df_snap,
            velocity=df_velocity,
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
            velocity_df = cache.velocity_by_window.get(window_id, pd.DataFrame(columns=VELOCITY_COLUMNS))
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
    
    # Aggregate across right (C/P) and side (A/B)
    df_agg = (
        df.groupby(
            ["window_end_ts_ns", "spot_ref_price_int", "rel_ticks"],
            as_index=False,
        )
        .agg(liquidity_velocity=("liquidity_velocity", "sum"))
    )
    return df_agg[OPTIONS_COLUMNS]


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
