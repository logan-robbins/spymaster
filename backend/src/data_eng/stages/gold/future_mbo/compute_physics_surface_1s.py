from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....filters.gold_strict_filters import apply_gold_strict_filters
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

logger = logging.getLogger(__name__)

EPS_QTY = 1.0


class GoldComputePhysicsSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_compute_physics_surface_1s",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.book_snapshot_1s",
                    "silver.future_mbo.depth_and_flow_1s",
                ],
                output="gold.future_mbo.physics_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        snap_key = "silver.future_mbo.book_snapshot_1s"
        flow_key = "silver.future_mbo.depth_and_flow_1s"

        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        flow_ref = partition_ref(cfg, flow_key, symbol, dt)

        for ref in (snap_ref, flow_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Input not ready: {ref.dataset_key} dt={dt}")

        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        flow_contract = load_avro_contract(repo_root / cfg.dataset(flow_key).contract)

        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_flow = enforce_contract(read_partition(flow_ref), flow_contract)

        # Apply gold strict filters for institutional-grade data
        original_snap_count = len(df_snap)
        original_flow_count = len(df_flow)
        
        df_snap, snap_stats = apply_gold_strict_filters(df_snap, product_type="futures", return_stats=True)
        df_flow, flow_stats = apply_gold_strict_filters(df_flow, product_type="futures", return_stats=True)
        
        if snap_stats.get("total_filtered", 0) > 0 or flow_stats.get("total_filtered", 0) > 0:
            logger.info(
                f"Gold filters for {dt}: snap={snap_stats.get('total_filtered', 0)}/{original_snap_count}, "
                f"flow={flow_stats.get('total_filtered', 0)}/{original_flow_count}"
            )

        df_out = self.transform(df_snap, df_flow)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": flow_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(flow_ref)},
        ]

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )

    def transform(
        self,
        df_snap: pd.DataFrame,
        df_flow: pd.DataFrame,
    ) -> pd.DataFrame:
        if df_snap.empty or df_flow.empty:
            return pd.DataFrame(
                columns=[
                    "window_end_ts_ns",
                    "event_ts_ns",
                    "spot_ref_price_int",
                    "rel_ticks",
                    "rel_ticks_side",
                    "side",
                    "add_intensity",
                    "fill_intensity",
                    "pull_intensity",
                    "liquidity_velocity",
                ]
            )

        # -----------------------------
        # Symmetrical Mechanics
        # -----------------------------
        df = df_flow.copy()

        # Inputs
        depth_start = df["depth_qty_start"].astype(float).to_numpy()
        depth_end = df["depth_qty_end"].astype(float).to_numpy()
        depth_rest = df["depth_qty_rest"].astype(float).to_numpy()
        
        # Quantities
        add_qty = df["add_qty"].astype(float).to_numpy()
        fill_qty = df["fill_qty"].astype(float).to_numpy()
        pull_qty = df["pull_qty"].astype(float).to_numpy()
        
        # Intensities = Qty / (Depth + Epsilon)
        denom = depth_start + EPS_QTY
        
        df["add_intensity"] = add_qty / denom
        df["fill_intensity"] = fill_qty / denom
        df["pull_intensity"] = pull_qty / denom

        # Net Velocity (add - pull - fill)
        df["liquidity_velocity"] = df["add_intensity"] - df["pull_intensity"] - df["fill_intensity"]

        # -------------------------------------------------------------
        # 4. Obstacles basic fields (rho, phi_rest)
        # -------------------------------------------------------------
        import numpy as np
        df["rho"] = np.log(1.0 + depth_end)
        df["phi_rest"] = depth_rest / (depth_end + 1.0)

        # -------------------------------------------------------------
        # 2. Temporal Awareness (EMAs) per cell (rel_ticks, side)
        # -------------------------------------------------------------
        # Sort for EWM
        df = df.sort_values(["rel_ticks", "side", "window_end_ts_ns"])
        
        g = df.groupby(["rel_ticks", "side"])["liquidity_velocity"]
        
        # Alphas
        alpha_2 = 1.0 - np.exp(-1.0 / 2.0)
        alpha_8 = 1.0 - np.exp(-1.0 / 8.0)
        alpha_32 = 1.0 - np.exp(-1.0 / 32.0)
        alpha_128 = 1.0 - np.exp(-1.0 / 128.0)
        
        # Compute EMAs
        # adjust=False corresponds to the recursive definition in spec
        df["u_ema_2"] = g.ewm(alpha=alpha_2, adjust=False).mean().reset_index(level=[0, 1], drop=True)
        df["u_ema_8"] = g.ewm(alpha=alpha_8, adjust=False).mean().reset_index(level=[0, 1], drop=True)
        df["u_ema_32"] = g.ewm(alpha=alpha_32, adjust=False).mean().reset_index(level=[0, 1], drop=True)
        df["u_ema_128"] = g.ewm(alpha=alpha_128, adjust=False).mean().reset_index(level=[0, 1], drop=True)
        
        # Bands
        df["u_band_fast"] = df["u_ema_2"] - df["u_ema_8"]
        df["u_band_mid"] = df["u_ema_8"] - df["u_ema_32"]
        df["u_band_slow"] = df["u_ema_32"] - df["u_ema_128"]
        
        # Energy
        df["u_wave_energy"] = np.sqrt(
            df["u_band_fast"]**2 + df["u_band_mid"]**2 + df["u_band_slow"]**2
        )
        
        # Temporal Derivatives (du_dt, d2u_dt2) based on u_ema_2
        # We need to take diff per group
        # Since df is sorted, we can use shift/diff but must respect groups (or use optimized diff if groups contiguous)
        # Using groupby shift is safer
        u2_curr = df["u_ema_2"]
        u2_prev = df.groupby(["rel_ticks", "side"])["u_ema_2"].shift(1)
        df["du_dt"] = (u2_curr - u2_prev).fillna(0.0) # Delta t = 1s
        
        du_curr = df["du_dt"]
        du_prev = df.groupby(["rel_ticks", "side"])["du_dt"].shift(1)
        df["d2u_dt2"] = (du_curr - du_prev).fillna(0.0)

        # Persistence-weighted velocity
        df["u_p"] = df["phi_rest"] * df["u_ema_8"]
        df["u_p_slow"] = df["phi_rest"] * df["u_ema_32"]
        
        # Obstacle Strength Omega
        # Omega = rho * (0.5 + 0.5*phi_rest) * (1 + max(0, u_p_slow))
        term_rest = 0.5 + 0.5 * df["phi_rest"]
        term_reinforce = 1.0 + np.maximum(0.0, df["u_p_slow"])
        df["Omega"] = df["rho"] * term_rest * term_reinforce

        # -------------------------------------------------------------
        # 3. Spatial Awareness (Smoothing) per frame (ts, side)
        # -------------------------------------------------------------
        # We need to smooth 'u_ema_8' -> 'u_near', 'u_ema_32' -> 'u_far'
        # Also 'Omega' -> 'Omega_near', 'Omega_far'
        
        # We define a helper to apply spatial smoothing
        def apply_spatial_smoothing(df_in, target_col, sigma, n_ticks, out_col_name):
            """Apply Gaussian spatial smoothing along rel_ticks dimension."""
            import warnings
            
            # Pivot to [ts, side] x [rel_ticks]
            # Use groupby().unstack() to enforce MultiIndex and handle duplicates via mean
            pivoted = df_in.groupby(["window_end_ts_ns", "side", "rel_ticks"])[target_col].mean().unstack("rel_ticks")
            
            # Reindex to dense grid -200 to +200
            full_ticks = np.arange(-200, 201)
            pivoted = pivoted.reindex(columns=full_ticks, fill_value=0.0)
            
            # Apply rolling gaussian along columns (rel_ticks dimension)
            win_size = int(2 * n_ticks + 1)
            
            # Use apply with rolling on each row to avoid deprecated axis param
            # Or use the deprecated axis=1 with suppressed warning for now
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                smoothed = pivoted.rolling(
                    window=win_size, 
                    win_type='gaussian', 
                    center=True, 
                    axis=1
                ).mean(std=sigma)
            
            # Stack back to long format - explicitly preserve MultiIndex
            # First reset index to flatten MultiIndex to columns
            smoothed = smoothed.reset_index()
            
            # Melt from wide to long format
            melted = smoothed.melt(
                id_vars=["window_end_ts_ns", "side"],
                var_name="rel_ticks",
                value_name=out_col_name
            )
            
            return melted

        # Apply smoothing
        # u_near: sigma=6, N=16, source=u_ema_8
        u_near_df = apply_spatial_smoothing(df, "u_ema_8", 6, 16, "u_near")
        # u_far: sigma=24, N=64, source=u_ema_32
        u_far_df = apply_spatial_smoothing(df, "u_ema_32", 24, 64, "u_far")
        
        # Omega_near: sigma=6, N=16
        omega_near_df = apply_spatial_smoothing(df, "Omega", 6, 16, "Omega_near")
        # Omega_far: sigma=24, N=64
        omega_far_df = apply_spatial_smoothing(df, "Omega", 24, 64, "Omega_far")
        
        # Merge back
        # We merge on keys. Note that Pivot filled missing with 0, so we might have new rows (0-filled).
        # We should perform Left Join if we only want original rows, OR Outer Join if we want the full field.
        # Physics field should arguably be dense.
        # Gold layer usually assumes we output what we got from Silver.
        # But if we want 0s in vacuous regions, we should keep them.
        # For efficiency, let's keep only rows that existed in silver (+ rows that were filled by smoothing near boundaries?)
        # Let's Left Merge to df. This implies we discard smoothed values in empty regions?
        # If the grid is sparse, derivatives might be wrong if we don't have neighbors.
        # But `apply_spatial_smoothing` *created* a dense grid intermediate.
        # If we discard rows, we lose the density.
        # However, writing Dense 401 ticks * 2 sides * 3600 seconds = 2.8M rows. That's fine for Gold Parquet.
        # It ensures downstream (stream server) doesn't have holes.
        # Let's use the dense grid from smoothing as the base!
        
        # Base frame from u_near (dense)
        df_dense = u_near_df.merge(u_far_df, on=["window_end_ts_ns", "side", "rel_ticks"], how="outer")
        df_dense = df_dense.merge(omega_near_df, on=["window_end_ts_ns", "side", "rel_ticks"], how="outer")
        df_dense = df_dense.merge(omega_far_df, on=["window_end_ts_ns", "side", "rel_ticks"], how="outer")
        
        # Merge original data
        # "right" merge to keep dense, or "left" to keep sparse. 
        # Using "right" merge means we get the dense grid. original cols will be NaN where missing.
        df_merged = df.merge(
            df_dense, 
            on=["window_end_ts_ns", "side", "rel_ticks"], 
            how="right"
        )
        
        # Fill NaNs in original fields with 0 (since they were empty in silver)
        # Also fill spatial smoothing fields whose Gaussian boundary produces NaN
        fill_cols = [
            "add_intensity", "fill_intensity", "pull_intensity", "liquidity_velocity",
            "rho", "phi_rest", "u_ema_2", "u_ema_8", "u_ema_32", "u_ema_128",
            "u_band_fast", "u_band_mid", "u_band_slow", "u_wave_energy",
            "du_dt", "d2u_dt2", "u_p", "u_p_slow", "Omega",
            "u_near", "u_far", "Omega_near", "Omega_far",
        ]
        for c in fill_cols:
            if c in df_merged.columns:
                df_merged[c] = df_merged[c].fillna(0.0)

        # Cast rel_ticks to int (melt produces object dtype from column names)
        df_merged["rel_ticks"] = df_merged["rel_ticks"].astype(int)
                
        # Fill spot_ref_price_int?
        # It's constant per window_end_ts_ns.
        # We need to backfill it from existing info.
        # Map timestamp -> spot_ref (window_end_ts_ns is our event reference)
        spot_map = df[["window_end_ts_ns", "spot_ref_price_int"]].drop_duplicates()
        # If duplicated timestamps have diff spot (unlikely given grain), take first.
        spot_map = spot_map.drop_duplicates("window_end_ts_ns")
        
        # Determine rel_ticks_side for new rows?
        # rel_ticks_side depends on bid/ask anchor.
        # We can't easily reconstruct it without book state.
        # Just fill with 0 or rel_ticks?
        # It's an informational column.
        # Let's just drop spot_ref_price_int from merge and re-join it.
        if "spot_ref_price_int" in df_merged.columns:
             df_merged = df_merged.drop(columns=["spot_ref_price_int"])
             
        df_merged = df_merged.merge(spot_map, on="window_end_ts_ns", how="left")
        
        # -------------------------------------------------------------
        # Derived Spatial Fields
        # -------------------------------------------------------------
        # Prominence
        df_merged["u_prom"] = df_merged["u_near"] - df_merged["u_far"]
        df_merged["Omega_prom"] = df_merged["Omega_near"] - df_merged["Omega_far"]
        
        # Spatial Derivatives on u_near
        # Sort by rel_ticks
        df_merged = df_merged.sort_values(["window_end_ts_ns", "side", "rel_ticks"])
        
        # Group by frame to differentiate ticks
        g_space = df_merged.groupby(["window_end_ts_ns", "side"])
        
        # Central difference: (x+1 - x-1)/2
        # Use shift(-1) - shift(1) / 2
        u_near = df_merged["u_near"]
        df_merged["du_dx"] = (g_space["u_near"].shift(-1) - g_space["u_near"].shift(1)) * 0.5
        df_merged["du_dx"] = df_merged["du_dx"].fillna(0.0) # Boundaries
        
        # d2u_dx2 = x+1 - 2x + x-1
        df_merged["d2u_dx2"] = g_space["u_near"].shift(-1) - 2.0 * u_near + g_space["u_near"].shift(1)
        df_merged["d2u_dx2"] = df_merged["d2u_dx2"].fillna(0.0)
        
        # -------------------------------------------------------------
        # 4.6 Effective Viscosity, Resistance, Permeability
        # -------------------------------------------------------------
        # R = 1 + Omega_near + 2*max(0, Omega_prom)
        # nu = R
        # kappa = 1 / nu
        Omega_near = df_merged["Omega_near"]
        Omega_prom = df_merged["Omega_prom"]
        
        R = 1.0 + Omega_near + 2.0 * np.maximum(0.0, Omega_prom)
        df_merged["nu"] = R
        df_merged["kappa"] = 1.0 / R
        
        # -------------------------------------------------------------
        # 5. Pressure Gradient Force
        # -------------------------------------------------------------
        # Bid side: +u_p
        # Ask side: -u_p
        # u_p is persistence weighted u_ema_8.
        # We need u_p (which was filled with 0 for new rows).
        # Wait, if we use dense grid, we should probably recompute u_p from smoothed u??
        # Spec says: "Using the persistence-weighted mid-scale velocity u_p: ... = +/- u_p"
        # u_p = phi_rest * u_ema_8.
        # Is u_p smoothed? Spec doesn't say "u_p_near".
        # It implies local cell value.
        # But we filled u_p with 0 in empty regions.
        # This is correct: no persistence -> no pressure source.
        
        u_p = df_merged["u_p"]
        side_cond = (df_merged["side"] == "B")
        df_merged["pressure_grad"] = np.where(side_cond, u_p, -u_p)
        
        # Fill missing informational columns
        if "rel_ticks_side" in df_merged.columns:
            df_merged["rel_ticks_side"] = df_merged["rel_ticks_side"].fillna(0).astype(int)
        else:
            df_merged["rel_ticks_side"] = 0

        df_merged["event_ts_ns"] = df_merged["window_end_ts_ns"]

        return df_merged[[
            "window_end_ts_ns", 
            "event_ts_ns", 
            "spot_ref_price_int", 
            "rel_ticks", 
            "rel_ticks_side",
            "side", 
            "add_intensity",
            "fill_intensity",
            "pull_intensity",
            "liquidity_velocity",
            # New Fields
            "u_ema_2", "u_ema_8", "u_ema_32", "u_ema_128", 
            "u_band_fast", "u_band_mid", "u_band_slow", "u_wave_energy",
            "du_dt", "d2u_dt2",
            "u_near", "u_far", "u_prom",
            "du_dx", "d2u_dx2",
            "rho", "phi_rest", "u_p", "u_p_slow",
            "Omega", "Omega_near", "Omega_far", "Omega_prom",
            "nu", "kappa",
            "pressure_grad"
        ]]


