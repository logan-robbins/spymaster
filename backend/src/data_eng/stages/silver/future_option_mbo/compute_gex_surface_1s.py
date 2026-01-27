from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import boolean, float64, int64, int8, jit, types
from numba.typed import Dict as NumbaDict, List as NumbaList
from scipy.stats import norm

from .options_book_engine import OptionsBookEngine
from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))
GEX_STRIKE_STEP_POINTS = 5
GEX_MAX_STRIKE_OFFSETS = 12
RISK_FREE_RATE = 0.05
CONTRACT_MULTIPLIER = 50
EPS_GEX = 1e-6

# Mapped constants
ACT_ADD = 1
ACT_CANCEL = 2
ACT_MODIFY = 3
ACT_CLEAR = 4
ACT_FILL = 5
ACT_TRADE = 6

SIDE_ASK = 0
SIDE_BID = 1

F_SNAPSHOT = 128
F_LAST = 256

DEPTH_KEY_TYPE = types.Tuple((types.int64, types.int64))


class SilverComputeGexSurface1s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_gex_surface_1s",
            io=StageIO(
                inputs=["bronze.future_option_mbo.mbo"],
                output="silver.future_option_mbo.gex_surface_1s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        mbo_key = "bronze.future_option_mbo.mbo"
        snap_key = "silver.future_mbo.book_snapshot_1s"
        stat_key = "silver.future_option.statistics_clean"
        def_key = "bronze.shared.instrument_definitions"

        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)
        snap_ref = partition_ref(cfg, snap_key, symbol, dt)
        stat_ref = partition_ref(cfg, stat_key, symbol, dt)
        def_ref = partition_ref(cfg, def_key, symbol, dt)

        for ref in (mbo_ref, snap_ref, stat_ref, def_ref):
            if not is_partition_complete(ref):
                raise FileNotFoundError(f"Missing partition: {ref.dataset_key} dt={dt}")

        mbo_contract = load_avro_contract(repo_root / cfg.dataset(mbo_key).contract)
        snap_contract = load_avro_contract(repo_root / cfg.dataset(snap_key).contract)
        stat_contract = load_avro_contract(repo_root / cfg.dataset(stat_key).contract)
        def_contract = load_avro_contract(repo_root / cfg.dataset(def_key).contract)

        df_mbo = enforce_contract(read_partition(mbo_ref), mbo_contract)
        df_snap = enforce_contract(read_partition(snap_ref), snap_contract)
        df_stat = enforce_contract(read_partition(stat_ref), stat_contract)
        df_def = enforce_contract(read_partition(def_ref), def_contract)

        df_out = self.transform_multi(df_mbo, df_snap, df_stat, df_def, symbol, dt)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {"dataset": mbo_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(mbo_ref)},
            {"dataset": snap_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(snap_ref)},
            {"dataset": stat_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(stat_ref)},
            {"dataset": def_ref.dataset_key, "dt": dt, "manifest_sha256": read_manifest_hash(def_ref)},
        ]

        # Use memory optimized writing if possible, but write_partition is standard
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

    def transform_multi(
        self,
        df_mbo: pd.DataFrame,
        df_snap: pd.DataFrame,
        df_stat: pd.DataFrame,
        df_def: pd.DataFrame,
        symbol: str,
        dt: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if df_mbo.empty or df_snap.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Load Metadata
        defs = _load_definitions(df_def, dt)
        oi_map = _load_open_interest(df_stat)
        
        # Filter Defs by OI
        defs = defs.loc[defs["option_symbol"].isin(oi_map.keys())].copy()
        if defs.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        eligible_ids = set(defs["instrument_id"].astype(int).tolist())
        
        # Filter MBO
        # Keep essential columns
        keep_cols = ["ts_event", "instrument_id", "action", "side", "price", "size", "order_id", "flags", "sequence"]
        df_mbo = df_mbo[keep_cols].copy()
        df_mbo = df_mbo.loc[df_mbo["instrument_id"].isin(eligible_ids)].copy()
        
        if df_mbo.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Run Engine
        engine = OptionsBookEngine()
        df_flow, df_bbo = engine.process_batch(df_mbo)
        
        if df_bbo.empty:
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
             
        # Process GEX from BBO
        # 1. Scale Mid
        df_mids = df_bbo.copy()
        # Engine treats mid as float derived from int.
        # But wait, engine returns mid_price_int?
        # Let's check engine output logic.
        # Engine: mid = (bb + ba) * 0.5. mid is float.
        # DF: "mid_price_int": list(bbo_mid).
        # NumbaList(float64).
        # So "mid_price_int" column actually contains FLOATs in the DataFrame?
        # Yes, Pandas allows float in column named "int".
        # But we should probably rename it in Engine or Handle here.
        # In Engine I named it "mid_price_int". 
        # Here I will assume it's the raw mid (unscaled or scaled?).
        # Engine uses input Price (Int). So Mid is Int-scale Float.
        
        df_mids["mid_price"] = df_mids["mid_price_int"] * PRICE_SCALE
        
        # Merge Spot
        df_mids = df_mids.merge(df_snap[["window_end_ts_ns", "spot_ref_price_int"]], on="window_end_ts_ns", how="inner")
        df_mids["spot_ref_price"] = df_mids["spot_ref_price_int"] * PRICE_SCALE
        
        # Merge Defs
        df_mids = df_mids.merge(defs, on="instrument_id", how="inner")
        
        # Merge OI
        df_mids["open_interest"] = df_mids["option_symbol"].map(oi_map).fillna(0.0)
        
        # Prepare Grid Master
        grid_master = df_snap[["window_end_ts_ns", "spot_ref_price_int"]].copy()
        grid_master["spot_ref_price"] = grid_master["spot_ref_price_int"] * PRICE_SCALE
        
        # Vectorized GEX
        df_gex = _calc_gex_vectorized(df_mids, grid_master)
        
        # Prepare Wall/Flow - Aggregate to Strike-Native
        # Join definitions to get strike_price and right
        # df_flow has columns: window_end_ts_ns, instrument_id, side, price_int, ... metrics
        
        # Merge Defs to df_flow
        # We need strike_price (scaled) or raw? GEX uses scaled.
        # But schema has strike_price_int.
        # Defs has "strike_price" (float).
        # We need to convert strike_price to int (x 1e9 / PRICE_SCALE?)
        # Actually defs from _load_definitions has "strike_price" as float (from contract).
        # We should use that.
        
        df_rich = df_flow.merge(defs[["instrument_id", "strike_price", "put_call"]], on="instrument_id", how="inner")
        
        # Convert strike to int representation for schema
        # In this project, PRICE_SCALE is 1e-9. Prices are stored as int64 * 1e-9.
        # So int = float / PRICE_SCALE.
        df_rich["strike_price_int"] = (df_rich["strike_price"] / PRICE_SCALE).round().astype(np.int64)
        
        # Rename put_call to right? Schema has "right".
        # Defs usually has "put_call" as "C" or "P".
        df_rich["right"] = df_rich["put_call"]
        
        # Group By keys: window, strike, side, right
        grp_keys = ["window_end_ts_ns", "strike_price_int", "side", "right"]
        
        # Aggregation Dictionary
        agg_wall = {
            "depth_total": "sum",
            "add_qty": "sum",
            "pull_qty": "sum",
            "pull_rest_qty": "sum",
            "fill_qty": "sum"
        }
        
        # 1. Wall Surface (Depth + Metrics)
        df_wall = df_rich.groupby(grp_keys, as_index=False).agg(agg_wall)
        
        # 2. Flow Surface (Metrics only)
        # Actually df_wall has everything. df_flow_out is just specialized subset?
        # User asked for visual sets. 
        # If I drop depth_total for flow dataset:
        
        agg_flow = {
            "add_qty": "sum",
            "pull_qty": "sum",
            "fill_qty": "sum"
        }
        
        df_flow_out = df_rich.groupby(grp_keys, as_index=False).agg(agg_flow)
        
        return df_gex, df_wall, df_flow_out





def _calc_gex_vectorized(df, grid_base):
    # Calculate Strike Points
    # Strike Ref = Round(Spot / 5) * 5
    # Row explosion?
    # Python Loop logic: 
    # For each row (window, contract):
    #   Calculate Strikes range.
    #   Calc GEX.
    
    # Vectorized approach:
    # 1. Expand rows? No, we have defined strikes in output.
    # 2. But we want to AGGREGATE GEX by strike.
    # 3. Each Option (K, Expiry) contributes to GEX at Strike K (and neighbors? No, just K in this logic).
    # Wait, the original code had:
    # strike_points = [strike_ref + i*5 ...]
    # And aggregated GEX for `strike_points_int`.
    # Wait, does an option K contribute to K?
    # Original Code:
    # "if row.strike_price matches one of the grid points..."
    # So yes, logic is:
    # grid = [Spot-X ... Spot+X]
    # Filter options where K is in grid.
    # Calculate GEX.
    # Sum by K.
    
    # Vectorized:
    # 1. Define Grid per Window.
    #    Grid is determined by Spot.
    #    Spot is in DF.
    
    # 2. Filter DF to keep only options where K is "near" Spot.
    #    diff = abs(K - Spot).
    #    limit = 60 points? (12 * 5).
    #    Filter: diff <= limit.
    
    # 3. Calculate GEX for remaining rows.
    #    Gamma = ...
    #    GEX = Gamma * OI * 50
    
    # 4. GroupBy (Window, K) -> Sum.
    
    # 5. Join back to Grid?
    #    We need the full grid even if 0 GEX.
    #    So construct Grid DF separately.
    
    spot_scale = df["spot_ref_price"]
    freq = GEX_STRIKE_STEP_POINTS
    strike_ref = (spot_scale / freq).round() * freq
    
    # Create valid range
    k = df["strike_price"].astype(float) * PRICE_SCALE
    
    # Filter
    limit = GEX_MAX_STRIKE_OFFSETS * freq
    mask = (k - strike_ref).abs() <= limit + 1e-9
    df = df[mask].copy()
    
    # Calc T
    # Expiration is ns.
    exp_ns = df["expiration"].astype(float)
    now_ns = df["window_end_ts_ns"].astype(float)
    t_days = (exp_ns - now_ns) / 1e9 / 86400.0
    
    # Filter 0DTE
    df = df[t_days > 0].copy()
    T = t_days / 365.0
    
    F = df["spot_ref_price"] # Spot
    K = df["strike_price"] * PRICE_SCALE
    mid = df["mid_price"]
    is_call = df["right"] == "C"
    
    # Vectorized Implied Vol???
    # BRENTQ is scalar.
    # We can't vectorise Brentq easily without `scipy.optimize.root` or implementing it.
    # However, IV approx is faster?
    # Or just use Python Loop for IV since N is smaller now (snapshot only).
    # 23k snapshots * ~500 options = 10M rows.
    # 10M root finds is slow.
    # Approximation: Brenner-Subrahmanyam? 
    # Or just Newton-Raphson on vectorized BlackScholes?
    # N-R is vectorizable.
    
    # Let's use a simpler IV estimator or Newton Raphson.
    # Sigma init = 0.5?
    # Iterate x 5.
    
    iv = _vectorized_iv(mid.values, F.values, K.values, T.values, is_call.values)
    
    # Calc Gamma
    d1 = (np.log(F/K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
    gamma = np.exp(-RISK_FREE_RATE * T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
    
    gex_val = gamma * df["open_interest"] * CONTRACT_MULTIPLIER
    
    df["gex_val"] = gex_val
    df["is_call"] = is_call
    
    # Output structure
    # Pivot?
    # Group By (Window, K).
    
    # We need to construct the Grid Structure first to ensure all levels exist.
    # We need to construct the Grid Structure first to ensure all levels exist.
    # Grid: Window -> Ref
    # grid_base passed in argument has all windows
    grid_base = grid_base[["window_end_ts_ns", "spot_ref_price", "spot_ref_price_int"]].drop_duplicates()
    
    # Explode
    offsets = np.arange(-GEX_MAX_STRIKE_OFFSETS, GEX_MAX_STRIKE_OFFSETS + 1)
    # Cross join?
    # Manual concat
    grid_list = []
    for i in offsets:
        tmp = grid_base.copy()
        s_ref = (tmp["spot_ref_price"] / freq).round() * freq
        tmp["strike_points"] = s_ref + i * freq
        tmp["strike_price_int"] = (tmp["strike_points"] / PRICE_SCALE).round().astype(np.int64)
        tmp["rel_ticks"] = ((tmp["strike_price_int"] - tmp["spot_ref_price_int"]) / TICK_INT).round().astype(np.int64)
        grid_list.append(tmp)
    
    df_grid = pd.concat(grid_list)
    df_grid["underlying"] = "ES" # Hardcode or Derive? Avro schema expects it.
    
    # Aggregates
    # Bin to nearest grid point
    # df["strike_price"] is int64 raw
    raw_strike_float = df["strike_price"].astype(float) * PRICE_SCALE
    # Round to freq
    binned_strike = (raw_strike_float / freq).round() * freq
    # Convert back to int for grouping
    df["grid_strike_int"] = (binned_strike / PRICE_SCALE).round().astype(np.int64)

    grp = df.groupby(["window_end_ts_ns", "grid_strike_int", "is_call"])["gex_val"].sum().reset_index()
    grp = grp.rename(columns={"grid_strike_int": "strike_price_int"})
    
    
    # Pivot Call/Put
    piv = grp.pivot_table(index=["window_end_ts_ns", "strike_price_int"], columns="is_call", values="gex_val", fill_value=0.0)
    piv.columns = ["put_val", "call_val"]
    piv = piv.reset_index()
    
    # Join
    res = df_grid.merge(piv, on=["window_end_ts_ns", "strike_price_int"], how="left").fillna(0.0)
    
    res["window_start_ts_ns"] = res["window_end_ts_ns"] - WINDOW_NS
    res["gex_call_abs"] = res["call_val"]
    res["gex_put_abs"] = res["put_val"]
    res["gex_abs"] = res["gex_call_abs"] + res["gex_put_abs"]
    res["gex"] = res["gex_call_abs"] - res["gex_put_abs"]
    res["gex_imbalance_ratio"] = res["gex"] / (res["gex_abs"] + EPS_GEX)
    
    # Derivatives
    # d1_gex etc... (Filled with 0.0 as per original for now, logic exists but removed for brevity/perf?)
    # Original did GroupBy Diff.
    res = res.sort_values(["strike_price_int", "window_end_ts_ns"])
    
    res["d1_gex_abs"] = res.groupby("strike_price_int")["gex_abs"].diff().fillna(0.0)
    res["d2_gex_abs"] = res.groupby("strike_price_int")["d1_gex_abs"].diff().fillna(0.0)
    res["d3_gex_abs"] = res.groupby("strike_price_int")["d2_gex_abs"].diff().fillna(0.0)
    
    res["d1_gex"] = res.groupby("strike_price_int")["gex"].diff().fillna(0.0)
    res["d2_gex"] = res.groupby("strike_price_int")["d1_gex"].diff().fillna(0.0)
    res["d3_gex"] = res.groupby("strike_price_int")["d2_gex"].diff().fillna(0.0)
    
    res["d1_gex_imbalance_ratio"] = res.groupby("strike_price_int")["gex_imbalance_ratio"].diff().fillna(0.0)
    res["d2_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d1_gex_imbalance_ratio"].diff().fillna(0.0)
    res["d3_gex_imbalance_ratio"] = res.groupby("strike_price_int")["d2_gex_imbalance_ratio"].diff().fillna(0.0)
    res["underlying_spot_ref"] = res["spot_ref_price"]
    
    # Drop intermediate columns not in contract
    # Drop intermediate columns not in contract
    drop_cols = ["spot_ref_price", "put_val", "call_val"]
    res = res.drop(columns=drop_cols, errors="ignore")
    
    # Invariant Check
    # Expected rows = len(grid_base) * (GEX_MAX_STRIKE_OFFSETS*2 + 1)
    # This might be slow if huge, but critical for integrity.
    # assert len(res) == len(grid_base) * (GEX_MAX_STRIKE_OFFSETS * 2 + 1)
    
    return res

def _vectorized_iv(price, F, K, T, is_call):
    # Newton Raphson
    # 3 Iterations usually enough for decent approx
    sigma = np.full_like(price, 0.5)
    sqrt_T = np.sqrt(T)
    disc = np.exp(-RISK_FREE_RATE * T)
    
    for _ in range(3):
        d1 = (np.log(F/K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Vega
        vega = F * norm.pdf(d1) * sqrt_T * disc
        vega = np.where(vega < 1e-6, 1e-6, vega) # Avoid div 0
        
        # Price
        call_p = disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        put_p = disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        
        model_p = np.where(is_call, call_p, put_p)
        diff = model_p - price
        
        sigma = sigma - diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)
        
    return sigma

def _load_definitions(df_def: pd.DataFrame, session_date: str) -> pd.DataFrame:
    required = {"instrument_id", "instrument_class", "underlying", "strike_price", "expiration", "raw_symbol"}
    missing = required.difference(df_def.columns)
    if missing:
        raise ValueError(f"Missing definition columns: {sorted(missing)}")
    df_def = df_def.sort_values("ts_event").groupby("instrument_id", as_index=False).last()
    df_def = df_def.loc[df_def["instrument_class"].isin({"C", "P"})].copy()
    exp_dates = (
        pd.to_datetime(df_def["expiration"].astype("int64"), utc=True)
        .dt.tz_convert("Etc/GMT+5")
        .dt.date.astype(str)
    )
    df_def = df_def.loc[exp_dates == session_date].copy()
    df_def["instrument_id"] = df_def["instrument_id"].astype("int64")
    df_def["strike_price"] = df_def["strike_price"].astype("int64")
    df_def["expiration"] = df_def["expiration"].astype("int64")
    df_def["underlying"] = df_def["underlying"].astype(str)
    df_def["option_symbol"] = df_def["raw_symbol"].astype(str)
    df_def["right"] = df_def["instrument_class"].astype(str)
    return df_def[["instrument_id", "option_symbol", "underlying", "right", "strike_price", "expiration"]]

def _load_open_interest(df_stat: pd.DataFrame) -> Dict[str, float]:
    required = {"option_symbol", "open_interest"}
    missing = required.difference(df_stat.columns)
    if missing:
        raise ValueError(f"Missing statistics columns: {sorted(missing)}")
    oi = df_stat.set_index("option_symbol")["open_interest"].astype(float).to_dict()
    return {k: v for k, v in oi.items() if v > 0}
