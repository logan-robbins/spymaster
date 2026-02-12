from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref

def check_integrity_v2(symbol: str, dt: str) -> None:
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
    print(f"Checking integrity (V2) for {symbol} on {dt}")

    # =========================================================================
    # OPTION DEPTH/FLOW CHECKS
    # =========================================================================
    print("\n[checking Option Depth+Flow Surface...]")
    opt_ref = partition_ref(cfg, "silver.future_option_mbo.depth_and_flow_1s", symbol, dt)
    if not opt_ref.dir.exists():
        print(f"SKIPPING: Option depth/flow partition missing at {opt_ref.dir}")
    else:
        df_opt = pd.read_parquet(opt_ref.dir)
        print(f"Loaded {len(df_opt)} rows.")

        if df_opt.empty:
            print("WARNING: Option depth/flow dataframe is empty.")
        else:
            # Expect 21 strikes * 2 rights * 2 sides = 84 rows per window
            counts = df_opt.groupby("window_end_ts_ns").size()
            invalid_counts = counts[counts != 84]
            if not invalid_counts.empty:
                print(f"FAIL: Found {len(invalid_counts)} windows without exactly 84 rows.")
                print(invalid_counts.head())
            else:
                print("PASS: All windows have 84 strike/right/side rows.")

            required = {"rel_ticks", "strike_price_int", "spot_ref_price_int"}
            if required.issubset(df_opt.columns):
                tick_int = 250_000_000
                strike_step_int = 20 * tick_int

                strike_mod = df_opt["strike_price_int"].astype("int64") % strike_step_int != 0
                if strike_mod.any():
                    print(f"FAIL: Found {int(strike_mod.sum())} rows where strike_price_int is not on $5 grid.")
                else:
                    print("PASS: strike_price_int aligned to $5 grid.")

                delta = df_opt["strike_price_int"].astype("int64") - df_opt["spot_ref_price_int"].astype("int64")
                tick_misaligned = (delta % tick_int != 0)
                if tick_misaligned.any():
                    print(f"FAIL: Found {int(tick_misaligned.sum())} rows where strike delta is not on $0.25 ticks.")
                else:
                    expected_rel = (delta // tick_int).astype(int)
                    rel_mismatch = expected_rel != df_opt["rel_ticks"].astype(int)
                    if rel_mismatch.any():
                        print(f"FAIL: Found {int(rel_mismatch.sum())} rows where rel_ticks != (strike - spot) in ticks.")
                        print(df_opt.loc[rel_mismatch, ["window_end_ts_ns", "rel_ticks"]].head())
                    else:
                        print("PASS: rel_ticks matches strike - spot in ticks.")

                range_fails = df_opt[df_opt["rel_ticks"].abs() > 210]
                if not range_fails.empty:
                    print(f"FAIL: Found {len(range_fails)} rows where rel_ticks > 210.")
                else:
                    print("PASS: All rel_ticks within ±210 range.")
            else:
                print("WARNING: Missing rel_ticks/strike_price_int/spot_ref_price_int in option depth/flow data.")

    # =========================================================================
    # OPTION PHYSICS CHECKS
    # =========================================================================
    print("\n[checking Option Physics Surface...]")
    opt_phys_ref = partition_ref(cfg, "gold.future_option_mbo.physics_surface_1s", symbol, dt)
    if not opt_phys_ref.dir.exists():
        print(f"SKIPPING: Option physics partition missing at {opt_phys_ref.dir}")
    else:
        df_opt_phys = pd.read_parquet(opt_phys_ref.dir)
        print(f"Loaded {len(df_opt_phys)} rows.")

        if df_opt_phys.empty:
            print("WARNING: Option physics dataframe is empty.")
        else:
            required_cols = {"add_intensity", "pull_intensity", "fill_intensity", "liquidity_velocity"}
            missing = required_cols.difference(df_opt_phys.columns)
            if missing:
                print(f"FAIL: Missing option physics columns: {sorted(missing)}")
            else:
                print("PASS: Option physics columns present.")

    # =========================================================================
    # PHYSICS SURFACE CHECKS
    # =========================================================================
    print("\n[checking Physics Surface...]")
    phys_ref = partition_ref(cfg, "gold.future_mbo.physics_surface_1s", symbol, dt)
    if not phys_ref.dir.exists():
        print(f"SKIPPING: Physics partition missing at {phys_ref.dir}")
    else:
        df_phys = pd.read_parquet(phys_ref.dir)
        print(f"Loaded {len(df_phys)} rows.")
        
        if df_phys.empty:
            print("WARNING: Physics dataframe is empty.")
        else:
            # 1. Check rel_ticks exist
            if "rel_ticks" not in df_phys.columns:
                print("FAIL: rel_ticks missing in physics.")
            else:
                # 2. Check Score Bounds
                if "physics_score" in df_phys.columns:
                    bad_score = df_phys[~df_phys["physics_score"].between(0, 1)]
                    if not bad_score.empty:
                        print(f"FAIL: Found {len(bad_score)} rows with physics_score outside [0, 1].")
                    else:
                        print("PASS: physics_score in [0, 1].")
                
                if "physics_score_signed" in df_phys.columns:
                    bad_signed = df_phys[~df_phys["physics_score_signed"].between(-1, 1)]
                    if not bad_signed.empty:
                        print(f"FAIL: Found {len(bad_signed)} rows with physics_score_signed outside [-1, 1].")
                    else:
                        print("PASS: physics_score_signed in [-1, 1].")

            # 3. Check Timestamp Safety (Task 13 related)
            # Ensure no NaNs in window_end_ts_ns
            if df_phys["window_end_ts_ns"].isnull().any():
                print("FAIL: NaNs in window_end_ts_ns")
            else:
                print("PASS: Timestamps valid.")

    print("\nIntegrity V2 Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="Resolved contract, e.g. ESH6 or MNQH6")
    parser.add_argument("--dt", required=True, help="Session date YYYY-MM-DD")
    args = parser.parse_args()
    check_integrity_v2(args.symbol, args.dt)
