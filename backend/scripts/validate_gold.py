
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Expected Schema configuration
EXPECTED_DIM = 149  # per EPISODE_VECTOR_SCHEMA.md
GOLD_BASE = Path("data/gold/episodes/es_level_episodes/version=4.5.0")

def validate_date(date: str) -> Dict:
    """Validate Gold vectors for a single date."""
    vector_file = GOLD_BASE / "vectors" / f"date={date}" / "episodes.npy"
    meta_file = GOLD_BASE / "metadata" / f"date={date}" / "metadata.parquet"
    
    result = {
        "date": date,
        "status": "PASS",
        "count": 0,
        "issues": []
    }
    
    if not vector_file.parent.exists():
        result["status"] = "MISSING"
        return result
        
    if not vector_file.exists():
        result["status"] = "FAIL"
        result["issues"].append("Missing episodes.npy")
        return result
        
    if not meta_file.exists():
        result["status"] = "FAIL"
        result["issues"].append("Missing metadata.parquet")
        return result
        
    try:
        # Load Data
        vectors = np.load(vector_file)
        meta = pd.read_parquet(meta_file)
        
        count = vectors.shape[0]
        result["count"] = count
        
        # 1. Structural Checks
        if vectors.ndim != 2:
            result["status"] = "FAIL"
            result["issues"].append(f"Invalid dimensions: {vectors.ndim} (expected 2)")
            
        if vectors.shape[1] != EXPECTED_DIM:
            result["status"] = "FAIL"
            result["issues"].append(f"Schema Mismatch: {vectors.shape[1]}D (expected {EXPECTED_DIM}D)")
            
        if len(meta) != count:
            result["status"] = "FAIL"
            result["issues"].append(f"Mismatched counts: Vectors={count}, Meta={len(meta)}")
            
        # 2. Numerical Checks (NaNs/Infs)
        n_nans = np.isnan(vectors).sum()
        n_infs = np.isinf(vectors).sum()
        
        if n_nans > 0:
            result["status"] = "FAIL"
            result["issues"].append(f"Found {n_nans} NaNs in vectors")
            
        if n_infs > 0:
            result["status"] = "FAIL"
            result["issues"].append(f"Found {n_infs} Infs in vectors")
            
        # 3. Content Checks (Skeptical Validation)
        # Check for "Dead" vectors (all zeros)
        # Some zero vectors might be valid if everything is 0, but unlikely in 149D space
        zero_rows = (np.abs(vectors).sum(axis=1) == 0).sum()
        
        if zero_rows > 0:
             result["issues"].append(f"Found {zero_rows} ALL-ZERO vectors (suspicious)")
             if result["status"] == "PASS": result["status"] = "WARN"

        # Check DCT Coefficients (Section F: 117-148)
        # Indices 117-124 are DCT Distance. Should decay.
        if count > 0:
            dct_dist = vectors[:, 117:125]
            avg_energy = np.mean(np.abs(dct_dist), axis=0)
            
            # Check if high-freq coeffs are exploding (bad normalization)
            if avg_energy[-1] > avg_energy[0] * 2: # heuristic: high freq shouldn't be 2x DC
                 result["issues"].append("DCT coefficients rising (instability?)")
                 if result["status"] == "PASS": result["status"] = "WARN"
                 
            # Check OFI features (Section B OFI: 49-52) not getting zeroed out
            ofi_cols = vectors[:, 49:53]
            if np.all(ofi_cols == 0):
                result["status"] = "FAIL"
                result["issues"].append("OFI features (49-52) are ALL ZEROS")
                
            # Check Microstructure (Index 41: Vacuum)
            # Should have SOME non-zero values if backfill worked
            vacuum_col = vectors[:, 41]
            if np.all(vacuum_col == 0) and count > 0:
                 result["issues"].append("Vacuum feature (41) is ALL ZEROS")
                 if result["status"] == "PASS": result["status"] = "WARN"
                
    except Exception as e:
        result["status"] = "FAIL"
        result["issues"].append(f"Crash: {str(e)}")
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Validate Gold Pipelines")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    
    dates = pd.date_range(start=args.start, end=args.end, freq='B') # Business days
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    
    print(f"\nVALIDATING GOLD PIPELINE (149D) ({args.start} to {args.end})")
    print(f"Schema Expectation: {EXPECTED_DIM} dimensions")
    print("-" * 80)
    print(f"{'DATE':<12} | {'STATUS':<8} | {'VECTORS':<6} | {'ISSUES'}")
    print("-" * 80)
    
    passed = 0
    failed = 0
    
    for date in date_strs:
        res = validate_date(date)
        
        status_color = GREEN if res["status"] == "PASS" else (YELLOW if res["status"] == "WARN" else RED)
        status_str = f"{status_color}{res['status']}{RESET}"
        
        issues = "; ".join(res["issues"]) if res["issues"] else ""
        if len(issues) > 50: issues = issues[:47] + "..."
        
        print(f"{date:<12} | {status_str:<17} | {res['count']:<7} | {issues}")
        
        if res["status"] == "PASS" or res["status"] == "WARN":
            if res["status"] != "MISSING":
                passed += 1
        elif res["status"] == "FAIL":
            failed += 1
            
    print("-" * 80)
    print(f"SUMMARY: Passed: {passed} | Failed: {failed}")
    
    if failed > 0:
        exit(1)
        
if __name__ == "__main__":
    main()
