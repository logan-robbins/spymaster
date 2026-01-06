"""Sanity check approach features for data quality issues."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LAKE_ROOT = REPO_ROOT / "lake"

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]

FEATURE_PREFIXES = [
    "bar5s_approach_",
    "bar5s_cumul_",
    "bar5s_deriv_",
    "bar5s_lvl_",
    "bar5s_setup_",
    "rvol_",
]

EXPECTED_NULL_FEATURES = {
    "rvol_trade_vol_ratio",
    "rvol_trade_vol_zscore",
    "rvol_trade_cnt_ratio",
    "rvol_trade_cnt_zscore",
    "rvol_trade_aggbuy_ratio",
    "rvol_trade_aggsell_ratio",
    "rvol_trade_aggbuy_zscore",
    "rvol_trade_aggsell_zscore",
    "rvol_flow_add_bid_ratio",
    "rvol_flow_add_ask_ratio",
    "rvol_flow_add_bid_zscore",
    "rvol_flow_add_ask_zscore",
    "rvol_flow_net_bid_ratio",
    "rvol_flow_net_ask_ratio",
    "rvol_flow_net_bid_zscore",
    "rvol_flow_net_ask_zscore",
    "rvol_flow_add_total_ratio",
    "rvol_flow_add_total_zscore",
    "rvol_cumul_trade_vol_dev",
    "rvol_cumul_trade_vol_dev_pct",
    "rvol_cumul_flow_imbal_dev",
    "rvol_cumul_msg_dev",
    "rvol_bid_ask_add_asymmetry",
    "rvol_bid_ask_rem_asymmetry",
    "rvol_bid_ask_net_asymmetry",
    "rvol_aggbuy_aggsell_asymmetry",
    "rvol_lookback_trade_vol_mean_ratio",
    "rvol_lookback_trade_vol_max_ratio",
    "rvol_lookback_trade_vol_trend",
    "rvol_lookback_elevated_bars",
    "rvol_lookback_depressed_bars",
    "rvol_lookback_asymmetry_mean",
    "rvol_recent_vs_lookback_vol_ratio",
    "rvol_recent_vs_lookback_asymmetry",
}


def load_approach_data(symbol: str, dt: str, level_type: str) -> pd.DataFrame | None:
    path = LAKE_ROOT / f"silver/product_type=future/symbol={symbol}/table=market_by_price_10_{level_type}_approach/dt={dt}"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        return None
    return df


def get_computed_features(df: pd.DataFrame) -> List[str]:
    features = []
    for col in df.columns:
        for prefix in FEATURE_PREFIXES:
            if col.startswith(prefix):
                features.append(col)
                break
    return sorted(features)


def compute_feature_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    stats = []
    for feat in features:
        if feat not in df.columns:
            continue
        vals = df[feat]
        n_total = len(vals)
        n_null = vals.isna().sum()
        n_zero = (vals == 0).sum()
        n_inf = np.isinf(vals.replace([np.inf, -np.inf], np.nan).dropna()).sum() if vals.dtype in [np.float64, np.float32] else 0

        non_null = vals.dropna()
        if len(non_null) > 0:
            stats.append({
                "feature": feat,
                "n_total": n_total,
                "n_null": n_null,
                "pct_null": n_null / n_total * 100,
                "n_zero": n_zero,
                "pct_zero": n_zero / n_total * 100,
                "n_inf": n_inf,
                "min": non_null.min(),
                "max": non_null.max(),
                "mean": non_null.mean(),
                "std": non_null.std(),
                "median": non_null.median(),
                "q01": non_null.quantile(0.01),
                "q99": non_null.quantile(0.99),
                "n_unique": non_null.nunique(),
            })
        else:
            stats.append({
                "feature": feat,
                "n_total": n_total,
                "n_null": n_null,
                "pct_null": 100.0,
                "n_zero": 0,
                "pct_zero": 0.0,
                "n_inf": 0,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "median": np.nan,
                "q01": np.nan,
                "q99": np.nan,
                "n_unique": 0,
            })
    return pd.DataFrame(stats)


def flag_issues(stats_df: pd.DataFrame) -> pd.DataFrame:
    issues = []
    for _, row in stats_df.iterrows():
        feat = row["feature"]
        feat_issues = []

        is_rvol = feat.startswith("rvol_")
        is_deriv = feat.startswith("bar5s_deriv_")
        expected_null = feat in EXPECTED_NULL_FEATURES

        # Derivative features have expected NULLs at episode starts due to shift operations
        # Window sizes: w3=0.8%, w12=3.3%, w36=10%, w72=20% expected nulls (x2 for d2)
        if row["pct_null"] > 0 and not expected_null and not is_deriv:
            feat_issues.append(f"UNEXPECTED_NULL({row['pct_null']:.1f}%)")
        if row["pct_null"] == 100:
            feat_issues.append("ALL_NULL")

        # One-hot encoding features are expected to be mostly zero / constant within a level type
        is_onehot = feat in {"bar5s_approach_is_or_high", "bar5s_approach_is_or_low",
                             "bar5s_approach_is_pm_high", "bar5s_approach_is_pm_low",
                             "bar5s_approach_level_polarity"}
        # Depth at level is expected to be mostly zero (price rarely exactly at level)
        is_depth_at = feat == "bar5s_lvl_depth_at_qty_eob"
        # Lookback depressed bars can be 0 if no bars had depressed volume
        is_lookback_count = feat in {"rvol_lookback_depressed_bars", "rvol_lookback_elevated_bars"}

        if row["pct_zero"] > 95 and not is_rvol and not is_onehot and not is_depth_at:
            feat_issues.append(f"MOSTLY_ZERO({row['pct_zero']:.1f}%)")
        if row["n_inf"] > 0:
            feat_issues.append(f"HAS_INF({row['n_inf']})")
        if row["n_unique"] == 1 and row["pct_null"] < 100 and not is_onehot and not is_lookback_count:
            feat_issues.append("CONSTANT")
        if not np.isnan(row["std"]) and row["std"] == 0 and row["pct_null"] < 100 and not is_onehot and not is_lookback_count:
            feat_issues.append("ZERO_VARIANCE")
        if not np.isnan(row["max"]) and abs(row["max"]) > 1e10:
            feat_issues.append(f"EXTREME_MAX({row['max']:.2e})")
        if not np.isnan(row["min"]) and abs(row["min"]) > 1e10:
            feat_issues.append(f"EXTREME_MIN({row['min']:.2e})")

        issues.append({
            "feature": feat,
            "issues": "; ".join(feat_issues) if feat_issues else "OK",
            "has_issues": len(feat_issues) > 0,
        })
    return pd.DataFrame(issues)


def analyze_feature_groups(stats_df: pd.DataFrame) -> Dict[str, Dict]:
    groups = defaultdict(list)
    for _, row in stats_df.iterrows():
        feat = row["feature"]
        if feat.startswith("bar5s_approach_"):
            groups["approach"].append(row)
        elif feat.startswith("bar5s_cumul_"):
            groups["cumulative"].append(row)
        elif feat.startswith("bar5s_deriv_"):
            groups["derivative"].append(row)
        elif feat.startswith("bar5s_lvl_"):
            groups["level_relative"].append(row)
        elif feat.startswith("bar5s_setup_"):
            groups["setup_profile"].append(row)
        elif feat.startswith("rvol_"):
            groups["relative_volume"].append(row)

    summary = {}
    for group_name, rows in groups.items():
        if not rows:
            continue
        df_group = pd.DataFrame(rows)
        summary[group_name] = {
            "n_features": len(rows),
            "avg_pct_null": df_group["pct_null"].mean(),
            "max_pct_null": df_group["pct_null"].max(),
            "avg_pct_zero": df_group["pct_zero"].mean(),
            "features_with_issues": sum(1 for r in rows if r["pct_null"] > 50 or r["pct_zero"] > 95),
        }
    return summary


def run_sanity_check(symbol: str, dates: List[str], verbose: bool = True) -> Dict:
    all_results = {}

    for dt in dates:
        print(f"\n{'='*70}")
        print(f"DATE: {dt}")
        print("="*70)

        date_results = {}

        for level_type in LEVEL_TYPES:
            df = load_approach_data(symbol, dt, level_type)
            if df is None:
                print(f"\n  {level_type.upper()}: No data")
                continue

            print(f"\n  {level_type.upper()}: {len(df)} rows, {len(df.columns)} cols")

            features = get_computed_features(df)
            print(f"    Computed features: {len(features)}")

            stats = compute_feature_stats(df, features)
            issues = flag_issues(stats)
            group_summary = analyze_feature_groups(stats)

            merged = stats.merge(issues[["feature", "issues", "has_issues"]], on="feature")

            problem_features = merged[merged["has_issues"]]
            if len(problem_features) > 0 and verbose:
                print(f"\n    ⚠️  Features with issues ({len(problem_features)}):")
                for _, row in problem_features.iterrows():
                    print(f"      - {row['feature']}: {row['issues']}")

            print(f"\n    Feature Group Summary:")
            for group_name, summary in group_summary.items():
                status = "✅" if summary["features_with_issues"] == 0 else "⚠️"
                print(f"      {status} {group_name}: {summary['n_features']} features, "
                      f"avg null={summary['avg_pct_null']:.1f}%, avg zero={summary['avg_pct_zero']:.1f}%")

            date_results[level_type] = {
                "n_rows": len(df),
                "n_features": len(features),
                "stats": merged,
                "group_summary": group_summary,
                "n_issues": len(problem_features),
            }

        all_results[dt] = date_results

    return all_results


def compare_across_dates(results: Dict) -> pd.DataFrame:
    comparisons = []
    dates = sorted(results.keys())

    for level_type in LEVEL_TYPES:
        for dt in dates:
            if dt not in results or level_type not in results[dt]:
                continue
            r = results[dt][level_type]
            comparisons.append({
                "date": dt,
                "level_type": level_type,
                "n_rows": r["n_rows"],
                "n_features": r["n_features"],
                "n_issues": r["n_issues"],
            })

    return pd.DataFrame(comparisons)


def detailed_feature_comparison(results: Dict, feature_pattern: str = None) -> pd.DataFrame:
    rows = []
    dates = sorted(results.keys())

    for level_type in LEVEL_TYPES:
        all_features = set()
        for dt in dates:
            if dt in results and level_type in results[dt]:
                all_features.update(results[dt][level_type]["stats"]["feature"].tolist())

        for feat in sorted(all_features):
            if feature_pattern and feature_pattern not in feat:
                continue

            row = {"level_type": level_type, "feature": feat}
            for dt in dates:
                if dt in results and level_type in results[dt]:
                    stats = results[dt][level_type]["stats"]
                    feat_row = stats[stats["feature"] == feat]
                    if len(feat_row) > 0:
                        row[f"{dt}_mean"] = feat_row.iloc[0]["mean"]
                        row[f"{dt}_std"] = feat_row.iloc[0]["std"]
                        row[f"{dt}_pct_null"] = feat_row.iloc[0]["pct_null"]
                    else:
                        row[f"{dt}_mean"] = np.nan
                        row[f"{dt}_std"] = np.nan
                        row[f"{dt}_pct_null"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check approach features")
    parser.add_argument("--symbol", default="ESU5", help="Symbol to check")
    parser.add_argument("--dates", default="2025-06-11,2025-06-12,2025-06-13", help="Dates to check")
    parser.add_argument("--verbose", action="store_true", default=True, help="Show detailed issues")
    parser.add_argument("--compare", action="store_true", help="Show cross-date comparison")
    parser.add_argument("--feature", type=str, default=None, help="Filter features by pattern")

    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",")]

    results = run_sanity_check(args.symbol, dates, verbose=args.verbose)

    if args.compare:
        print("\n" + "="*70)
        print("CROSS-DATE COMPARISON")
        print("="*70)
        comparison = compare_across_dates(results)
        print(comparison.to_string(index=False))

    if args.feature:
        print(f"\n" + "="*70)
        print(f"FEATURE PATTERN: {args.feature}")
        print("="*70)
        detailed = detailed_feature_comparison(results, args.feature)
        print(detailed.to_string(index=False))

    total_issues = sum(
        results[dt][lt]["n_issues"]
        for dt in results
        for lt in results[dt]
    )

    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total dates checked: {len(dates)}")
    print(f"Total issues found: {total_issues}")
    if total_issues == 0:
        print("✅ All features pass sanity checks!")
    else:
        print("⚠️  Some features have potential issues - review above")

    return results


if __name__ == "__main__":
    main()
