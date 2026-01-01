"""
Episode-level BREAK vs REJECT driver analysis for PM/OR level interactions.

Why this exists:
- Avoids the "30s bars are samples" pitfall: analysis is done per interaction episode
- Focuses on trader-relevant TA anchors: PM_HIGH/PM_LOW/OR_HIGH/OR_LOW
- Ranks which parts of the 144D episode vector explain BREAK vs REJECT (by direction/time bucket)
- Optionally compares against Pentaview stream proxies (sigma_s, sigma_r, sigma_b_slope, sigma_p_slope, sigma_d)

Usage examples:
  uv run python scripts/analyze_level_break_drivers.py --start 2025-11-03 --end 2025-12-19 --version v4.0.0
  uv run python scripts/analyze_level_break_drivers.py --date 2025-12-18 --version v4.0.0 --horizon 4min
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from src.ml.constants import VECTOR_SECTIONS
from src.ml.episode_vector import construct_episodes_from_events, get_feature_names
from src.ml.normalization import load_normalization_stats


PM_OR_LEVELS_DEFAULT = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
OUTCOME_KEEP = {"BREAK", "REJECT"}


def _parse_levels(val: str) -> List[str]:
    items = [v.strip() for v in val.split(",") if v.strip()]
    if not items:
        raise ValueError("levels must be a non-empty comma-separated list")
    return items


def _iter_available_dates(signals_root: Path) -> List[str]:
    dates: List[str] = []
    if not signals_root.exists():
        return dates
    for d in signals_root.glob("date=*"):
        if not d.is_dir():
            continue
        # Partition format: date=YYYY-MM-DD
        date_str = d.name.split("date=", 1)[-1]
        if date_str:
            dates.append(date_str)
    return sorted(set(dates))


def _filter_dates(dates: Sequence[str], start: Optional[str], end: Optional[str], date: Optional[str]) -> List[str]:
    if date is not None:
        if date not in dates:
            raise FileNotFoundError(f"Requested date={date} not found under dataset partitions.")
        return [date]
    out = list(dates)
    if start is not None:
        out = [d for d in out if d >= start]
    if end is not None:
        out = [d for d in out if d <= end]
    return out


def _safe_bool_series(s: pd.Series) -> np.ndarray:
    return s.astype(str).to_numpy()


def _build_episode_dataset_for_dates(
    *,
    data_root: Path,
    version: str,
    dates: Sequence[str],
    normalization_stats_path: Path,
    include_streams: bool,
) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Returns:
      X: (N, 144) float32 normalized episode vectors
      meta: (N, *) episode metadata aligned to X
      streams: optional (N, *) merged stream features aligned to X
    """
    stats = load_normalization_stats(normalization_stats_path)

    X_all: List[np.ndarray] = []
    meta_all: List[pd.DataFrame] = []
    streams_all: List[pd.DataFrame] = []

    for date in dates:
        signals_path = data_root / "silver" / "features" / "es_pipeline" / f"version={version}" / f"date={date}" / "signals.parquet"
        state_path = data_root / "silver" / "state" / "es_level_state" / f"version={version}" / f"date={date}" / "state.parquet"

        if not signals_path.exists():
            raise FileNotFoundError(f"Missing signals parquet: {signals_path}")
        if not state_path.exists():
            raise FileNotFoundError(f"Missing state parquet: {state_path}")

        events_df = pd.read_parquet(signals_path)
        state_df = pd.read_parquet(state_path)

        X, meta, _seq = construct_episodes_from_events(
            events_df=events_df,
            state_df=state_df,
            normalization_stats=stats,
        )

        if len(X) == 0 or meta.empty:
            continue

        # Some historical runs produced NaNs in a small number of vector dimensions
        # (schema drift / missing upstream columns). Treat NaNs as "neutral" (0 after normalization).
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        meta = meta.copy()
        meta["date"] = meta["date"].astype(str)

        X_all.append(X.astype(np.float32, copy=False))
        meta_all.append(meta)

        if include_streams:
            # Stream partition format: date=YYYY-MM-DD_00:00:00
            stream_dir = data_root / "gold" / "streams" / "pentaview" / f"version={version}" / f"date={date}_00:00:00"
            stream_path = stream_dir / "stream_bars.parquet"
            if stream_path.exists():
                sb = pd.read_parquet(stream_path)
                # Align exactly on the episode anchor (timestamp + level_kind).
                # NOTE: direction in historical stream_bars can be inconsistent depending on how it was inferred
                # from signed distance; timestamp+level_kind is unambiguous for a given market snapshot.
                key_cols = ["timestamp", "level_kind"]
                keep_cols = [
                    "sigma_s", "sigma_r", "sigma_b_slope", "sigma_p_slope", "sigma_d",
                    "sigma_m", "sigma_f", "sigma_b", "sigma_p",
                    "alignment", "alignment_adj", "divergence",
                ]
                keep_cols = [c for c in keep_cols if c in sb.columns]
                sb_small = sb[key_cols + keep_cols].drop_duplicates(key_cols)
                merged = meta[key_cols].merge(sb_small, on=key_cols, how="left")
                streams_all.append(merged[keep_cols])
            else:
                # Keep alignment: fill NaNs if streams missing for this date.
                streams_all.append(pd.DataFrame(index=meta.index))

    if not X_all:
        return np.zeros((0, 144), dtype=np.float32), pd.DataFrame(), None

    X_cat = np.concatenate(X_all, axis=0)
    meta_cat = pd.concat(meta_all, ignore_index=True)

    streams_cat: Optional[pd.DataFrame] = None
    if include_streams:
        streams_cat = pd.concat(streams_all, ignore_index=True)

    return X_cat, meta_cat, streams_cat


@dataclass
class SegmentReport:
    key: str
    n: int
    break_rate: float
    auc_mean: Optional[float]
    auc_std: Optional[float]
    top_break_drivers: List[Tuple[str, float]]
    top_reject_drivers: List[Tuple[str, float]]


def _logo_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    # Needs at least 2 classes.
    if len(np.unique(y)) < 2:
        return None, None

    logo = LeaveOneGroupOut()
    aucs: List[float] = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        y_train = y[train_idx]
        y_test = y[test_idx]
        # Skip folds with a single class.
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=2000,
            class_weight="balanced",
        )
        model.fit(X[train_idx], y_train)
        p = model.predict_proba(X[test_idx])[:, 1]
        aucs.append(float(roc_auc_score(y_test, p)))

    if not aucs:
        return None, None
    return float(np.mean(aucs)), float(np.std(aucs))


def _fit_sparse_lr_drivers(X: np.ndarray, y: np.ndarray, feature_names: List[str], top_k: int = 10) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    if len(np.unique(y)) < 2:
        return [], []

    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=5000,
        class_weight="balanced",
    )
    model.fit(X, y)
    coefs = model.coef_.reshape(-1)

    pairs = list(zip(feature_names, coefs.astype(float)))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    break_drivers = [(f, c) for f, c in pairs if c > 0][:top_k]
    reject_drivers = [(f, abs(c)) for f, c in pairs if c < 0][:top_k]
    return break_drivers, reject_drivers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--version", type=str, default="v4.0.0")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--horizon", type=str, choices=["2min", "4min", "8min"], default="4min")
    parser.add_argument("--levels", type=str, default=",".join(PM_OR_LEVELS_DEFAULT))
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--include-streams", action="store_true", default=True)
    parser.add_argument("--no-streams", dest="include_streams", action="store_false")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    version = args.version
    levels = set(_parse_levels(args.levels))
    horizon_col = f"outcome_{args.horizon}"

    signals_root = data_root / "silver" / "features" / "es_pipeline" / f"version={version}"
    dates = _iter_available_dates(signals_root)
    if not dates:
        raise FileNotFoundError(f"No date partitions found under {signals_root}")
    dates = _filter_dates(dates, args.start, args.end, args.date)

    stats_path = data_root / "gold" / "normalization" / "current.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")

    feature_names = get_feature_names()

    X, meta, streams = _build_episode_dataset_for_dates(
        data_root=data_root,
        version=version,
        dates=dates,
        normalization_stats_path=stats_path,
        include_streams=args.include_streams,
    )

    if meta.empty or len(X) == 0:
        raise ValueError("No episodes produced for the requested date range.")

    # Filter to PM/OR only
    keep_mask = meta["level_kind"].isin(levels)
    meta = meta.loc[keep_mask].reset_index(drop=True)
    X = X[keep_mask.to_numpy()]
    if streams is not None:
        streams = streams.loc[keep_mask].reset_index(drop=True)

    # Filter outcomes
    if horizon_col not in meta.columns:
        raise ValueError(f"Missing label column in episode metadata: {horizon_col}")
    meta[horizon_col] = meta[horizon_col].astype(str)
    keep_outcome = meta[horizon_col].isin(OUTCOME_KEEP)
    meta = meta.loc[keep_outcome].reset_index(drop=True)
    X = X[keep_outcome.to_numpy()]
    if streams is not None:
        streams = streams.loc[keep_outcome].reset_index(drop=True)

    # Binary label: 1=BREAK, 0=REJECT
    y = (meta[horizon_col].astype(str) == "BREAK").to_numpy(dtype=np.int8)
    groups = meta["date"].astype(str).to_numpy()

    # Segment key
    meta["segment"] = (
        meta["level_kind"].astype(str) + "|" +
        meta["direction"].astype(str) + "|" +
        meta["time_bucket"].astype(str)
    )

    # Overall summary
    overall_auc_mean, overall_auc_std = _logo_auc(X, y, groups)
    overall_break_rate = float(y.mean()) if len(y) else 0.0

    # Segment reports
    reports: List[SegmentReport] = []
    for seg, idxs in meta.groupby("segment").groups.items():
        idxs = np.asarray(list(idxs), dtype=np.int64)
        n = int(len(idxs))
        if n < args.min_samples:
            continue

        y_seg = y[idxs]
        if len(np.unique(y_seg)) < 2:
            continue

        X_seg = X[idxs]
        g_seg = groups[idxs]
        auc_mean, auc_std = _logo_auc(X_seg, y_seg, g_seg)
        break_drivers, reject_drivers = _fit_sparse_lr_drivers(X_seg, y_seg, feature_names)

        reports.append(
            SegmentReport(
                key=str(seg),
                n=n,
                break_rate=float(y_seg.mean()),
                auc_mean=auc_mean,
                auc_std=auc_std,
                top_break_drivers=break_drivers,
                top_reject_drivers=reject_drivers,
            )
        )

    reports.sort(key=lambda r: (-(r.auc_mean or 0.0), -r.n))

    # Predictable segment convenience view (per PENTAVIEW_RESEARCH.md)
    predictable_mask = (
        meta["direction"].astype(str).eq("UP") &
        meta["level_kind"].isin({"PM_LOW", "OR_LOW"}) &
        (~meta["time_bucket"].astype(str).eq("T0_15"))
    )
    pred = {
        "n": int(predictable_mask.sum()),
        "break_rate": float(y[predictable_mask.to_numpy()].mean()) if int(predictable_mask.sum()) else None,
    }

    # Optional: stream-only baseline AUC on predictable segment
    stream_baseline = None
    if args.include_streams and streams is not None and not streams.empty:
        stream_cols = [c for c in ["sigma_s", "sigma_r", "sigma_b_slope", "sigma_p_slope", "sigma_d"] if c in streams.columns]
        if stream_cols and int(predictable_mask.sum()) >= args.min_samples:
            Xs = streams.loc[predictable_mask, stream_cols].fillna(0.0).to_numpy(dtype=np.float64)
            ys = y[predictable_mask.to_numpy()].astype(np.int8)
            gs = groups[predictable_mask.to_numpy()]
            auc_mean, auc_std = _logo_auc(Xs, ys, gs)
            stream_baseline = {
                "features": stream_cols,
                "auc_mean": auc_mean,
                "auc_std": auc_std,
            }

    payload: Dict[str, Any] = {
        "config": {
            "data_root": str(data_root),
            "version": version,
            "dates": list(dates),
            "levels": sorted(levels),
            "horizon": args.horizon,
            "min_samples": args.min_samples,
            "include_streams": bool(args.include_streams),
        },
        "overall": {
            "n": int(len(meta)),
            "break_rate": overall_break_rate,
            "auc_mean": overall_auc_mean,
            "auc_std": overall_auc_std,
        },
        "predictable_segment": pred,
        "stream_baseline": stream_baseline,
        "segments": [
            {
                "segment": r.key,
                "n": r.n,
                "break_rate": r.break_rate,
                "auc_mean": r.auc_mean,
                "auc_std": r.auc_std,
                "top_break_drivers": [{"feature": f, "coef": c} for f, c in r.top_break_drivers],
                "top_reject_drivers": [{"feature": f, "coef": c} for f, c in r.top_reject_drivers],
            }
            for r in reports
        ],
    }

    print(json.dumps(payload, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

