"""KNN Cluster Regime (KNN_CL) experiment.

Uses K-Nearest Neighbors to classify current microstructure state by finding
the most similar historical states and voting on direction.

Approach:
    1. Build a compact state vector from derivative asymmetries + spatial features
    2. Label each historical bin with TP/SL outcome
    3. For each new bin, find K nearest historical neighbors
    4. Vote: if majority of neighbors are +1, signal up; if -1, signal down
    5. Confidence = margin of majority vote

Key differences from other experiments:
    - Non-parametric: no model fitting, pure similarity matching
    - Spatial awareness: includes sampled pressure profile as features
    - Longer lookback: uses 600-bin rolling normalization for stationarity
    - Distance metric: Euclidean on standardized features

Walk-forward: expanding KNN pool, re-standardize every 300 bins.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_harness import (
    WARMUP_BINS,
    TICK_SIZE,
    TP_TICKS,
    SL_TICKS,
    MAX_HOLD_BINS,
    load_dataset,
    evaluate_tp_sl,
    write_results,
    rolling_ols_slope,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_NEIGHBORS = [5, 11, 21, 31]
BAND_DEFS = [
    ("inner", list(range(47, 50)), list(range(51, 54))),
    ("mid",   list(range(39, 47)), list(range(54, 62))),
    ("outer", list(range(27, 39)), list(range(62, 74))),
]
SPATIAL_SAMPLE_COLS = list(range(0, 101, 10))  # 11 sampled ticks
MIN_TRAIN = 1800
RESTANDARDIZE_INTERVAL = 300
COOLDOWN_BINS = 30


def compute_labels(mid_price: np.ndarray, n_bins: int) -> np.ndarray:
    """Label: +1 if long TP first, -1 if short TP first, 0 otherwise."""
    labels = np.zeros(n_bins, dtype=np.int8)
    tp_d = TP_TICKS * TICK_SIZE
    sl_d = SL_TICKS * TICK_SIZE

    for i in range(n_bins):
        entry = mid_price[i]
        if entry <= 0:
            continue
        end = min(i + MAX_HOLD_BINS, n_bins)
        for j in range(i + 1, end):
            p = mid_price[j]
            if p <= 0:
                continue
            diff = p - entry
            if diff >= tp_d:
                labels[i] = 1
                break
            elif diff <= -tp_d:
                labels[i] = -1
                break
            elif diff <= -sl_d:
                labels[i] = -1
                break
            elif diff >= sl_d:
                labels[i] = 1
                break
    return labels


def build_features(
    grids: dict,
    mid_price: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Build compact state vector."""
    features = []

    # 1. Derivative asymmetries (6 columns x 3 bands = 18 features)
    for col in ["v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull"]:
        g = grids[col]
        for _, bid_cols, ask_cols in BAND_DEFS:
            if "add" in col:
                asym = g[:, bid_cols].mean(axis=1) - g[:, ask_cols].mean(axis=1)
            else:
                asym = g[:, ask_cols].mean(axis=1) - g[:, bid_cols].mean(axis=1)
            features.append(asym.reshape(-1, 1))

    # 2. Rolling slopes of combined asymmetry (3 windows)
    combined = np.zeros(n_bins)
    for col in ["v_add", "v_pull"]:
        g = grids[col]
        for _, bid_cols, ask_cols in BAND_DEFS:
            if "add" in col:
                combined += g[:, bid_cols].mean(axis=1) - g[:, ask_cols].mean(axis=1)
            else:
                combined += g[:, ask_cols].mean(axis=1) - g[:, bid_cols].mean(axis=1)

    for w in [10, 50, 200]:
        slope = rolling_ols_slope(combined, w)
        slope = np.nan_to_num(slope, nan=0.0)
        features.append(slope.reshape(-1, 1))

    # 3. Sampled spatial PV profile (11 features)
    pv = grids["pressure_variant"] - grids["vacuum_variant"]
    features.append(pv[:, SPATIAL_SAMPLE_COLS])

    # 4. Mid-price momentum features (3 features)
    ret = np.zeros(n_bins)
    ret[1:] = np.diff(mid_price) / TICK_SIZE
    for w in [20, 100, 600]:
        rm = np.convolve(ret, np.ones(w) / w, mode="full")[:n_bins]
        features.append(rm.reshape(-1, 1))

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("KNN Cluster Regime (KNN_CL)")
    print("=" * 70)

    # 1. Load
    print("Loading dataset ...", flush=True)
    cols = ["v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull",
            "pressure_variant", "vacuum_variant"]
    ds = load_dataset(columns=cols)
    mid_price = ds["mid_price"]
    ts_ns = ds["ts_ns"]
    n_bins = ds["n_bins"]
    grids = {c: ds[c] for c in cols}
    print(f"  n_bins={n_bins}, load={time.perf_counter()-t0:.1f}s", flush=True)

    # 2. Features
    print("Building features ...", flush=True)
    t1 = time.perf_counter()
    X = build_features(grids, mid_price, n_bins)
    print(f"  shape={X.shape}, time={time.perf_counter()-t1:.1f}s", flush=True)

    # 3. Labels
    print("Computing labels ...", flush=True)
    t2 = time.perf_counter()
    labels = compute_labels(mid_price, n_bins)
    print(f"  +1={(labels==1).sum()}, -1={(labels==-1).sum()}, 0={(labels==0).sum()}, "
          f"time={time.perf_counter()-t2:.1f}s", flush=True)

    # 4. Walk-forward for each K
    all_results = []
    start_bin = max(WARMUP_BINS, MIN_TRAIN)

    for k in K_NEIGHBORS:
        print(f"\nKNN k={k} walk-forward ...", flush=True)
        t3 = time.perf_counter()

        predictions = np.zeros(n_bins, dtype=np.float64)
        has_prediction = np.zeros(n_bins, dtype=bool)
        model = None
        scaler = None
        last_fit_bin = -1

        for i in range(start_bin, n_bins):
            if model is None or (i - last_fit_bin) >= RESTANDARDIZE_INTERVAL:
                train_mask = labels[:i] != 0
                if train_mask.sum() < k + 10:
                    continue

                X_train = X[:i][train_mask]
                y_train = labels[:i][train_mask]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)

                model = KNeighborsClassifier(
                    n_neighbors=min(k, len(X_train)),
                    weights="distance",
                    metric="euclidean",
                    n_jobs=1,
                )
                model.fit(X_train_s, y_train)
                last_fit_bin = i

            x_i = scaler.transform(X[i:i+1])
            proba = model.predict_proba(x_i)[0]
            classes = model.classes_

            # Get probability of +1 and -1
            p_up = proba[classes == 1][0] if 1 in classes else 0.0
            p_down = proba[classes == -1][0] if -1 in classes else 0.0

            # Store signed confidence
            predictions[i] = p_up - p_down
            has_prediction[i] = True

        n_pred = has_prediction.sum()
        print(f"  {n_pred} predictions, time={time.perf_counter()-t3:.1f}s", flush=True)

        # Generate signals at various margin thresholds
        for margin in [0.0, 0.1, 0.2, 0.3]:
            signals = []
            last_signal_bin = -COOLDOWN_BINS

            for i in range(n_bins):
                if not has_prediction[i]:
                    continue
                if i - last_signal_bin < COOLDOWN_BINS:
                    continue

                if predictions[i] > margin:
                    signals.append({"bin_idx": i, "direction": "up"})
                    last_signal_bin = i
                elif predictions[i] < -margin:
                    signals.append({"bin_idx": i, "direction": "down"})
                    last_signal_bin = i

            if not signals:
                all_results.append({
                    "n_signals": 0,
                    "tp_rate": float("nan"),
                    "sl_rate": float("nan"),
                    "timeout_rate": float("nan"),
                    "events_per_hour": 0.0,
                    "mean_pnl_ticks": float("nan"),
                    "median_time_to_outcome_ms": float("nan"),
                    "threshold": f"k={k}_margin={margin}",
                    "cooldown_bins": COOLDOWN_BINS,
                })
                continue

            eval_result = evaluate_tp_sl(
                signals=signals,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )
            eval_result.pop("outcomes", None)
            eval_result["threshold"] = f"k={k}_margin={margin}"
            eval_result["cooldown_bins"] = COOLDOWN_BINS
            all_results.append(eval_result)

    # 5. Report
    print("\nResults:", flush=True)
    for r in all_results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "N/A"
        n = r["n_signals"]
        pnl = r.get("mean_pnl_ticks", float("nan"))
        pnl_str = f"{pnl:+.2f}" if not np.isnan(pnl) else "N/A"
        print(f"  {r['threshold']:>20s}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t", flush=True)

    params = {
        "model": "KNeighborsClassifier",
        "k_values": K_NEIGHBORS,
        "weights": "distance",
        "metric": "euclidean",
        "n_features": X.shape[1],
        "min_train": MIN_TRAIN,
        "restandardize_interval": RESTANDARDIZE_INTERVAL,
        "cooldown_bins": COOLDOWN_BINS,
    }

    out_path = write_results(
        agent_name="knn_cl",
        experiment_name="knn_cluster_regime",
        params=params,
        results_by_threshold=all_results,
    )

    valid = [r for r in all_results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(f"\nBest: {best['threshold']}  TP={best['tp_rate']:.1%}  "
              f"n={best['n_signals']}  PnL={best['mean_pnl_ticks']:+.2f}t")

    print(f"Results written to {out_path}")
    print(f"Total runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
