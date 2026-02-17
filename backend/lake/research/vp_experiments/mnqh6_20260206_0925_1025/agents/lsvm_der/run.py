"""Linear SVM Derivative (LSVM_DER) experiment.

Online linear SVM (SGDClassifier with hinge loss) on rolling derivative
slope features with periodic batch retrain on 600-bin expanding window.

Key idea: uses ONLY derivative-chain features (v, a, j) — no pressure/vacuum
composites. Tests whether raw physics derivatives alone carry enough signal.

Features (per bin):
    1. 6 derivative columns x 3 bands = 18 asymmetry features
    2. 18 x rolling OLS slope (window=100) = 18 slope features
    3. 18 x rolling OLS slope (window=300) = 18 slope features (longer lookback)
    4. 6 derivative columns: spatial weighted sum across all 48 ticks
       (w=1/|k|) on each side, then divergence = 6 features
    Total: 60 features

Walk-forward with SGDClassifier warm_start for incremental updates.
Retrain from scratch every 1200 bins (full re-fit).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier
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
DERIV_COLS = ["v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull"]
BAND_DEFS = [
    ("inner", list(range(47, 50)), list(range(51, 54))),
    ("mid",   list(range(39, 47)), list(range(54, 62))),
    ("outer", list(range(27, 39)), list(range(62, 74))),
]
BID_COLS_FULL = np.arange(26, 50)   # k=-24..-1
ASK_COLS_FULL = np.arange(51, 75)   # k=+1..+24
BID_WEIGHTS = 1.0 / (50 - BID_COLS_FULL).astype(np.float64)
BID_WEIGHTS /= BID_WEIGHTS.sum()
ASK_WEIGHTS = 1.0 / (ASK_COLS_FULL - 50).astype(np.float64)
ASK_WEIGHTS /= ASK_WEIGHTS.sum()

SLOPE_WINDOWS = [100, 300]
MIN_TRAIN = 1500
RETRAIN_INTERVAL = 600
FULL_REFIT_INTERVAL = 1200
COOLDOWN_BINS = 30
CONFIDENCE_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 1.0]


def compute_labels(mid_price: np.ndarray, n_bins: int) -> np.ndarray:
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
                labels[i] = 1; break
            elif diff <= -tp_d:
                labels[i] = -1; break
            elif diff <= -sl_d:
                labels[i] = -1; break
            elif diff >= sl_d:
                labels[i] = 1; break
    return labels


def build_features(grids: dict, n_bins: int) -> np.ndarray:
    features = []

    # 1. Band asymmetries (18 features)
    band_asyms = []
    for col in DERIV_COLS:
        g = grids[col]
        for _, bid_cols, ask_cols in BAND_DEFS:
            if "add" in col:
                asym = g[:, bid_cols].mean(axis=1) - g[:, ask_cols].mean(axis=1)
            else:
                asym = g[:, ask_cols].mean(axis=1) - g[:, bid_cols].mean(axis=1)
            features.append(asym.reshape(-1, 1))
            band_asyms.append(asym)

    # 2. Rolling OLS slopes of band asymmetries (18 x 2 = 36 features)
    for asym in band_asyms:
        for w in SLOPE_WINDOWS:
            slope = rolling_ols_slope(asym, w)
            slope = np.nan_to_num(slope, nan=0.0)
            features.append(slope.reshape(-1, 1))

    # 3. Full-width distance-weighted divergences (6 features)
    for col in DERIV_COLS:
        g = grids[col]
        bid_wm = g[:, BID_COLS_FULL] @ BID_WEIGHTS
        ask_wm = g[:, ASK_COLS_FULL] @ ASK_WEIGHTS
        if "add" in col:
            div = bid_wm - ask_wm
        else:
            div = ask_wm - bid_wm
        features.append(div.reshape(-1, 1))

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("Linear SVM Derivative (LSVM_DER)")
    print("=" * 70)

    # 1. Load
    print("Loading dataset ...", flush=True)
    ds = load_dataset(columns=DERIV_COLS)
    mid_price = ds["mid_price"]
    ts_ns = ds["ts_ns"]
    n_bins = ds["n_bins"]
    grids = {c: ds[c] for c in DERIV_COLS}
    print(f"  n_bins={n_bins}, load={time.perf_counter()-t0:.1f}s", flush=True)

    # 2. Features
    print("Building features ...", flush=True)
    t1 = time.perf_counter()
    X = build_features(grids, n_bins)
    print(f"  shape={X.shape}, time={time.perf_counter()-t1:.1f}s", flush=True)

    # 3. Labels
    print("Computing labels ...", flush=True)
    t2 = time.perf_counter()
    labels = compute_labels(mid_price, n_bins)
    print(f"  +1={(labels==1).sum()}, -1={(labels==-1).sum()}, 0={(labels==0).sum()}", flush=True)

    # 4. Walk-forward with SGDClassifier
    print(f"Walk-forward (min_train={MIN_TRAIN}, retrain={RETRAIN_INTERVAL}) ...", flush=True)
    t3 = time.perf_counter()

    decisions = np.zeros(n_bins, dtype=np.float64)
    has_prediction = np.zeros(n_bins, dtype=bool)

    model = None
    scaler = None
    last_train_bin = -1
    start_bin = max(WARMUP_BINS, MIN_TRAIN)

    for i in range(start_bin, n_bins):
        need_retrain = (model is None or
                       (i - last_train_bin) >= RETRAIN_INTERVAL)

        if need_retrain:
            train_mask = labels[:i] != 0
            if train_mask.sum() < 30:
                continue

            X_train = X[:i][train_mask]
            y_train = labels[:i][train_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            # Full refit or partial update
            full_refit = (model is None or
                         (i - last_train_bin) >= FULL_REFIT_INTERVAL)

            if full_refit:
                model = SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=1000,
                    class_weight="balanced",
                    warm_start=True,
                    random_state=42,
                )
                model.fit(X_train_s, y_train)
            else:
                # Partial fit on recent data
                recent_mask = np.zeros(i, dtype=bool)
                recent_mask[max(0, i - RETRAIN_INTERVAL):i] = True
                recent_mask &= (labels[:i] != 0)
                if recent_mask.sum() > 5:
                    X_recent = scaler.transform(X[:i][recent_mask])
                    y_recent = labels[:i][recent_mask]
                    model.partial_fit(X_recent, y_recent)

            last_train_bin = i

        x_i = scaler.transform(X[i:i+1])
        dec = model.decision_function(x_i)[0]
        decisions[i] = dec
        has_prediction[i] = True

    n_pred = has_prediction.sum()
    print(f"  {n_pred} predictions, time={time.perf_counter()-t3:.1f}s", flush=True)

    # 5. Generate signals
    all_results = []
    for conf in CONFIDENCE_THRESHOLDS:
        signals = []
        last_signal_bin = -COOLDOWN_BINS

        for i in range(n_bins):
            if not has_prediction[i]:
                continue
            if abs(decisions[i]) < conf:
                continue
            if i - last_signal_bin < COOLDOWN_BINS:
                continue

            direction = "up" if decisions[i] > 0 else "down"
            signals.append({"bin_idx": i, "direction": direction})
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
                "threshold": conf,
                "cooldown_bins": COOLDOWN_BINS,
            })
            continue

        eval_result = evaluate_tp_sl(
            signals=signals,
            mid_price=mid_price,
            ts_ns=ts_ns,
        )
        eval_result.pop("outcomes", None)
        eval_result["threshold"] = conf
        eval_result["cooldown_bins"] = COOLDOWN_BINS
        all_results.append(eval_result)

    # 6. Report
    print("\nResults:", flush=True)
    for r in all_results:
        tp = r["tp_rate"]
        tp_str = f"{tp:.1%}" if not np.isnan(tp) else "N/A"
        n = r["n_signals"]
        pnl = r.get("mean_pnl_ticks", float("nan"))
        pnl_str = f"{pnl:+.2f}" if not np.isnan(pnl) else "N/A"
        print(f"  conf={r['threshold']:.1f}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t", flush=True)

    params = {
        "model": "SGDClassifier (hinge loss = linear SVM)",
        "alpha": 1e-4,
        "n_features": X.shape[1],
        "slope_windows": SLOPE_WINDOWS,
        "min_train": MIN_TRAIN,
        "retrain_interval": RETRAIN_INTERVAL,
        "full_refit_interval": FULL_REFIT_INTERVAL,
        "cooldown_bins": COOLDOWN_BINS,
    }

    out_path = write_results(
        agent_name="lsvm_der",
        experiment_name="linear_svm_derivative",
        params=params,
        results_by_threshold=all_results,
    )

    valid = [r for r in all_results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(f"\nBest: conf={best['threshold']:.1f}  TP={best['tp_rate']:.1%}  "
              f"n={best['n_signals']}  PnL={best['mean_pnl_ticks']:+.2f}t")

    print(f"Results written to {out_path}")
    print(f"Total runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
