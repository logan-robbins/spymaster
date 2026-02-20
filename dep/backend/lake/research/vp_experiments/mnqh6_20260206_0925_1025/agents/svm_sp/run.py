"""SVM Spatial Profile (SVM_SP) experiment.

Uses LinearSVC on spatial pressure-vacuum profiles with rolling
statistical features over longer lookback windows.

Feature vector per bin (all from the 101-tick spatial grid):
    1. Spatial profile: pressure_variant - vacuum_variant at 21 sampled ticks
       (every 5th tick: k=-50,-45,...,0,...,+45,+50)
    2. Bid/ask asymmetry stats from v_add, v_pull over 3 bands (inner/mid/outer)
       computed with 600-bin rolling mean and std
    3. Rolling mean/std of mid_price returns over [50, 200, 600] windows

Labels: +1 if TP hit before SL (8t/4t), -1 if SL hit first, 0 if timeout.
        Only train/predict on +1/-1 (discard timeouts from training).

Walk-forward: train on [0..t-1], predict at t, retrain every 300 bins.
Minimum 1200 bins of training data before first prediction.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
from sklearn.svm import LinearSVC
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
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLED_K_INDICES = list(range(0, 101, 5))  # 21 ticks sampled every 5
BAND_DEFS = [
    ("inner", list(range(47, 50)), list(range(51, 54))),   # k=-3..-1, +1..+3
    ("mid",   list(range(39, 47)), list(range(54, 62))),   # k=-11..-4, +4..+11
    ("outer", list(range(27, 39)), list(range(62, 74))),   # k=-23..-12, +12..+23
]
ROLLING_WINDOWS = [50, 200, 600]
MIN_TRAIN_BINS = 1200
RETRAIN_INTERVAL = 300
COOLDOWN_BINS = 30
CONFIDENCE_THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8]


def compute_labels(mid_price: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute TP/SL labels for each bin. +1=TP up, -1=SL, 0=timeout/ambiguous."""
    labels = np.zeros(n_bins, dtype=np.int8)
    tp_dollars = TP_TICKS * TICK_SIZE
    sl_dollars = SL_TICKS * TICK_SIZE

    for i in range(n_bins):
        entry = mid_price[i]
        if entry <= 0:
            continue
        end = min(i + MAX_HOLD_BINS, n_bins)

        # Check both directions — which resolves first?
        up_tp = up_sl = down_tp = down_sl = end
        for j in range(i + 1, end):
            p = mid_price[j]
            if p <= 0:
                continue
            diff = p - entry
            if up_tp == end and diff >= tp_dollars:
                up_tp = j
            if up_sl == end and diff <= -sl_dollars:
                up_sl = j
            if down_tp == end and diff <= -tp_dollars:
                down_tp = j
            if down_sl == end and diff >= sl_dollars:
                down_sl = j
            if up_tp < end and up_sl < end and down_tp < end and down_sl < end:
                break

        # Determine which direction resolves favorably first
        up_resolve = min(up_tp, up_sl)
        down_resolve = min(down_tp, down_sl)

        if up_tp < up_sl and up_tp < end:
            labels[i] = 1   # long TP hit first
        elif down_tp < down_sl and down_tp < end:
            labels[i] = -1  # short TP hit first
        # else 0 (ambiguous or timeout)

    return labels


def build_features(
    pv_diff: np.ndarray,
    v_add: np.ndarray,
    v_pull: np.ndarray,
    mid_price: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Build feature matrix (n_bins, n_features)."""
    features_list = []

    # 1. Sampled spatial profile (21 features)
    spatial = pv_diff[:, SAMPLED_K_INDICES]
    features_list.append(spatial)

    # 2. Band asymmetry rolling stats
    for name, bid_cols, ask_cols in BAND_DEFS:
        add_asym = v_add[:, bid_cols].mean(axis=1) - v_add[:, ask_cols].mean(axis=1)
        pull_asym = v_pull[:, ask_cols].mean(axis=1) - v_pull[:, bid_cols].mean(axis=1)
        combined = add_asym + pull_asym

        for w in ROLLING_WINDOWS:
            # Rolling mean and std via cumsum
            cs = np.cumsum(combined)
            cs2 = np.cumsum(combined ** 2)

            rmean = np.zeros(n_bins)
            rstd = np.zeros(n_bins)
            for i in range(w - 1, n_bins):
                s = cs[i] - (cs[i - w] if i >= w else 0.0)
                s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
                m = s / w
                rmean[i] = m
                var = s2 / w - m * m
                rstd[i] = np.sqrt(max(var, 0.0))

            features_list.append(rmean.reshape(-1, 1))
            features_list.append(rstd.reshape(-1, 1))

    # 3. Mid-price return rolling stats
    returns = np.zeros(n_bins)
    returns[1:] = np.diff(mid_price) / TICK_SIZE  # in ticks
    for w in ROLLING_WINDOWS:
        cs = np.cumsum(returns)
        cs2 = np.cumsum(returns ** 2)
        rmean = np.zeros(n_bins)
        rstd = np.zeros(n_bins)
        for i in range(w - 1, n_bins):
            s = cs[i] - (cs[i - w] if i >= w else 0.0)
            s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
            m = s / w
            rmean[i] = m
            var = s2 / w - m * m
            rstd[i] = np.sqrt(max(var, 0.0))
        features_list.append(rmean.reshape(-1, 1))
        features_list.append(rstd.reshape(-1, 1))

    X = np.hstack(features_list)
    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("SVM Spatial Profile (SVM_SP)")
    print("=" * 70)

    # 1. Load
    print("Loading dataset ...", flush=True)
    ds = load_dataset(columns=["pressure_variant", "vacuum_variant", "v_add", "v_pull"])
    P = ds["pressure_variant"]
    V = ds["vacuum_variant"]
    v_add = ds["v_add"]
    v_pull = ds["v_pull"]
    mid_price = ds["mid_price"]
    ts_ns = ds["ts_ns"]
    n_bins = ds["n_bins"]
    print(f"  n_bins={n_bins}, load time={time.perf_counter()-t0:.1f}s", flush=True)

    pv_diff = P - V

    # 2. Features
    print("Building features ...", flush=True)
    t1 = time.perf_counter()
    X = build_features(pv_diff, v_add, v_pull, mid_price, n_bins)
    print(f"  shape={X.shape}, build time={time.perf_counter()-t1:.1f}s", flush=True)

    # 3. Labels
    print("Computing labels ...", flush=True)
    t2 = time.perf_counter()
    labels = compute_labels(mid_price, n_bins)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == -1)
    n_zero = np.sum(labels == 0)
    print(f"  +1={n_pos}, -1={n_neg}, 0={n_zero}, time={time.perf_counter()-t2:.1f}s", flush=True)

    # 4. Walk-forward prediction
    print(f"Walk-forward (min_train={MIN_TRAIN_BINS}, retrain_every={RETRAIN_INTERVAL}) ...", flush=True)
    t3 = time.perf_counter()

    predictions = np.zeros(n_bins, dtype=np.float64)  # decision function values
    pred_labels = np.zeros(n_bins, dtype=np.int8)
    has_prediction = np.zeros(n_bins, dtype=bool)

    model = None
    scaler = None
    last_train_bin = -1

    start_bin = max(WARMUP_BINS, MIN_TRAIN_BINS)

    for i in range(start_bin, n_bins):
        # Retrain?
        if model is None or (i - last_train_bin) >= RETRAIN_INTERVAL:
            # Training set: all bins with non-zero labels up to i
            train_mask = (labels[:i] != 0)
            if train_mask.sum() < 20:
                continue

            X_train = X[:i][train_mask]
            y_train = labels[:i][train_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            model = LinearSVC(
                C=0.1,
                loss="hinge",
                max_iter=2000,
                class_weight="balanced",
                dual=True,
            )
            model.fit(X_train_s, y_train)
            last_train_bin = i

        # Predict
        x_i = scaler.transform(X[i:i+1])
        dec = model.decision_function(x_i)[0]
        predictions[i] = dec
        pred_labels[i] = 1 if dec > 0 else -1
        has_prediction[i] = True

    n_predicted = has_prediction.sum()
    print(f"  {n_predicted} predictions, time={time.perf_counter()-t3:.1f}s", flush=True)

    # 5. Convert to signals with confidence thresholds
    all_results = []
    for conf_thr in CONFIDENCE_THRESHOLDS:
        signals = []
        last_signal_bin = -COOLDOWN_BINS

        for i in range(n_bins):
            if not has_prediction[i]:
                continue
            if abs(predictions[i]) < conf_thr:
                continue
            if i - last_signal_bin < COOLDOWN_BINS:
                continue

            direction = "up" if predictions[i] > 0 else "down"
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
                "threshold": conf_thr,
                "cooldown_bins": COOLDOWN_BINS,
            })
            continue

        eval_result = evaluate_tp_sl(
            signals=signals,
            mid_price=mid_price,
            ts_ns=ts_ns,
        )
        eval_result.pop("outcomes", None)
        eval_result["threshold"] = conf_thr
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
        print(f"  conf_thr={r['threshold']:.1f}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t", flush=True)

    params = {
        "model": "LinearSVC",
        "C": 0.1,
        "features": "spatial_profile_21 + band_asym_rolling + return_rolling",
        "n_features": X.shape[1],
        "rolling_windows": ROLLING_WINDOWS,
        "min_train_bins": MIN_TRAIN_BINS,
        "retrain_interval": RETRAIN_INTERVAL,
        "cooldown_bins": COOLDOWN_BINS,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
    }

    out_path = write_results(
        agent_name="svm_sp",
        experiment_name="svm_spatial_profile",
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
