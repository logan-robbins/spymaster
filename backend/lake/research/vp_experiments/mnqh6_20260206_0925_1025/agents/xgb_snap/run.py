"""XGBoost Snapshot (XGB_SNAP) experiment.

XGBoost on full spatial snapshot of pressure, vacuum, and spectrum fields
flattened to a wide feature vector. Walk-forward with expanding window.

Key idea: let the tree ensemble discover spatial patterns directly from
the raw 101-tick profiles rather than hand-engineering spatial features.

Features (per bin):
    1. pressure_variant[k] for k in [-25..+25] (51 ticks, center window) = 51
    2. vacuum_variant[k] for k in [-25..+25] = 51
    3. spectrum_score[k] for k in [-25..+25] = 51
    4. Mid-price returns rolling stats (mean/std at 50, 200, 600) = 6
    5. Total pressure bid vs ask (2) + total vacuum bid vs ask (2) = 4
    Total: 163 features

Walk-forward: retrain every 600 bins, min 3000 bins training.
XGBoost with early stopping.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xgboost as xgb

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

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CENTER_WINDOW = slice(25, 76)  # k=-25..+25 = 51 ticks
ROLLING_WINDOWS = [50, 200, 600]
MIN_TRAIN = 3000
RETRAIN_INTERVAL = 600
COOLDOWN_BINS = 30
PROB_THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "gamma": 0.1,
    "seed": 42,
    "nthread": 1,
    "verbosity": 0,
}
NUM_BOOST = 150
EARLY_STOP = 15


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


def rolling_mean_std(arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(arr)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    rm = np.zeros(n)
    rs = np.zeros(n)
    for i in range(w - 1, n):
        s = cs[i] - (cs[i - w] if i >= w else 0.0)
        s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
        m = s / w
        rm[i] = m
        rs[i] = np.sqrt(max(s2 / w - m * m, 0.0))
    return rm, rs


def build_features(grids: dict, mid_price: np.ndarray, n_bins: int) -> np.ndarray:
    features = []

    # 1. Center-window spatial profiles (51 each, 153 total)
    for col in ["pressure_variant", "vacuum_variant", "spectrum_score"]:
        features.append(grids[col][:, CENTER_WINDOW])

    # 2. Mid-price return rolling stats (6 features)
    ret = np.zeros(n_bins)
    ret[1:] = np.diff(mid_price) / TICK_SIZE
    for w in ROLLING_WINDOWS:
        rm, rs = rolling_mean_std(ret, w)
        features.append(rm.reshape(-1, 1))
        features.append(rs.reshape(-1, 1))

    # 3. Total pressure/vacuum by side (4 features)
    P = grids["pressure_variant"]
    V = grids["vacuum_variant"]
    features.append(P[:, 26:50].sum(axis=1).reshape(-1, 1))  # pressure bid
    features.append(P[:, 51:75].sum(axis=1).reshape(-1, 1))  # pressure ask
    features.append(V[:, 26:50].sum(axis=1).reshape(-1, 1))  # vacuum bid
    features.append(V[:, 51:75].sum(axis=1).reshape(-1, 1))  # vacuum ask

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("XGBoost Snapshot (XGB_SNAP)")
    print("=" * 70)

    # 1. Load
    print("Loading dataset ...", flush=True)
    cols = ["pressure_variant", "vacuum_variant", "spectrum_score"]
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
    print(f"  +1={(labels==1).sum()}, -1={(labels==-1).sum()}, 0={(labels==0).sum()}", flush=True)

    # 4. Walk-forward
    print(f"Walk-forward (min_train={MIN_TRAIN}, retrain={RETRAIN_INTERVAL}) ...", flush=True)
    t3 = time.perf_counter()

    binary_labels = np.zeros(n_bins, dtype=np.int32)
    binary_labels[labels == 1] = 1

    proba_up = np.full(n_bins, 0.5, dtype=np.float64)
    has_prediction = np.zeros(n_bins, dtype=bool)

    model = None
    last_train_bin = -1
    start_bin = max(WARMUP_BINS, MIN_TRAIN)

    for i in range(start_bin, n_bins):
        if model is None or (i - last_train_bin) >= RETRAIN_INTERVAL:
            train_mask = labels[:i] != 0
            if train_mask.sum() < 50:
                continue

            X_train = X[:i][train_mask]
            y_train = binary_labels[:i][train_mask]

            split = int(len(X_train) * 0.8)
            if split < 30 or (len(X_train) - split) < 10:
                continue

            dtrain = xgb.DMatrix(X_train[:split], label=y_train[:split])
            dval = xgb.DMatrix(X_train[split:], label=y_train[split:])

            model = xgb.train(
                XGB_PARAMS,
                dtrain,
                num_boost_round=NUM_BOOST,
                evals=[(dval, "val")],
                early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )
            last_train_bin = i

        dtest = xgb.DMatrix(X[i:i+1])
        prob = model.predict(dtest)[0]
        proba_up[i] = prob
        has_prediction[i] = True

    n_pred = has_prediction.sum()
    print(f"  {n_pred} predictions, time={time.perf_counter()-t3:.1f}s", flush=True)

    # 5. Signals
    all_results = []
    for prob_thr in PROB_THRESHOLDS:
        signals = []
        last_signal_bin = -COOLDOWN_BINS

        for i in range(n_bins):
            if not has_prediction[i]:
                continue
            if i - last_signal_bin < COOLDOWN_BINS:
                continue

            p = proba_up[i]
            if p >= prob_thr:
                signals.append({"bin_idx": i, "direction": "up"})
                last_signal_bin = i
            elif p <= (1.0 - prob_thr):
                signals.append({"bin_idx": i, "direction": "down"})
                last_signal_bin = i

        if not signals:
            all_results.append({
                "n_signals": 0, "tp_rate": float("nan"), "sl_rate": float("nan"),
                "timeout_rate": float("nan"), "events_per_hour": 0.0,
                "mean_pnl_ticks": float("nan"), "median_time_to_outcome_ms": float("nan"),
                "threshold": prob_thr, "cooldown_bins": COOLDOWN_BINS,
            })
            continue

        eval_result = evaluate_tp_sl(signals=signals, mid_price=mid_price, ts_ns=ts_ns)
        eval_result.pop("outcomes", None)
        eval_result["threshold"] = prob_thr
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
        print(f"  prob_thr={r['threshold']:.2f}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t", flush=True)

    # Feature importance
    if model is not None:
        imp = model.get_score(importance_type="gain")
        if imp:
            sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\nTop 10 features by gain:")
            for fname, score in sorted_imp:
                print(f"  {fname}: {score:.1f}")

    params = {
        "model": "XGBoost",
        "xgb_params": XGB_PARAMS,
        "num_boost": NUM_BOOST,
        "early_stop": EARLY_STOP,
        "n_features": X.shape[1],
        "center_window": "k=-25..+25 (51 ticks)",
        "min_train": MIN_TRAIN,
        "retrain_interval": RETRAIN_INTERVAL,
        "cooldown_bins": COOLDOWN_BINS,
    }

    out_path = write_results(
        agent_name="xgb_snap",
        experiment_name="xgboost_snapshot",
        params=params,
        results_by_threshold=all_results,
    )

    valid = [r for r in all_results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(f"\nBest: prob_thr={best['threshold']:.2f}  TP={best['tp_rate']:.1%}  "
              f"n={best['n_signals']}  PnL={best['mean_pnl_ticks']:+.2f}t")

    print(f"Results written to {out_path}")
    print(f"Total runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
