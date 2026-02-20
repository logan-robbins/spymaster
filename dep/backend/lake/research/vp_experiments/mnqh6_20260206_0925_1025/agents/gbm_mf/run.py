"""LightGBM Multi-Feature (GBM_MF) experiment.

Gradient boosted trees on a rich feature vector from multiple derivative
channels with 300/600-bin lookback windows. Walk-forward with expanding
window retrain.

Feature vector (per bin):
    1. Derivative asymmetries: for each of {v_add, v_pull, a_add, a_pull, j_add, j_pull}
       compute bid-ask asymmetry in 3 bands (inner/mid/outer) = 18 features
    2. Rolling OLS slopes of the 18 asymmetries over window=50 = 18 features
    3. Pressure-vacuum net at k=0 (spot) and rolling mean/std over [50, 300] = 5 features
    4. Mid-price return stats: rolling mean/std/skew over [50, 200, 600] = 9 features
    5. Spread (best_ask - best_bid) and rolling volatility = 3 features
    Total: ~53 features

Label: direction of first TP/SL exit (+1 long TP, -1 short TP, excluded if timeout)
Walk-forward: retrain every 600 bins, min 2400 bins training.
LightGBM with early stopping on last 20% of train set.
"""
from __future__ import annotations

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import lightgbm as lgb

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

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BAND_DEFS = [
    ("inner", list(range(47, 50)), list(range(51, 54)), 3),
    ("mid",   list(range(39, 47)), list(range(54, 62)), 8),
    ("outer", list(range(27, 39)), list(range(62, 74)), 12),
]
DERIV_COLS = ["v_add", "v_pull", "a_add", "a_pull", "j_add", "j_pull"]
SLOPE_WINDOW = 50
ROLLING_WINDOWS = [50, 200, 600]
MIN_TRAIN_BINS = 2400
RETRAIN_INTERVAL = 600
COOLDOWN_BINS = 30
PROB_THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": 1,
    "seed": 42,
}
NUM_BOOST_ROUND = 200
EARLY_STOPPING = 20


def compute_labels(mid_price: np.ndarray, n_bins: int) -> np.ndarray:
    """Label each bin: +1 if long TP resolves first, -1 if short TP, 0 if timeout."""
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
            elif diff <= -sl_d:
                # Check: is this a short TP or long SL?
                # For labeling: price went down by SL amount — could be short TP
                if diff <= -tp_d:
                    labels[i] = -1
                else:
                    labels[i] = -1  # SL on long = direction was down
                break
            elif diff <= -tp_d:
                labels[i] = -1
                break
            elif diff >= sl_d:
                labels[i] = 1
                break

    return labels


def rolling_mean_std(arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Efficient rolling mean and std using cumsum."""
    n = len(arr)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    rmean = np.zeros(n)
    rstd = np.zeros(n)
    for i in range(w - 1, n):
        s = cs[i] - (cs[i - w] if i >= w else 0.0)
        s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
        m = s / w
        rmean[i] = m
        rstd[i] = np.sqrt(max(s2 / w - m * m, 0.0))
    return rmean, rstd


def build_features(grids: dict, mid_price: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix."""
    features = []
    names = []

    # 1. Derivative asymmetries (18 features)
    asym_signals = []
    for col_name in DERIV_COLS:
        grid = grids[col_name]
        for band_name, bid_cols, ask_cols, _ in BAND_DEFS:
            if col_name.startswith(("v_add", "a_add", "j_add")):
                # add: more on bid = bullish
                asym = grid[:, bid_cols].mean(axis=1) - grid[:, ask_cols].mean(axis=1)
            else:
                # pull: more on ask = bullish
                asym = grid[:, ask_cols].mean(axis=1) - grid[:, bid_cols].mean(axis=1)
            features.append(asym.reshape(-1, 1))
            names.append(f"{col_name}_{band_name}_asym")
            asym_signals.append(asym)

    # 2. Rolling OLS slopes of asymmetries (18 features)
    for i, asym in enumerate(asym_signals):
        slope = rolling_ols_slope(asym, SLOPE_WINDOW)
        slope = np.nan_to_num(slope, nan=0.0)
        features.append(slope.reshape(-1, 1))
        names.append(f"slope_{names[i]}")

    # 3. PV net at spot + rolling stats (5 features)
    pv_spot = grids["pressure_variant"][:, 50] - grids["vacuum_variant"][:, 50]
    features.append(pv_spot.reshape(-1, 1))
    names.append("pv_spot")
    for w in [50, 300]:
        rm, rs = rolling_mean_std(pv_spot, w)
        features.append(rm.reshape(-1, 1))
        features.append(rs.reshape(-1, 1))
        names.extend([f"pv_spot_mean_{w}", f"pv_spot_std_{w}"])

    # 4. Mid-price return rolling stats (9 features)
    returns = np.zeros(n_bins)
    returns[1:] = np.diff(mid_price) / TICK_SIZE
    for w in ROLLING_WINDOWS:
        rm, rs = rolling_mean_std(returns, w)
        features.append(rm.reshape(-1, 1))
        features.append(rs.reshape(-1, 1))
        names.extend([f"ret_mean_{w}", f"ret_std_{w}"])

        # Rolling skewness
        cs = np.cumsum(returns)
        cs2 = np.cumsum(returns ** 2)
        cs3 = np.cumsum(returns ** 3)
        skew = np.zeros(n_bins)
        for i in range(w - 1, n_bins):
            s = cs[i] - (cs[i - w] if i >= w else 0.0)
            s2 = cs2[i] - (cs2[i - w] if i >= w else 0.0)
            s3 = cs3[i] - (cs3[i - w] if i >= w else 0.0)
            m = s / w
            var = s2 / w - m * m
            if var > 1e-12:
                std = np.sqrt(var)
                m3 = s3 / w - 3 * m * s2 / w + 2 * m ** 3
                skew[i] = m3 / (std ** 3)
        features.append(skew.reshape(-1, 1))
        names.append(f"ret_skew_{w}")

    # 5. Spread proxy (best ask - best bid approximation from pressure)
    # Use pressure at k=+1 vs k=-1 as spread proxy
    spread_proxy = grids["pressure_variant"][:, 51] - grids["pressure_variant"][:, 49]
    features.append(spread_proxy.reshape(-1, 1))
    names.append("spread_proxy")
    rm_sp, rs_sp = rolling_mean_std(spread_proxy, 200)
    features.append(rm_sp.reshape(-1, 1))
    features.append(rs_sp.reshape(-1, 1))
    names.extend(["spread_mean_200", "spread_std_200"])

    X = np.hstack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("LightGBM Multi-Feature (GBM_MF)")
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
    X, feat_names = build_features(grids, mid_price, n_bins)
    print(f"  shape={X.shape}, {len(feat_names)} features, time={time.perf_counter()-t1:.1f}s", flush=True)

    # 3. Labels
    print("Computing labels ...", flush=True)
    t2 = time.perf_counter()
    labels = compute_labels(mid_price, n_bins)
    n_pos = (labels == 1).sum()
    n_neg = (labels == -1).sum()
    print(f"  +1={n_pos}, -1={n_neg}, 0={(labels==0).sum()}, time={time.perf_counter()-t2:.1f}s", flush=True)

    # 4. Walk-forward
    print(f"Walk-forward (min_train={MIN_TRAIN_BINS}, retrain={RETRAIN_INTERVAL}) ...", flush=True)
    t3 = time.perf_counter()

    # Binary: predict P(up) vs P(down). Convert labels: +1 -> 1, -1 -> 0
    binary_labels = np.zeros(n_bins, dtype=np.int32)
    binary_labels[labels == 1] = 1
    binary_labels[labels == -1] = 0

    proba_up = np.full(n_bins, 0.5, dtype=np.float64)
    has_prediction = np.zeros(n_bins, dtype=bool)

    model = None
    last_train_bin = -1
    start_bin = max(WARMUP_BINS, MIN_TRAIN_BINS)

    for i in range(start_bin, n_bins):
        if model is None or (i - last_train_bin) >= RETRAIN_INTERVAL:
            train_mask = (labels[:i] != 0)
            if train_mask.sum() < 40:
                continue

            X_train = X[:i][train_mask]
            y_train = binary_labels[:i][train_mask]

            # Split last 20% for early stopping
            split = int(len(X_train) * 0.8)
            if split < 20 or (len(X_train) - split) < 10:
                continue

            dtrain = lgb.Dataset(X_train[:split], label=y_train[:split])
            dval = lgb.Dataset(X_train[split:], label=y_train[split:], reference=dtrain)

            callbacks = [lgb.early_stopping(EARLY_STOPPING, verbose=False)]
            model = lgb.train(
                LGB_PARAMS,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dval],
                callbacks=callbacks,
            )
            last_train_bin = i
            if i == start_bin or (i - start_bin) % 3000 == 0:
                print(f"  retrained at bin {i}, best_iter={model.best_iteration}", flush=True)

        # Predict
        prob = model.predict(X[i:i+1])[0]
        proba_up[i] = prob
        has_prediction[i] = True

    n_predicted = has_prediction.sum()
    print(f"  {n_predicted} predictions, time={time.perf_counter()-t3:.1f}s", flush=True)

    # 5. Convert to directional signals at various probability thresholds
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
                "n_signals": 0,
                "tp_rate": float("nan"),
                "sl_rate": float("nan"),
                "timeout_rate": float("nan"),
                "events_per_hour": 0.0,
                "mean_pnl_ticks": float("nan"),
                "median_time_to_outcome_ms": float("nan"),
                "threshold": prob_thr,
                "cooldown_bins": COOLDOWN_BINS,
            })
            continue

        eval_result = evaluate_tp_sl(
            signals=signals,
            mid_price=mid_price,
            ts_ns=ts_ns,
        )
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
        evhr = r.get("events_per_hour", 0)
        print(f"  prob_thr={r['threshold']:.2f}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t  ev/hr={evhr:.0f}", flush=True)

    # Feature importance
    if model is not None:
        imp = model.feature_importance(importance_type="gain")
        top_idx = np.argsort(imp)[::-1][:10]
        print("\nTop 10 features by gain:")
        for idx in top_idx:
            print(f"  {feat_names[idx]}: {imp[idx]:.1f}")

    params = {
        "model": "LightGBM",
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping": EARLY_STOPPING,
        "n_features": X.shape[1],
        "feature_names": feat_names,
        "min_train_bins": MIN_TRAIN_BINS,
        "retrain_interval": RETRAIN_INTERVAL,
        "cooldown_bins": COOLDOWN_BINS,
    }

    out_path = write_results(
        agent_name="gbm_mf",
        experiment_name="lightgbm_multi_feature",
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
