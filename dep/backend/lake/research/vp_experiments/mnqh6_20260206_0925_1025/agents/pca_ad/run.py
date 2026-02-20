"""Rolling PCA Anomaly Detection (PCA_AD) experiment.

Uses PCA on the spatial pressure-vacuum grid to detect anomalous
microstructure configurations. Anomalies (high reconstruction error
or extreme PC scores) often precede regime transitions.

Approach:
    1. Build rolling PCA on pressure-vacuum difference profile (101 ticks)
    2. Track reconstruction error and Mahalanobis distance on PC scores
    3. Combine with directional bias from top PC loadings
    4. Signal fires when anomaly score is high AND direction is clear

Key insight: "normal" microstructure has a few dominant PCA modes
(bid-ask balance, curvature, etc.). When the grid enters a configuration
that doesn't fit these modes well, something unusual is happening.

Rolling PCA: fit on trailing 600-bin window, project current bin.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
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
    robust_zscore,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_COMPONENTS = 10
PCA_WINDOW = 600
REFIT_INTERVAL = 100
MIN_WARMUP = 700  # PCA_WINDOW + some margin
COOLDOWN_BINS = 30
THRESHOLDS = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]


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


def main() -> None:
    t0 = time.perf_counter()
    print("=" * 70)
    print("Rolling PCA Anomaly Detection (PCA_AD)")
    print("=" * 70)

    # 1. Load
    print("Loading dataset ...", flush=True)
    ds = load_dataset(columns=["pressure_variant", "vacuum_variant", "v_add", "v_pull"])
    P = ds["pressure_variant"]  # (n_bins, 101)
    V = ds["vacuum_variant"]
    v_add = ds["v_add"]
    v_pull = ds["v_pull"]
    mid_price = ds["mid_price"]
    ts_ns = ds["ts_ns"]
    n_bins = ds["n_bins"]
    print(f"  n_bins={n_bins}, load={time.perf_counter()-t0:.1f}s", flush=True)

    # PV difference is the primary spatial field
    pv_diff = P - V  # (n_bins, 101)

    # Also build add-pull asymmetry field for directional bias
    add_pull_diff = v_add - v_pull  # (n_bins, 101)

    # 2. Rolling PCA
    print(f"Rolling PCA (n_components={N_COMPONENTS}, window={PCA_WINDOW}, "
          f"refit_every={REFIT_INTERVAL}) ...", flush=True)
    t1 = time.perf_counter()

    recon_error = np.zeros(n_bins, dtype=np.float64)
    mahal_dist = np.zeros(n_bins, dtype=np.float64)
    pc1_score = np.zeros(n_bins, dtype=np.float64)
    pc1_direction = np.zeros(n_bins, dtype=np.float64)  # sign of PC1 loading asymmetry

    pca_model = None
    pca_scaler = None
    last_fit = -1

    for i in range(MIN_WARMUP, n_bins):
        if pca_model is None or (i - last_fit) >= REFIT_INTERVAL:
            # Fit PCA on trailing window
            start = max(0, i - PCA_WINDOW)
            window_data = pv_diff[start:i]

            pca_scaler = StandardScaler()
            window_scaled = pca_scaler.fit_transform(window_data)

            n_comp = min(N_COMPONENTS, window_data.shape[0], window_data.shape[1])
            pca_model = PCA(n_components=n_comp)
            pca_model.fit(window_scaled)

            # Direction from PC1 loadings: sum of loadings on bid vs ask
            pc1_loadings = pca_model.components_[0]
            bid_loading = pc1_loadings[:50].sum()  # k < 0
            ask_loading = pc1_loadings[51:].sum()   # k > 0
            pc1_dir_sign = 1.0 if bid_loading > ask_loading else -1.0

            last_fit = i

        # Project current bin
        x_scaled = pca_scaler.transform(pv_diff[i:i+1])
        scores = pca_model.transform(x_scaled)[0]
        x_recon = pca_model.inverse_transform(scores.reshape(1, -1))

        # Reconstruction error (L2 norm)
        err = np.sqrt(np.sum((x_scaled[0] - x_recon[0]) ** 2))
        recon_error[i] = err

        # Mahalanobis-like distance on PC scores
        # Use explained variance as scaling
        variances = pca_model.explained_variance_
        valid_var = variances > 1e-10
        if valid_var.sum() > 0:
            scaled_scores = scores[valid_var] ** 2 / variances[valid_var]
            mahal_dist[i] = np.sqrt(scaled_scores.sum())

        # PC1 score (directional component)
        pc1_score[i] = scores[0] * pc1_dir_sign

        # Direction from add-pull field
        ap = add_pull_diff[i]
        bid_ap = ap[:50].mean()
        ask_ap = ap[51:].mean()
        pc1_direction[i] = bid_ap - ask_ap  # positive = bullish

    print(f"  PCA done in {time.perf_counter()-t1:.1f}s", flush=True)

    # 3. Build anomaly-gated directional signal
    print("Building anomaly-gated signal ...", flush=True)

    # Z-score the reconstruction error (anomaly detector)
    z_recon = robust_zscore(recon_error, window=300)
    z_mahal = robust_zscore(mahal_dist, window=300)

    # Combined anomaly score
    anomaly_score = 0.5 * np.maximum(z_recon, 0) + 0.5 * np.maximum(z_mahal, 0)

    # Directional bias: combine PC1 score with add-pull asymmetry
    # Normalize both to similar scale
    z_pc1 = robust_zscore(pc1_score, window=300)
    z_dir = robust_zscore(pc1_direction, window=300)
    direction_signal = 0.6 * np.tanh(z_pc1 / 3.0) + 0.4 * np.tanh(z_dir / 3.0)

    # Final signal: direction * anomaly gate
    # Only fire when anomaly is elevated AND direction is clear
    signal = direction_signal * anomaly_score

    # Signal stats
    active = signal[MIN_WARMUP:]
    nonzero = active[active != 0]
    if len(nonzero) > 0:
        pcts = np.percentile(nonzero, [5, 25, 50, 75, 95])
        print(f"  signal stats (non-zero): p5={pcts[0]:.4f} p25={pcts[1]:.4f} "
              f"p50={pcts[2]:.4f} p75={pcts[3]:.4f} p95={pcts[4]:.4f}")
        print(f"  n_nonzero={len(nonzero)}/{len(active)} ({100*len(nonzero)/len(active):.1f}%)")

    # 4. Threshold sweep
    print(f"Sweeping {len(THRESHOLDS)} thresholds (cooldown={COOLDOWN_BINS}) ...", flush=True)

    all_results = []
    for thr in THRESHOLDS:
        signals = []
        last_signal_bin = -COOLDOWN_BINS
        prev_state = "flat"

        for i in range(MIN_WARMUP, n_bins):
            if signal[i] >= thr:
                cur_state = "up"
            elif signal[i] <= -thr:
                cur_state = "down"
            else:
                cur_state = "flat"

            if cur_state != "flat" and cur_state != prev_state:
                if i - last_signal_bin >= COOLDOWN_BINS:
                    signals.append({"bin_idx": i, "direction": cur_state})
                    last_signal_bin = i
            prev_state = cur_state

        if not signals:
            all_results.append({
                "n_signals": 0, "tp_rate": float("nan"), "sl_rate": float("nan"),
                "timeout_rate": float("nan"), "events_per_hour": 0.0,
                "mean_pnl_ticks": float("nan"), "median_time_to_outcome_ms": float("nan"),
                "threshold": thr, "cooldown_bins": COOLDOWN_BINS,
            })
            continue

        eval_result = evaluate_tp_sl(signals=signals, mid_price=mid_price, ts_ns=ts_ns)
        eval_result.pop("outcomes", None)
        eval_result["threshold"] = thr
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
        print(f"  thr={r['threshold']:.1f}  n={n:>4d}  TP={tp_str}  PnL={pnl_str}t", flush=True)

    params = {
        "model": "Rolling PCA + anomaly gating",
        "n_components": N_COMPONENTS,
        "pca_window": PCA_WINDOW,
        "refit_interval": REFIT_INTERVAL,
        "anomaly_blend": "0.5*max(z_recon,0) + 0.5*max(z_mahal,0)",
        "direction_blend": "0.6*tanh(z_pc1/3) + 0.4*tanh(z_dir/3)",
        "signal": "direction * anomaly_score",
        "cooldown_bins": COOLDOWN_BINS,
    }

    out_path = write_results(
        agent_name="pca_ad",
        experiment_name="rolling_pca_anomaly",
        params=params,
        results_by_threshold=all_results,
    )

    valid = [r for r in all_results if r["n_signals"] >= 5]
    if valid:
        best = max(valid, key=lambda r: r["tp_rate"])
        print(f"\nBest: thr={best['threshold']:.1f}  TP={best['tp_rate']:.1%}  "
              f"n={best['n_signals']}  PnL={best['mean_pnl_ticks']:+.2f}t")

    print(f"Results written to {out_path}")
    print(f"Total runtime: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
