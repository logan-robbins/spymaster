"""Entropy Regime Detector (ERD) experiment.

Detects entropy spikes in the spectrum state field as precursors to
regime transitions. Shannon entropy of the 3-state distribution across
ticks measures disorder; asymmetry between above/below spot provides
directional bias.

Two signal variants:
  A) signal = score_direction * max(0, z_H - 0.5)
  B) signal = entropy_asym * max(0, z_H - 0.5)

Best variant selected by TP rate (min 5 signals).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Harness import
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_harness import (
    WARMUP_BINS,
    N_TICKS,
    K_MAX,
    K_MIN,
    load_dataset,
    detect_signals,
    evaluate_tp_sl,
    sweep_thresholds,
    write_results,
    robust_zscore,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AGENT_NAME = "erd"
EXPERIMENT_NAME = "entropy_regime_detector"
ZSCORE_WINDOW = 100
COOLDOWN_BINS = 40
SPIKE_FLOOR = 0.5  # z_H must exceed this before signal activates
LOG2_3 = np.log2(3.0)  # max entropy for 3 states = 1.585 bits

# Threshold grid for sweep
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0, 1.5, 2.0]


def shannon_entropy_3state(state_arr: np.ndarray) -> float:
    """Compute Shannon entropy (bits) of a 1D array with values in {-1, 0, 1}.

    H = -sum_{i} p_i * log2(p_i)  for p_i > 0
    where p_i = count_i / n for each of the 3 states.

    Args:
        state_arr: 1D array of int8 state codes {-1, 0, 1}.

    Returns:
        Entropy in bits. Range [0, log2(3)] = [0, 1.585].
    """
    n = len(state_arr)
    if n == 0:
        return 0.0
    n_pressure = np.sum(state_arr == 1)
    n_vacuum = np.sum(state_arr == -1)
    n_neutral = np.sum(state_arr == 0)

    h = 0.0
    for count in (n_pressure, n_vacuum, n_neutral):
        if count > 0:
            p = count / n
            h -= p * np.log2(p + 1e-12)
    return float(h)


def compute_entropy_arrays(
    state_code: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full, above-spot, and below-spot entropy per bin.

    Args:
        state_code: (n_bins, 101) int array. Col 50 = spot (k=0).
            Cols 0..49 = below spot (k=-50..-1).
            Cols 51..100 = above spot (k=+1..+50).

    Returns:
        (H_full, H_above, H_below) each shape (n_bins,).
    """
    n_bins = state_code.shape[0]
    h_full = np.zeros(n_bins, dtype=np.float64)
    h_above = np.zeros(n_bins, dtype=np.float64)
    h_below = np.zeros(n_bins, dtype=np.float64)

    # Vectorized: count states across tick axis for each region
    # Full: all 101 ticks
    for state_val, label in [(1, "p"), (-1, "v"), (0, "n")]:
        mask_full = (state_code == state_val)
        mask_above = (state_code[:, 51:101] == state_val)
        mask_below = (state_code[:, 0:50] == state_val)

        count_full = mask_full.sum(axis=1).astype(np.float64)
        count_above = mask_above.sum(axis=1).astype(np.float64)
        count_below = mask_below.sum(axis=1).astype(np.float64)

        # Full: n=101
        p_full = count_full / 101.0
        valid = p_full > 0
        h_full[valid] -= p_full[valid] * np.log2(p_full[valid] + 1e-12)

        # Above: n=50 (cols 51..100)
        p_above = count_above / 50.0
        valid = p_above > 0
        h_above[valid] -= p_above[valid] * np.log2(p_above[valid] + 1e-12)

        # Below: n=50 (cols 0..49)
        p_below = count_below / 50.0
        valid = p_below > 0
        h_below[valid] -= p_below[valid] * np.log2(p_below[valid] + 1e-12)

    return h_full, h_above, h_below


def main() -> None:
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("[ERD] Loading dataset (spectrum_state_code, spectrum_score)...")
    ds = load_dataset(columns=["spectrum_state_code", "spectrum_score"])
    n_bins = ds["n_bins"]
    mid_price = ds["mid_price"]
    ts_ns = ds["ts_ns"]
    state_code = ds["spectrum_state_code"]  # (n_bins, 101) float64 from harness
    score = ds["spectrum_score"]            # (n_bins, 101) float64
    print(f"[ERD] Loaded: {n_bins} bins, {N_TICKS} ticks per bin")

    # Cast state_code to int8 for counting (harness returns float64)
    state_int = state_code.astype(np.int8)

    # ------------------------------------------------------------------
    # 2. Compute Shannon entropy per bin
    # ------------------------------------------------------------------
    print("[ERD] Computing Shannon entropy (full, above, below)...")
    h_full, h_above, h_below = compute_entropy_arrays(state_int)

    # Entropy asymmetry: positive = more disorder above spot
    entropy_asym = h_above - h_below

    print(f"[ERD] H_full  stats: mean={h_full.mean():.4f}  std={h_full.std():.4f}  "
          f"p5={np.percentile(h_full, 5):.4f}  p25={np.percentile(h_full, 25):.4f}  "
          f"p50={np.percentile(h_full, 50):.4f}  p75={np.percentile(h_full, 75):.4f}  "
          f"p95={np.percentile(h_full, 95):.4f}  max={h_full.max():.4f}")
    print(f"[ERD] H_above stats: mean={h_above.mean():.4f}  std={h_above.std():.4f}  "
          f"p50={np.percentile(h_above, 50):.4f}")
    print(f"[ERD] H_below stats: mean={h_below.mean():.4f}  std={h_below.std():.4f}  "
          f"p50={np.percentile(h_below, 50):.4f}")
    print(f"[ERD] entropy_asym stats: mean={entropy_asym.mean():.4f}  "
          f"std={entropy_asym.std():.4f}  "
          f"p5={np.percentile(entropy_asym, 5):.4f}  "
          f"p95={np.percentile(entropy_asym, 95):.4f}")

    # ------------------------------------------------------------------
    # 3. Rolling z-score of H_full
    # ------------------------------------------------------------------
    print(f"[ERD] Computing robust z-score of H_full (window={ZSCORE_WINDOW})...")
    z_h = robust_zscore(h_full, window=ZSCORE_WINDOW)
    print(f"[ERD] z_H stats: mean={z_h.mean():.4f}  std={z_h.std():.4f}  "
          f"p5={np.percentile(z_h, 5):.4f}  p95={np.percentile(z_h, 95):.4f}  "
          f"max={z_h.max():.4f}")

    # ------------------------------------------------------------------
    # 4. Score direction: mean_score_below - mean_score_above
    # ------------------------------------------------------------------
    print("[ERD] Computing score direction...")
    mean_score_above = score[:, 51:101].mean(axis=1)  # cols 51..100 (k=+1..+50)
    mean_score_below = score[:, 0:50].mean(axis=1)    # cols 0..49  (k=-50..-1)
    score_direction = mean_score_below - mean_score_above

    print(f"[ERD] score_direction stats: mean={score_direction.mean():.4f}  "
          f"std={score_direction.std():.4f}  "
          f"p5={np.percentile(score_direction, 5):.4f}  "
          f"p95={np.percentile(score_direction, 95):.4f}")

    # ------------------------------------------------------------------
    # 5. Build signals
    # ------------------------------------------------------------------
    # Spike gate: max(0, z_H - SPIKE_FLOOR). Zero when entropy is normal,
    # positive when entropy spikes above the floor.
    spike_gate = np.maximum(0.0, z_h - SPIKE_FLOOR)

    # Variant A: score_direction * spike_gate
    signal_a = score_direction * spike_gate

    # Variant B: entropy_asym * spike_gate
    signal_b = entropy_asym * spike_gate

    nonzero_a = np.count_nonzero(signal_a[WARMUP_BINS:])
    nonzero_b = np.count_nonzero(signal_b[WARMUP_BINS:])
    print(f"[ERD] Signal A (score_dir * spike): {nonzero_a} non-zero bins after warmup")
    print(f"[ERD] Signal B (ent_asym * spike):  {nonzero_b} non-zero bins after warmup")

    # ------------------------------------------------------------------
    # 6. Sweep thresholds for both variants
    # ------------------------------------------------------------------
    print(f"[ERD] Sweeping {len(THRESHOLDS)} thresholds, cooldown={COOLDOWN_BINS}...")

    results_a = sweep_thresholds(
        signal=signal_a,
        thresholds=THRESHOLDS,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
        warmup_bins=WARMUP_BINS,
    )
    results_b = sweep_thresholds(
        signal=signal_b,
        thresholds=THRESHOLDS,
        cooldown_bins=COOLDOWN_BINS,
        mid_price=mid_price,
        ts_ns=ts_ns,
        warmup_bins=WARMUP_BINS,
    )

    # ------------------------------------------------------------------
    # 7. Print results and pick best variant
    # ------------------------------------------------------------------
    def _print_variant(label: str, results: list) -> None:
        print(f"\n[ERD] --- Variant {label} ---")
        print(f"  {'Thr':>6s}  {'N':>4s}  {'TP%':>6s}  {'SL%':>6s}  {'TO%':>6s}  {'PnL':>7s}  {'Ev/hr':>6s}")
        for r in results:
            tp = r['tp_rate'] * 100 if not np.isnan(r['tp_rate']) else 0
            sl = r['sl_rate'] * 100 if not np.isnan(r['sl_rate']) else 0
            to = r['timeout_rate'] * 100 if not np.isnan(r['timeout_rate']) else 0
            pnl = r['mean_pnl_ticks'] if not np.isnan(r['mean_pnl_ticks']) else 0
            evhr = r['events_per_hour']
            print(f"  {r['threshold']:6.3f}  {r['n_signals']:4d}  {tp:5.1f}%  {sl:5.1f}%  {to:5.1f}%  {pnl:+7.2f}  {evhr:6.1f}")

    _print_variant("A (score_dir * spike)", results_a)
    _print_variant("B (ent_asym * spike)", results_b)

    # Select best: highest TP rate with >= 5 signals
    def _best(results: list) -> dict | None:
        valid = [r for r in results if r["n_signals"] >= 5]
        if not valid:
            return None
        return max(valid, key=lambda r: r["tp_rate"])

    best_a = _best(results_a)
    best_b = _best(results_b)

    # Determine winner
    if best_a is None and best_b is None:
        print("\n[ERD] WARNING: Neither variant produced >= 5 signals at any threshold")
        winner_label = "A"
        winner_results = results_a
    elif best_a is None:
        winner_label = "B"
        winner_results = results_b
    elif best_b is None:
        winner_label = "A"
        winner_results = results_a
    elif best_b["tp_rate"] > best_a["tp_rate"]:
        winner_label = "B"
        winner_results = results_b
    else:
        winner_label = "A"
        winner_results = results_a

    best = _best(winner_results)
    print(f"\n[ERD] Winner: Variant {winner_label}")
    if best:
        print(f"[ERD] Best threshold: {best['threshold']:.3f}  "
              f"TP={best['tp_rate']*100:.1f}%  "
              f"N={best['n_signals']}  "
              f"PnL={best['mean_pnl_ticks']:+.2f}t  "
              f"Ev/hr={best['events_per_hour']:.1f}")

    # ------------------------------------------------------------------
    # 8. Write results
    # ------------------------------------------------------------------
    params = {
        "zscore_window": ZSCORE_WINDOW,
        "cooldown_bins": COOLDOWN_BINS,
        "spike_floor": SPIKE_FLOOR,
        "thresholds": THRESHOLDS,
        "winner_variant": winner_label,
        "variant_a_desc": "score_direction * max(0, z_H - spike_floor)",
        "variant_b_desc": "entropy_asym * max(0, z_H - spike_floor)",
        "entropy_formula": "H = -sum(p_i * log2(p_i + 1e-12)) for 3 states {-1, 0, 1}",
        "max_entropy_bits": LOG2_3,
    }

    # Also include the losing variant results under a separate key
    all_results = winner_results
    out_path = write_results(
        agent_name=AGENT_NAME,
        experiment_name=EXPERIMENT_NAME,
        params=params,
        results_by_threshold=all_results,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n[ERD] Results written to {out_path}")
    print(f"[ERD] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
