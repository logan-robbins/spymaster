"""Move-Size Signal Decomposition (MSD) experiment.

Produces three JSON outputs:
  1. forensic.json    — Per-bin signal decomposition for 09:27-09:32 window
  2. stratified.json  — TP/SL evaluation stratified by forward move-size tiers
  3. spatial_vacuum.json + results.json — Spatial vacuum asymmetry sweep

Replicates PFP, ADS, ERD signal computations exactly, then decomposes them
into all intermediate sub-components for forensic attribution.

Composite weights match production experiment-engine.ts:
    composite = 0.40 * pfp + 0.35 * ads + 0.25 * erd
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Harness import — matches existing agent pattern
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_harness import (
    WARMUP_BINS,
    N_TICKS,
    K_MIN,
    K_MAX,
    TICK_SIZE,
    TP_TICKS,
    SL_TICKS,
    MAX_HOLD_BINS,
    load_dataset,
    detect_signals,
    evaluate_tp_sl,
    rolling_ols_slope,
    robust_zscore,
    write_results,
)

# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------
AGENT_NAME = "msd"
EXPERIMENT_NAME = "move_size_signal_decomposition"

# ---------------------------------------------------------------------------
# PFP constants (exact copy from agents/pfp/run.py)
# ---------------------------------------------------------------------------
PFP_INNER_BID_COLS: list[int] = [47, 48, 49]          # k = -3, -2, -1
PFP_INNER_ASK_COLS: list[int] = [51, 52, 53]          # k = +1, +2, +3
PFP_OUTER_BID_COLS: list[int] = list(range(38, 46))   # k = -12 .. -5
PFP_OUTER_ASK_COLS: list[int] = list(range(55, 63))   # k = +5 .. +12
PFP_LAG_BINS: int = 5
PFP_EMA_ALPHA: float = 0.1
PFP_EPS: float = 1e-12
PFP_ADD_WEIGHT: float = 0.6
PFP_PULL_WEIGHT: float = 0.4

# ---------------------------------------------------------------------------
# ADS constants (exact copy from agents/ads/run.py)
# ---------------------------------------------------------------------------
ADS_BANDS: list[dict[str, object]] = [
    {
        "name": "inner",
        "bid_cols": list(range(47, 50)),     # k=-3..-1
        "ask_cols": list(range(51, 54)),      # k=+1..+3
        "width": 3,
    },
    {
        "name": "mid",
        "bid_cols": list(range(39, 47)),      # k=-11..-4
        "ask_cols": list(range(54, 62)),       # k=+4..+11
        "width": 8,
    },
    {
        "name": "outer",
        "bid_cols": list(range(27, 39)),      # k=-23..-12
        "ask_cols": list(range(62, 74)),       # k=+12..+23
        "width": 12,
    },
]
ADS_SLOPE_WINDOWS: list[int] = [10, 25, 50]
ADS_ZSCORE_WINDOW: int = 200
ADS_BLEND_WEIGHTS: list[float] = [0.40, 0.35, 0.25]
ADS_BLEND_SCALE: float = 3.0

# ---------------------------------------------------------------------------
# ERD constants (exact copy from agents/erd/run.py)
# ---------------------------------------------------------------------------
ERD_ZSCORE_WINDOW: int = 100
ERD_SPIKE_FLOOR: float = 0.5

# ---------------------------------------------------------------------------
# Composite weights (from experiment-engine.ts)
# ---------------------------------------------------------------------------
COMPOSITE_PFP_W: float = 0.40
COMPOSITE_ADS_W: float = 0.35
COMPOSITE_ERD_W: float = 0.25

# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------
COOLDOWN_BINS: int = 30
FORWARD_WINDOW_BINS: int = 600  # 60s at 100ms bins

# Move-size tier boundaries (in $0.25 ticks)
TIER_SMALL = (4, 8)
TIER_MEDIUM = (8, 16)
TIER_LARGE = (16, 32)
TIER_EXTREME = (32, None)  # 32+


# ===================================================================
# JSON serialization helper
# ===================================================================

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python builtins for json.dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _write_json(path: Path, data: Any) -> None:
    """Write data to JSON with numpy-safe serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_python(data), indent=2))
    print(f"  Wrote {path}", flush=True)


# ===================================================================
# PFP: exact replication from agents/pfp/run.py
# ===================================================================

def _ema_1d(arr: np.ndarray, alpha: float) -> np.ndarray:
    """EMA over 1D array. EMA[0]=arr[0], EMA[t]=alpha*arr[t]+(1-alpha)*EMA[t-1]."""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    out[0] = arr[0]
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * arr[i] + one_minus_alpha * out[i - 1]
    return out


def _compute_zone_intensity(
    v_add: np.ndarray,
    v_fill: np.ndarray,
    zone_cols: list[int],
) -> np.ndarray:
    """I_zone[t] = mean(v_add[t, zone_cols] + v_fill[t, zone_cols])."""
    return (v_add[:, zone_cols] + v_fill[:, zone_cols]).mean(axis=1)


def _compute_lead_metric(
    inner: np.ndarray,
    outer: np.ndarray,
    lag: int,
    alpha: float,
) -> np.ndarray:
    """EMA-based lead-lag ratio: ema(inner*outer_lagged) / (ema(inner*outer) + eps)."""
    n = len(inner)
    prod_unlagged = inner * outer
    outer_lagged = np.zeros(n, dtype=np.float64)
    outer_lagged[lag:] = outer[:n - lag]
    prod_lagged = inner * outer_lagged
    ema_lagged = _ema_1d(prod_lagged, alpha)
    ema_unlagged = _ema_1d(prod_unlagged, alpha)
    return ema_lagged / (ema_unlagged + PFP_EPS)


def compute_pfp_decomposed(
    v_add: np.ndarray,
    v_fill: np.ndarray,
    v_pull: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute PFP signal with all intermediate arrays exposed.

    Returns dict with keys:
        i_inner_bid, i_inner_ask, i_outer_bid, i_outer_ask,
        lead_bid, lead_ask, add_signal,
        pull_lead_bid, pull_lead_ask, pull_signal,
        final
    """
    # --- Add/fill channel ---
    i_inner_bid = _compute_zone_intensity(v_add, v_fill, PFP_INNER_BID_COLS)
    i_inner_ask = _compute_zone_intensity(v_add, v_fill, PFP_INNER_ASK_COLS)
    i_outer_bid = _compute_zone_intensity(v_add, v_fill, PFP_OUTER_BID_COLS)
    i_outer_ask = _compute_zone_intensity(v_add, v_fill, PFP_OUTER_ASK_COLS)

    lead_bid = _compute_lead_metric(i_inner_bid, i_outer_bid, PFP_LAG_BINS, PFP_EMA_ALPHA)
    lead_ask = _compute_lead_metric(i_inner_ask, i_outer_ask, PFP_LAG_BINS, PFP_EMA_ALPHA)
    add_signal = lead_bid - lead_ask

    # --- Pull channel ---
    p_inner_bid = v_pull[:, PFP_INNER_BID_COLS].mean(axis=1)
    p_inner_ask = v_pull[:, PFP_INNER_ASK_COLS].mean(axis=1)
    p_outer_bid = v_pull[:, PFP_OUTER_BID_COLS].mean(axis=1)
    p_outer_ask = v_pull[:, PFP_OUTER_ASK_COLS].mean(axis=1)

    pull_lead_bid = _compute_lead_metric(p_inner_bid, p_outer_bid, PFP_LAG_BINS, PFP_EMA_ALPHA)
    pull_lead_ask = _compute_lead_metric(p_inner_ask, p_outer_ask, PFP_LAG_BINS, PFP_EMA_ALPHA)
    pull_signal = pull_lead_ask - pull_lead_bid

    final = PFP_ADD_WEIGHT * add_signal + PFP_PULL_WEIGHT * pull_signal

    return {
        "i_inner_bid": i_inner_bid,
        "i_inner_ask": i_inner_ask,
        "i_outer_bid": i_outer_bid,
        "i_outer_ask": i_outer_ask,
        "lead_bid": lead_bid,
        "lead_ask": lead_ask,
        "add_signal": add_signal,
        "pull_lead_bid": pull_lead_bid,
        "pull_lead_ask": pull_lead_ask,
        "pull_signal": pull_signal,
        "final": final,
    }


# ===================================================================
# ADS: exact replication from agents/ads/run.py
# ===================================================================

def _compute_combined_asymmetry(
    v_add: np.ndarray,
    v_pull: np.ndarray,
) -> np.ndarray:
    """Bandwidth-weighted combined asymmetry across inner/mid/outer bands."""
    n_bins = v_add.shape[0]
    combined = np.zeros(n_bins, dtype=np.float64)
    total_weight = 0.0

    for band in ADS_BANDS:
        bid_cols = band["bid_cols"]
        ask_cols = band["ask_cols"]
        bw = band["width"]
        w = 1.0 / np.sqrt(bw)
        total_weight += w
        add_asym = v_add[:, bid_cols].mean(axis=1) - v_add[:, ask_cols].mean(axis=1)
        pull_asym = v_pull[:, ask_cols].mean(axis=1) - v_pull[:, bid_cols].mean(axis=1)
        combined += w * (add_asym + pull_asym)

    combined /= total_weight
    return combined


def compute_ads_decomposed(
    v_add: np.ndarray,
    v_pull: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ADS signal with all intermediate arrays exposed.

    Returns dict with keys:
        combined_asym, slope_10, slope_25, slope_50,
        z_10, z_25, z_50, final
    """
    combined_asym = _compute_combined_asymmetry(v_add, v_pull)

    slopes: dict[str, np.ndarray] = {}
    zscores: dict[str, np.ndarray] = {}
    n = len(combined_asym)
    signal = np.zeros(n, dtype=np.float64)

    for i, win in enumerate(ADS_SLOPE_WINDOWS):
        slope = rolling_ols_slope(combined_asym, win)
        z = robust_zscore(slope, ADS_ZSCORE_WINDOW)
        slopes[f"slope_{win}"] = slope
        zscores[f"z_{win}"] = z
        signal += ADS_BLEND_WEIGHTS[i] * np.tanh(z / ADS_BLEND_SCALE)

    return {
        "combined_asym": combined_asym,
        "slope_10": slopes["slope_10"],
        "slope_25": slopes["slope_25"],
        "slope_50": slopes["slope_50"],
        "z_10": zscores["z_10"],
        "z_25": zscores["z_25"],
        "z_50": zscores["z_50"],
        "final": signal,
    }


# ===================================================================
# ERD: exact replication from agents/erd/run.py
# ===================================================================

def _compute_entropy_arrays(
    state_int: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute full, above-spot, and below-spot Shannon entropy per bin.

    state_int: (n_bins, 101) int8 with values in {-1, 0, 1}.
    Returns (h_full, h_above, h_below) each shape (n_bins,).
    """
    n_bins = state_int.shape[0]
    h_full = np.zeros(n_bins, dtype=np.float64)
    h_above = np.zeros(n_bins, dtype=np.float64)
    h_below = np.zeros(n_bins, dtype=np.float64)

    for state_val in (1, -1, 0):
        mask_full = (state_int == state_val)
        mask_above = (state_int[:, 51:101] == state_val)
        mask_below = (state_int[:, 0:50] == state_val)

        count_full = mask_full.sum(axis=1).astype(np.float64)
        count_above = mask_above.sum(axis=1).astype(np.float64)
        count_below = mask_below.sum(axis=1).astype(np.float64)

        p_full = count_full / 101.0
        valid = p_full > 0
        h_full[valid] -= p_full[valid] * np.log2(p_full[valid] + 1e-12)

        p_above = count_above / 50.0
        valid = p_above > 0
        h_above[valid] -= p_above[valid] * np.log2(p_above[valid] + 1e-12)

        p_below = count_below / 50.0
        valid = p_below > 0
        h_below[valid] -= p_below[valid] * np.log2(p_below[valid] + 1e-12)

    return h_full, h_above, h_below


def compute_erd_decomposed(
    state_int: np.ndarray,
    spectrum_score: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute ERD signal with all intermediate arrays exposed.

    Returns dict with keys:
        h_full, h_above, h_below, entropy_asym,
        z_h, spike_gate, score_direction,
        signal_a, signal_b (signal_b is the production variant)
    """
    h_full, h_above, h_below = _compute_entropy_arrays(state_int)
    entropy_asym = h_above - h_below
    z_h = robust_zscore(h_full, window=ERD_ZSCORE_WINDOW)
    spike_gate = np.maximum(0.0, z_h - ERD_SPIKE_FLOOR)

    mean_score_above = spectrum_score[:, 51:101].mean(axis=1)
    mean_score_below = spectrum_score[:, 0:50].mean(axis=1)
    score_direction = mean_score_below - mean_score_above

    signal_a = score_direction * spike_gate
    signal_b = entropy_asym * spike_gate

    return {
        "h_full": h_full,
        "h_above": h_above,
        "h_below": h_below,
        "entropy_asym": entropy_asym,
        "z_h": z_h,
        "spike_gate": spike_gate,
        "score_direction": score_direction,
        "signal_a": signal_a,
        "signal_b": signal_b,
    }


# ===================================================================
# Spatial vacuum signal (new)
# ===================================================================

def compute_spatial_vacuum(
    vacuum_variant: np.ndarray,
    pressure_variant: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute spatial vacuum asymmetry signals.

    vacuum_variant: (n_bins, 101) vacuum field.
        cols 0:50  = below spot (k=-50..-1)
        cols 51:101 = above spot (k=+1..+50)
        col 50 = spot (k=0), excluded from asymmetry.

    Args:
        vacuum_variant: (n_bins, 101) vacuum field.
        pressure_variant: (n_bins, 101) pressure field.

    Returns dict with keys:
        vac_below_sum, vac_above_sum,
        pres_below_sum, pres_above_sum,
        signal_a (sum asymmetry), signal_c (distance-weighted asymmetry)
    """
    vac_below = vacuum_variant[:, 0:50]    # k=-50..-1
    vac_above = vacuum_variant[:, 51:101]  # k=+1..+50

    vac_below_sum = vac_below.sum(axis=1)
    vac_above_sum = vac_above.sum(axis=1)

    # Variant A: simple sum asymmetry. Positive = more vacuum above = bullish
    signal_a = vac_above_sum - vac_below_sum

    # Variant C: 1/|k| distance-weighted asymmetry. Near-spot vacuum weighted more.
    # Below: k=-1 is col 49, k=-2 is col 48, ..., k=-50 is col 0
    # |k| for col j (j in 0..49) = 50 - j
    # Above: k=+1 is col 51, k=+2 is col 52, ..., k=+50 is col 100
    # |k| for col j (j in 51..100) = j - 50
    weights_below = 1.0 / np.arange(50, 0, -1, dtype=np.float64)  # 1/50, 1/49, ..., 1/1
    weights_above = 1.0 / np.arange(1, 51, dtype=np.float64)       # 1/1, 1/2, ..., 1/50

    weighted_below = (vac_below * weights_below).sum(axis=1)
    weighted_above = (vac_above * weights_above).sum(axis=1)
    signal_c = weighted_above - weighted_below

    # Pressure sums for forensic context
    pres_below_sum = pressure_variant[:, 0:50].sum(axis=1)
    pres_above_sum = pressure_variant[:, 51:101].sum(axis=1)

    return {
        "vac_below_sum": vac_below_sum,
        "vac_above_sum": vac_above_sum,
        "pres_below_sum": pres_below_sum,
        "pres_above_sum": pres_above_sum,
        "signal_a": signal_a,
        "signal_c": signal_c,
    }


# ===================================================================
# Forward excursion computation
# ===================================================================

def compute_forward_excursion(
    mid_price: np.ndarray,
    idx: int,
    window: int,
) -> dict[str, float]:
    """Compute max up/down ticks over a forward window from a given bin.

    Args:
        mid_price: Full mid_price array.
        idx: Starting bin index.
        window: Number of forward bins to scan.

    Returns:
        dict with max_up_ticks, max_down_ticks (in $0.25 ticks).
    """
    entry = mid_price[idx]
    if entry <= 0:
        return {"max_up_ticks": 0.0, "max_down_ticks": 0.0}

    end = min(idx + window + 1, len(mid_price))
    forward = mid_price[idx + 1:end]
    if len(forward) == 0:
        return {"max_up_ticks": 0.0, "max_down_ticks": 0.0}

    # Filter out zero/invalid prices
    valid = forward[forward > 0]
    if len(valid) == 0:
        return {"max_up_ticks": 0.0, "max_down_ticks": 0.0}

    deltas = valid - entry
    max_up = float(np.max(deltas)) / TICK_SIZE
    max_down = float(-np.min(deltas)) / TICK_SIZE  # positive = price fell

    return {
        "max_up_ticks": max(0.0, max_up),
        "max_down_ticks": max(0.0, max_down),
    }


def compute_max_favorable_ticks(
    mid_price: np.ndarray,
    idx: int,
    direction: str,
    window: int,
) -> float:
    """Max favorable excursion in direction of the signal over forward window.

    Args:
        mid_price: Full mid_price array.
        idx: Entry bin index.
        direction: "up" or "down".
        window: Number of forward bins to scan.

    Returns:
        Max favorable ticks (always >= 0).
    """
    entry = mid_price[idx]
    if entry <= 0:
        return 0.0

    end = min(idx + window + 1, len(mid_price))
    forward = mid_price[idx + 1:end]
    valid = forward[forward > 0]
    if len(valid) == 0:
        return 0.0

    if direction == "up":
        return max(0.0, float(np.max(valid) - entry) / TICK_SIZE)
    else:
        return max(0.0, float(entry - np.min(valid)) / TICK_SIZE)


# ===================================================================
# Move-size tier classification
# ===================================================================

def classify_tier(max_favorable_ticks: float) -> str:
    """Classify a trade's max favorable excursion into a move-size tier."""
    t = abs(max_favorable_ticks)
    if t >= 32:
        return "extreme"
    if t >= 16:
        return "large"
    if t >= 8:
        return "medium"
    if t >= 4:
        return "small"
    return "micro"


# ===================================================================
# Part 1: Forensic attribution
# ===================================================================

def build_forensic(
    *,
    ts_ns: np.ndarray,
    mid_price: np.ndarray,
    pfp: dict[str, np.ndarray],
    ads: dict[str, np.ndarray],
    erd: dict[str, np.ndarray],
    spatial: dict[str, np.ndarray],
    composite: np.ndarray,
    n_bins: int,
) -> dict[str, Any]:
    """Build Part 1 forensic attribution for the 09:27-09:32 window.

    Returns:
        dict with 'window_start', 'window_end', 'bins' list, 'critical_bin' analysis.
    """
    # Convert ts_ns to pandas Timestamps for window lookup
    ts_series = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("US/Eastern")

    window_start = pd.Timestamp("2026-02-06 09:27:00", tz="US/Eastern")
    window_end = pd.Timestamp("2026-02-06 09:32:00", tz="US/Eastern")

    mask = (ts_series >= window_start) & (ts_series < window_end)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return {"error": "No bins found in 09:27-09:32 window"}

    print(f"  Forensic window: {len(indices)} bins, "
          f"idx {indices[0]}..{indices[-1]}", flush=True)

    bins_list: list[dict[str, Any]] = []
    for idx in indices:
        excursion = compute_forward_excursion(mid_price, idx, FORWARD_WINDOW_BINS)

        record: dict[str, Any] = {
            "bin_idx": int(idx),
            "ts_et": str(ts_series[idx]),
            "mid_price": float(mid_price[idx]),
            # PFP sub-components
            "pfp_final": float(pfp["final"][idx]),
            "pfp_add_signal": float(pfp["add_signal"][idx]),
            "pfp_pull_signal": float(pfp["pull_signal"][idx]),
            "pfp_lead_bid": float(pfp["lead_bid"][idx]),
            "pfp_lead_ask": float(pfp["lead_ask"][idx]),
            "pfp_i_inner_bid": float(pfp["i_inner_bid"][idx]),
            "pfp_i_inner_ask": float(pfp["i_inner_ask"][idx]),
            "pfp_i_outer_bid": float(pfp["i_outer_bid"][idx]),
            "pfp_i_outer_ask": float(pfp["i_outer_ask"][idx]),
            # ADS sub-components
            "ads_final": float(ads["final"][idx]),
            "ads_combined_asym": float(ads["combined_asym"][idx]),
            "ads_slope_10": float(ads["slope_10"][idx]),
            "ads_slope_25": float(ads["slope_25"][idx]),
            "ads_slope_50": float(ads["slope_50"][idx]),
            "ads_z_10": float(ads["z_10"][idx]),
            "ads_z_25": float(ads["z_25"][idx]),
            "ads_z_50": float(ads["z_50"][idx]),
            # ERD sub-components
            "erd_signal_b": float(erd["signal_b"][idx]),
            "erd_signal_a": float(erd["signal_a"][idx]),
            "erd_h_full": float(erd["h_full"][idx]),
            "erd_h_above": float(erd["h_above"][idx]),
            "erd_h_below": float(erd["h_below"][idx]),
            "erd_entropy_asym": float(erd["entropy_asym"][idx]),
            "erd_z_h": float(erd["z_h"][idx]),
            "erd_spike_gate": float(erd["spike_gate"][idx]),
            "erd_score_direction": float(erd["score_direction"][idx]),
            # Spatial vacuum
            "spatial_vac_a": float(spatial["signal_a"][idx]),
            "spatial_vac_c": float(spatial["signal_c"][idx]),
            "spatial_vac_below_sum": float(spatial["vac_below_sum"][idx]),
            "spatial_vac_above_sum": float(spatial["vac_above_sum"][idx]),
            "spatial_pres_below_sum": float(spatial["pres_below_sum"][idx]),
            "spatial_pres_above_sum": float(spatial["pres_above_sum"][idx]),
            # Composite
            "composite": float(composite[idx]),
            # Forward excursion
            "max_up_ticks": excursion["max_up_ticks"],
            "max_down_ticks": excursion["max_down_ticks"],
        }
        bins_list.append(record)

    # Find critical bin: closest to 09:29:58
    critical_ts = pd.Timestamp("2026-02-06 09:29:58", tz="US/Eastern")
    time_diffs = np.abs(
        np.array([(ts_series[i] - critical_ts).total_seconds() for i in indices])
    )
    critical_local_idx = int(np.argmin(time_diffs))
    critical_bin = bins_list[critical_local_idx]

    # Attribution: rank weighted contributions to composite at critical bin
    pfp_contribution = COMPOSITE_PFP_W * critical_bin["pfp_final"]
    ads_contribution = COMPOSITE_ADS_W * critical_bin["ads_final"]
    erd_contribution = COMPOSITE_ERD_W * critical_bin["erd_signal_b"]

    contributions = [
        {"signal": "PFP", "weight": COMPOSITE_PFP_W,
         "raw_value": critical_bin["pfp_final"],
         "weighted_contribution": pfp_contribution},
        {"signal": "ADS", "weight": COMPOSITE_ADS_W,
         "raw_value": critical_bin["ads_final"],
         "weighted_contribution": ads_contribution},
        {"signal": "ERD", "weight": COMPOSITE_ERD_W,
         "raw_value": critical_bin["erd_signal_b"],
         "weighted_contribution": erd_contribution},
    ]
    contributions.sort(key=lambda c: abs(c["weighted_contribution"]), reverse=True)

    attribution = {
        "critical_bin_idx": critical_bin["bin_idx"],
        "critical_ts_et": critical_bin["ts_et"],
        "composite_value": critical_bin["composite"],
        "contributions": contributions,
        "dominant_driver": contributions[0]["signal"],
        "spatial_vac_a_at_critical": critical_bin["spatial_vac_a"],
        "spatial_vac_c_at_critical": critical_bin["spatial_vac_c"],
    }

    return {
        "window_start": str(window_start),
        "window_end": str(window_end),
        "n_bins_in_window": len(bins_list),
        "critical_bin_attribution": attribution,
        "bins": bins_list,
    }


# ===================================================================
# Part 2: Move-size stratified evaluation
# ===================================================================

def _calibrate_thresholds(signal: np.ndarray, warmup: int) -> list[float]:
    """Percentile-calibrated thresholds from signal distribution."""
    active = signal[warmup:]
    abs_signal = np.abs(active)
    pcts = np.nanpercentile(abs_signal, [50, 70, 80, 90, 95, 99])
    thresholds = sorted(set(round(float(v), 6) for v in pcts if v > 0))
    if not thresholds:
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    return thresholds


def _stratify_outcomes(
    outcomes: list[dict[str, Any]],
    mid_price: np.ndarray,
) -> dict[str, Any]:
    """Stratify outcomes by move-size tier and compute per-tier stats.

    Enriches each outcome with max_favorable_ticks, then groups by tier.
    """
    enriched = []
    for o in outcomes:
        mft = compute_max_favorable_ticks(
            mid_price, o["bin_idx"], o["direction"], FORWARD_WINDOW_BINS,
        )
        tier = classify_tier(mft)
        enriched.append({**o, "max_favorable_ticks": mft, "tier": tier})

    tier_names = ["micro", "small", "medium", "large", "extreme"]
    tiers: dict[str, dict[str, Any]] = {}

    for tier_name in tier_names:
        tier_outcomes = [e for e in enriched if e["tier"] == tier_name]
        n = len(tier_outcomes)
        if n == 0:
            tiers[tier_name] = {
                "n": 0, "tp_rate": None, "sl_rate": None,
                "timeout_rate": None, "mean_pnl_ticks": None,
            }
            continue

        n_tp = sum(1 for e in tier_outcomes if e["outcome"] == "tp")
        n_sl = sum(1 for e in tier_outcomes if e["outcome"] == "sl")
        n_timeout = sum(1 for e in tier_outcomes if e["outcome"] == "timeout")
        pnls = [e["pnl_ticks"] for e in tier_outcomes]

        tiers[tier_name] = {
            "n": n,
            "tp_rate": n_tp / n,
            "sl_rate": n_sl / n,
            "timeout_rate": n_timeout / n,
            "mean_pnl_ticks": float(np.mean(pnls)),
        }

    total = len(enriched)
    n_large_or_extreme = sum(
        1 for e in enriched if e["tier"] in ("large", "extreme")
    )
    large_move_selectivity = n_large_or_extreme / total if total > 0 else 0.0

    return {
        "n_total_outcomes": total,
        "tiers": tiers,
        "large_move_selectivity": large_move_selectivity,
    }


def build_stratified(
    *,
    signals_map: dict[str, np.ndarray],
    mid_price: np.ndarray,
    ts_ns: np.ndarray,
) -> dict[str, Any]:
    """Build Part 2 stratified evaluation for all signals.

    For each signal:
      1. Calibrate thresholds from percentiles
      2. For each threshold: detect_signals -> evaluate_tp_sl (with full outcomes)
      3. Enrich with max_favorable_ticks -> stratify by tier
      4. Pick best threshold by TP rate (min 5 signals)

    Returns:
        dict keyed by signal name, each with threshold sweep results.
    """
    result: dict[str, Any] = {}

    for sig_name, signal in signals_map.items():
        print(f"  Stratifying {sig_name} ...", flush=True)
        thresholds = _calibrate_thresholds(signal, WARMUP_BINS)

        threshold_results: list[dict[str, Any]] = []
        for thr in thresholds:
            sigs = detect_signals(signal[WARMUP_BINS:], thr, COOLDOWN_BINS)
            # Adjust bin_idx back to global index
            for s in sigs:
                s["bin_idx"] += WARMUP_BINS

            eval_result = evaluate_tp_sl(
                signals=sigs,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )

            outcomes = eval_result.get("outcomes", [])
            stratified = _stratify_outcomes(outcomes, mid_price)

            threshold_results.append({
                "threshold": thr,
                "n_signals": eval_result["n_signals"],
                "tp_rate": eval_result["tp_rate"],
                "sl_rate": eval_result["sl_rate"],
                "timeout_rate": eval_result["timeout_rate"],
                "mean_pnl_ticks": eval_result["mean_pnl_ticks"],
                "events_per_hour": eval_result["events_per_hour"],
                "stratified": stratified,
            })

        # Best threshold by TP rate with min 5 signals
        valid = [r for r in threshold_results if r["n_signals"] >= 5]
        best = max(valid, key=lambda r: r["tp_rate"]) if valid else None

        result[sig_name] = {
            "thresholds": threshold_results,
            "best_threshold": best["threshold"] if best else None,
            "best_tp_rate": best["tp_rate"] if best else None,
            "best_n_signals": best["n_signals"] if best else None,
            "best_large_move_selectivity": (
                best["stratified"]["large_move_selectivity"] if best else None
            ),
        }

    return result


# ===================================================================
# Part 3: Spatial vacuum sweep + standard results.json
# ===================================================================

def build_spatial_vacuum_sweep(
    *,
    spatial: dict[str, np.ndarray],
    mid_price: np.ndarray,
    ts_ns: np.ndarray,
) -> dict[str, Any]:
    """Sweep spatial vacuum signals through percentile-calibrated thresholds.

    Returns dict with variant_a and variant_c sweep results.
    """
    result: dict[str, Any] = {}

    for variant_key, signal in [
        ("variant_a", spatial["signal_a"]),
        ("variant_c", spatial["signal_c"]),
    ]:
        thresholds = _calibrate_thresholds(signal, WARMUP_BINS)
        sweep: list[dict[str, Any]] = []

        for thr in thresholds:
            sigs = detect_signals(signal[WARMUP_BINS:], thr, COOLDOWN_BINS)
            for s in sigs:
                s["bin_idx"] += WARMUP_BINS

            eval_result = evaluate_tp_sl(
                signals=sigs,
                mid_price=mid_price,
                ts_ns=ts_ns,
            )
            outcomes = eval_result.pop("outcomes", [])
            eval_result["threshold"] = thr
            eval_result["cooldown_bins"] = COOLDOWN_BINS
            sweep.append(eval_result)

        result[variant_key] = sweep

    return result


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    t0 = time.perf_counter()
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("[MSD] Loading dataset ...", flush=True)
    ds = load_dataset(columns=[
        "v_add", "v_pull", "v_fill",
        "spectrum_state_code", "spectrum_score",
        "vacuum_variant", "pressure_variant",
    ])
    v_add: np.ndarray = ds["v_add"]
    v_pull: np.ndarray = ds["v_pull"]
    v_fill: np.ndarray = ds["v_fill"]
    state_code: np.ndarray = ds["spectrum_state_code"]
    spectrum_score: np.ndarray = ds["spectrum_score"]
    vacuum_variant: np.ndarray = ds["vacuum_variant"]
    pressure_variant: np.ndarray = ds["pressure_variant"]
    mid_price: np.ndarray = ds["mid_price"]
    ts_ns: np.ndarray = ds["ts_ns"]
    n_bins: int = ds["n_bins"]
    print(f"[MSD] Loaded: {n_bins} bins, grid shape=({n_bins}, {N_TICKS})", flush=True)
    print(f"[MSD] Load time: {time.perf_counter() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # 2. Compute all decomposed signals
    # ------------------------------------------------------------------
    print("[MSD] Computing PFP decomposed ...", flush=True)
    t1 = time.perf_counter()
    pfp = compute_pfp_decomposed(v_add, v_fill, v_pull)
    print(f"  PFP done in {time.perf_counter() - t1:.2f}s", flush=True)

    print("[MSD] Computing ADS decomposed ...", flush=True)
    t2 = time.perf_counter()
    ads = compute_ads_decomposed(v_add, v_pull)
    print(f"  ADS done in {time.perf_counter() - t2:.1f}s", flush=True)

    print("[MSD] Computing ERD decomposed ...", flush=True)
    t3 = time.perf_counter()
    state_int = state_code.astype(np.int8)
    erd = compute_erd_decomposed(state_int, spectrum_score)
    print(f"  ERD done in {time.perf_counter() - t3:.2f}s", flush=True)

    print("[MSD] Computing spatial vacuum ...", flush=True)
    t4 = time.perf_counter()
    spatial = compute_spatial_vacuum(vacuum_variant, pressure_variant)
    print(f"  Spatial done in {time.perf_counter() - t4:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 3. Composite signal
    # ------------------------------------------------------------------
    print("[MSD] Computing composite (0.40*PFP + 0.35*ADS + 0.25*ERD) ...", flush=True)
    composite = (
        COMPOSITE_PFP_W * pfp["final"]
        + COMPOSITE_ADS_W * ads["final"]
        + COMPOSITE_ERD_W * erd["signal_b"]
    )

    # ------------------------------------------------------------------
    # 4. Part 1: Forensic attribution
    # ------------------------------------------------------------------
    print("[MSD] Building Part 1: forensic attribution ...", flush=True)
    t5 = time.perf_counter()
    forensic = build_forensic(
        ts_ns=ts_ns,
        mid_price=mid_price,
        pfp=pfp,
        ads=ads,
        erd=erd,
        spatial=spatial,
        composite=composite,
        n_bins=n_bins,
    )
    _write_json(out_dir / "forensic.json", forensic)
    print(f"  Part 1 done in {time.perf_counter() - t5:.1f}s", flush=True)

    if "critical_bin_attribution" in forensic:
        attr = forensic["critical_bin_attribution"]
        print(f"  Critical bin: {attr['critical_ts_et']}", flush=True)
        print(f"  Composite: {attr['composite_value']:.6f}", flush=True)
        print(f"  Dominant driver: {attr['dominant_driver']}", flush=True)
        for c in attr["contributions"]:
            print(f"    {c['signal']}: raw={c['raw_value']:.6f}  "
                  f"weighted={c['weighted_contribution']:.6f}", flush=True)

    # ------------------------------------------------------------------
    # 5. Part 2: Stratified evaluation
    # ------------------------------------------------------------------
    print("[MSD] Building Part 2: stratified evaluation ...", flush=True)
    t6 = time.perf_counter()

    signals_map: dict[str, np.ndarray] = {
        "pfp": pfp["final"],
        "ads": ads["final"],
        "erd": erd["signal_b"],
        "composite": composite,
        "spatial_vacuum_a": spatial["signal_a"],
        "spatial_vacuum_weighted": spatial["signal_c"],
    }

    stratified = build_stratified(
        signals_map=signals_map,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )
    _write_json(out_dir / "stratified.json", stratified)
    print(f"  Part 2 done in {time.perf_counter() - t6:.1f}s", flush=True)

    # Print summary
    for sig_name, sig_data in stratified.items():
        best_tp = sig_data.get("best_tp_rate")
        best_n = sig_data.get("best_n_signals")
        best_sel = sig_data.get("best_large_move_selectivity")
        tp_str = f"{best_tp:.1%}" if best_tp is not None else "N/A"
        sel_str = f"{best_sel:.1%}" if best_sel is not None else "N/A"
        n_str = str(best_n) if best_n is not None else "N/A"
        print(f"  {sig_name}: best_TP={tp_str}  n={n_str}  "
              f"large_move_sel={sel_str}", flush=True)

    # ------------------------------------------------------------------
    # 6. Part 3: Spatial vacuum sweep + standard results.json
    # ------------------------------------------------------------------
    print("[MSD] Building Part 3: spatial vacuum sweep ...", flush=True)
    t7 = time.perf_counter()
    spatial_sweep = build_spatial_vacuum_sweep(
        spatial=spatial,
        mid_price=mid_price,
        ts_ns=ts_ns,
    )
    _write_json(out_dir / "spatial_vacuum.json", spatial_sweep)

    # Write standard results.json for comparison.py compatibility
    # Use variant_c (distance-weighted) as primary signal for results.json
    variant_c_results = spatial_sweep["variant_c"]
    write_results(
        agent_name=AGENT_NAME,
        experiment_name=EXPERIMENT_NAME,
        params={
            "columns": [
                "v_add", "v_pull", "v_fill",
                "spectrum_state_code", "spectrum_score",
                "vacuum_variant", "pressure_variant",
            ],
            "composite_weights": {
                "pfp": COMPOSITE_PFP_W,
                "ads": COMPOSITE_ADS_W,
                "erd": COMPOSITE_ERD_W,
            },
            "cooldown_bins": COOLDOWN_BINS,
            "warmup_bins": WARMUP_BINS,
            "forward_window_bins": FORWARD_WINDOW_BINS,
            "spatial_vacuum_primary": "variant_c",
        },
        results_by_threshold=variant_c_results,
    )
    print(f"  Part 3 done in {time.perf_counter() - t7:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t0
    print(f"\n[MSD] Complete. Total runtime: {elapsed:.1f}s", flush=True)
    print(f"[MSD] Outputs:", flush=True)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f}", flush=True)


if __name__ == "__main__":
    main()
