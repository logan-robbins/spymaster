"""Stream builder for Pentaview - STREAMS.md Sections 3-5."""
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy.fft import dct

from src.ml.stream_normalization import normalize_feature

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS (Section 9.1)
# ============================================================================

def wavg(values: List[float], decay: float = 0.7) -> float:
    """
    Weighted average with exponential decay.
    
    Per STREAMS.md Section 9.1:
    - Values ordered shortest→longest scale
    - Weights: [1, decay, decay^2, ...]
    
    Args:
        values: List of values (shortest to longest scale)
        decay: Decay factor (default 0.7)
    
    Returns:
        Weighted average
    """
    if not values:
        return 0.0
    
    weights = [decay ** i for i in range(len(values))]
    return sum(w * v for w, v in zip(weights, values)) / sum(weights)


def trend_score(dct_coeffs: np.ndarray) -> float:
    """
    Compute trend score from DCT coefficients.
    
    Per STREAMS.md Section 9.1:
    - Emphasize c1 (trend), c2 (curvature)
    - Formula: tanh(0.8*c1 + 0.2*c2)
    
    Args:
        dct_coeffs: DCT coefficients array (at least 3 elements)
    
    Returns:
        Trend score in (-1, +1)
    """
    if len(dct_coeffs) < 3:
        return 0.0
    
    c1 = dct_coeffs[1]
    c2 = dct_coeffs[2]
    
    return float(np.tanh(0.8 * c1 + 0.2 * c2))


def chop_score(dct_coeffs: np.ndarray) -> float:
    """
    Compute chop (high-frequency energy) score from DCT coefficients.
    
    Per STREAMS.md Section 9.1:
    - Formula: high_freq_energy / total_energy
    - Low frequency: |c1| + |c2|
    - High frequency: sum(|c3|..|c7|)
    
    Args:
        dct_coeffs: DCT coefficients array (at least 8 elements)
    
    Returns:
        Chop score in [0, 1]
    """
    if len(dct_coeffs) < 8:
        return 0.0
    
    low_energy = np.abs(dct_coeffs[1]) + np.abs(dct_coeffs[2])
    high_energy = np.sum(np.abs(dct_coeffs[3:8]))
    total_energy = low_energy + high_energy
    
    if total_energy < 1e-8:
        return 0.0
    
    return float(np.clip(high_energy / (total_energy + 1e-8), 0, 1))


def ema(x_t: float, prev: float, halflife_bars: float = 3.0) -> float:
    """
    Exponential moving average.
    
    Args:
        x_t: Current value
        prev: Previous EMA value
        halflife_bars: Halflife in bars (default 3)
    
    Returns:
        Updated EMA
    """
    alpha = 1 - np.exp(np.log(0.5) / halflife_bars)
    return alpha * x_t + (1 - alpha) * prev


# ============================================================================
# STREAM COMPUTATION (Section 9.2)
# ============================================================================

def compute_momentum_stream(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    stratum: Optional[str] = None
) -> float:
    """
    Compute Σ_M: MOMENTUM stream (directional).
    
    Per STREAMS.md Section 3.1:
    - Inputs: velocity, acceleration, jerk, momentum_trend (multi-scale)
    - Formula: 0.40*vel + 0.30*acc + 0.15*jerk + 0.15*trend
    - Semantics: Σ_M > 0 = upward momentum, < 0 = downward
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        stratum: Optional stratum key
    
    Returns:
        Momentum stream value in (-1, +1)
    """
    # Normalize velocity components
    v = [
        normalize_feature('velocity_1min', bar_row.get('velocity_1min', 0.0), stats, stratum),
        normalize_feature('velocity_3min', bar_row.get('velocity_3min', 0.0), stats, stratum),
        normalize_feature('velocity_5min', bar_row.get('velocity_5min', 0.0), stats, stratum),
        normalize_feature('velocity_10min', bar_row.get('velocity_10min', 0.0), stats, stratum),
        normalize_feature('velocity_20min', bar_row.get('velocity_20min', 0.0), stats, stratum),
    ]
    
    # Normalize acceleration components
    a = [
        normalize_feature('acceleration_1min', bar_row.get('acceleration_1min', 0.0), stats, stratum),
        normalize_feature('acceleration_3min', bar_row.get('acceleration_3min', 0.0), stats, stratum),
        normalize_feature('acceleration_5min', bar_row.get('acceleration_5min', 0.0), stats, stratum),
        normalize_feature('acceleration_10min', bar_row.get('acceleration_10min', 0.0), stats, stratum),
        normalize_feature('acceleration_20min', bar_row.get('acceleration_20min', 0.0), stats, stratum),
    ]
    
    # Normalize jerk components (emphasize shorter scales)
    j = [
        normalize_feature('jerk_1min', bar_row.get('jerk_1min', 0.0), stats, stratum),
        normalize_feature('jerk_3min', bar_row.get('jerk_3min', 0.0), stats, stratum),
        normalize_feature('jerk_5min', bar_row.get('jerk_5min', 0.0), stats, stratum),
    ]
    
    # Normalize momentum trend components
    mt = [
        normalize_feature('momentum_trend_3min', bar_row.get('momentum_trend_3min', 0.0), stats, stratum),
        normalize_feature('momentum_trend_5min', bar_row.get('momentum_trend_5min', 0.0), stats, stratum),
        normalize_feature('momentum_trend_10min', bar_row.get('momentum_trend_10min', 0.0), stats, stratum),
        normalize_feature('momentum_trend_20min', bar_row.get('momentum_trend_20min', 0.0), stats, stratum),
    ]
    
    # Weighted aggregation per STREAMS.md Section 3.1
    vel = wavg(v, decay=0.7)
    acc = wavg(a, decay=0.7)
    jer = wavg(j, decay=0.8)
    trd = wavg(mt, decay=0.7)
    
    # Combine with weights: 0.40*vel + 0.30*acc + 0.15*jer + 0.15*trd
    s_raw = 0.40 * vel + 0.30 * acc + 0.15 * jer + 0.15 * trd
    
    # Final tanh squashing
    return float(np.tanh(s_raw))


def compute_flow_stream(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    dct_ofi: Optional[np.ndarray] = None,
    dct_tape: Optional[np.ndarray] = None,
    stratum: Optional[str] = None
) -> float:
    """
    Compute Σ_F: FLOW stream (directional).
    
    Per STREAMS.md Section 3.2:
    - Inputs: OFI, tape imbalance/velocity, flow alignment, DCT trend
    - Formula: 0.25*ofi_core + 0.20*ofi_level + 0.25*imb + 0.15*acc + 0.10*aln + 0.05*shape
    - Semantics: Σ_F > 0 = net buying aggression, < 0 = net selling
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        dct_ofi: Optional DCT coefficients for ofi_60s trajectory
        dct_tape: Optional DCT coefficients for tape_imbalance trajectory
        stratum: Optional stratum key
    
    Returns:
        Flow stream value in (-1, +1)
    """
    # Core OFI
    ofi_core = normalize_feature('ofi_60s', bar_row.get('ofi_60s', 0.0), stats, stratum)
    
    # Level-specific OFI
    ofi_level = normalize_feature('ofi_near_level_60s', bar_row.get('ofi_near_level_60s', 0.0), stats, stratum)
    
    # Tape imbalance (use t-2, t-1, t0 if available, else just t0)
    tape_imb_t0 = normalize_feature('tape_imbalance', bar_row.get('tape_imbalance', 0.0), stats, stratum)
    # For 2-min bars, we won't have t-1/t-2 from micro-history, so just use current
    imb = tape_imb_t0
    
    # OFI acceleration
    acc_flow = normalize_feature('ofi_acceleration', bar_row.get('ofi_acceleration', 0.0), stats, stratum)
    
    # Flow alignment
    aln = normalize_feature('flow_alignment', bar_row.get('flow_alignment', 0.0), stats, stratum)
    
    # DCT-derived shape (trend component)
    shape = 0.0
    if dct_ofi is not None and len(dct_ofi) >= 3:
        shape += 0.5 * trend_score(dct_ofi)
    if dct_tape is not None and len(dct_tape) >= 3:
        shape += 0.5 * trend_score(dct_tape)
    
    # Combine with weights per STREAMS.md Section 3.2
    s_raw = (
        0.25 * ofi_core +
        0.20 * ofi_level +
        0.25 * imb +
        0.15 * acc_flow +
        0.10 * aln +
        0.05 * shape
    )
    
    return float(np.tanh(s_raw))


def compute_barrier_stream(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    stratum: Optional[str] = None
) -> float:
    """
    Compute Σ_B: BARRIER stream (directional via dir_sign).
    
    Per STREAMS.md Section 3.3:
    - Inputs: barrier consumption/replenishment, wall ratio, state
    - CRITICAL: Apply dir_sign multiplier for market semantics
    - Formula: dir_sign * tanh(0.50*consume + 0.25*rate + 0.15*state + 0.10*repl)
    - Semantics: Σ_B > 0 = favor up, < 0 = favor down
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        stratum: Optional stratum key
    
    Returns:
        Barrier stream value in (-1, +1)
    """
    # Direction sign per STREAMS.md Section 3.3
    direction = bar_row.get('direction', 'UP')
    dir_sign = +1.0 if direction == 'UP' else -1.0
    
    # Consumption (negative delta_liq means consumption)
    barrier_delta_liq_log = bar_row.get('barrier_delta_liq_log', 0.0)
    consume = -normalize_feature('barrier_delta_liq_log', barrier_delta_liq_log, stats, stratum)
    
    # Barrier rate
    rate = normalize_feature('barrier_delta_3min', bar_row.get('barrier_delta_3min', 0.0), stats, stratum)
    
    # Barrier state
    barrier_state_encoded = bar_row.get('barrier_state_encoded', 0)
    state = np.clip(barrier_state_encoded / 2.0, -1, 1)  # Normalize [-2..+2] → [-1..+1]
    
    # Replenishment (>0 means rebuilding)
    repl_ratio = bar_row.get('barrier_replenishment_ratio', 1.0) - 1.0
    repl = normalize_feature('barrier_replenishment_ratio', repl_ratio, stats, stratum)
    
    # Local score (absorption favors break, replenishment favors reject)
    s_local = 0.50 * consume + 0.25 * (-rate) + 0.15 * (-state) + 0.10 * (-repl)
    
    # Apply direction sign for market semantics
    return float(np.tanh(dir_sign * s_local))


def compute_dealer_stream(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    stratum: Optional[str] = None
) -> float:
    """
    Compute Σ_D: DEALER/GAMMA stream (non-directional amplifier).
    
    Per STREAMS.md Section 3.4:
    - Inputs: fuel_effect, gamma_exposure, gex_ratio, net_gex
    - Formula: 0.45*fuel + 0.25*(-ge) + 0.15*(-ratio) + 0.15*(-abs(local))
    - Semantics: Σ_D > 0 = amplification (fuel), < 0 = dampening (pin)
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        stratum: Optional stratum key
    
    Returns:
        Dealer stream value in (-1, +1)
    """
    # Fuel effect (primary indicator)
    fuel = float(bar_row.get('fuel_effect_encoded', 0))  # Already in {-1, 0, +1}
    
    # Gamma exposure (normalize)
    ge = normalize_feature('gamma_exposure', bar_row.get('gamma_exposure', 0.0), stats, stratum)
    
    # GEX ratio
    ratio = normalize_feature('gex_ratio', bar_row.get('gex_ratio', 0.0), stats, stratum)
    
    # Local GEX asymmetry
    local = normalize_feature('net_gex_2strike', bar_row.get('net_gex_2strike', 0.0), stats, stratum)
    
    # Combine per STREAMS.md Section 3.4
    s_raw = 0.45 * fuel + 0.25 * (-ge) + 0.15 * (-ratio) + 0.15 * (-np.abs(local))
    
    return float(np.tanh(s_raw))


def compute_setup_stream(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    dct_d_atr: Optional[np.ndarray] = None,
    stratum: Optional[str] = None
) -> float:
    """
    Compute Σ_S: SETUP/QUALITY stream (non-directional confidence).
    
    Per STREAMS.md Section 3.5:
    - Inputs: proximity, approach, freshness, confluence, trajectory cleanness
    - Formula: weighted sum → [0,1] → map to [-1,+1]
    - Semantics: Σ_S > +0.5 = high quality, < -0.5 = degraded
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        dct_d_atr: Optional DCT coefficients for distance trajectory
        stratum: Optional stratum key
    
    Returns:
        Setup stream value in (-1, +1)
    """
    # Proximity (exponential decay from level)
    d_atr = bar_row.get('distance_signed_atr', 0.0)
    proximity = np.exp(-np.abs(d_atr) / 1.5)
    
    # Recency (time since last touch)
    time_since_last_touch_sec = bar_row.get('time_since_last_touch_sec', 900.0)
    recency = np.exp(-time_since_last_touch_sec / 900.0)
    
    # Freshness (attempt index)
    attempt_index = bar_row.get('attempt_index', 0)
    if attempt_index <= 2:
        freshness = 1.0
    elif attempt_index <= 4:
        freshness = 0.6
    else:
        freshness = 0.3
    
    # Confluence (level stacking)
    stacking_5pt = bar_row.get('level_stacking_5pt', 0)
    stacking_10pt = bar_row.get('level_stacking_10pt', 0)
    confluence = np.clip((stacking_5pt + stacking_10pt) / 12.0, 0, 1)
    
    # Approach speed
    approach_velocity = bar_row.get('approach_velocity', 0.0)
    approach_speed_norm = normalize_feature('approach_velocity', approach_velocity, stats, stratum)
    approach_speed = np.tanh(approach_speed_norm)
    
    # Trajectory cleanness from DCT
    clean_trend = 0.5
    chop_pen = 0.0
    if dct_d_atr is not None and len(dct_d_atr) >= 8:
        clean_trend = 0.5 + 0.5 * trend_score(dct_d_atr)
        chop_pen = chop_score(dct_d_atr)
    
    # Combine per STREAMS.md Section 3.5
    q_0_1 = (
        0.28 * proximity +
        0.18 * (0.5 + 0.5 * approach_speed) +
        0.18 * freshness +
        0.16 * confluence +
        0.10 * recency +
        0.10 * clean_trend -
        0.10 * chop_pen
    )
    q_0_1 = np.clip(q_0_1, 0, 1)
    
    # Map [0,1] to [-1,+1]
    return float(2 * q_0_1 - 1)


def compute_all_streams(
    bar_row: pd.Series,
    stats: Dict[str, Any],
    dct_coeffs: Optional[Dict[str, np.ndarray]] = None,
    stratum: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute all 5 canonical streams + merged streams + composites.
    
    Per STREAMS.md Sections 3-5.
    
    Args:
        bar_row: 2-minute bar with aggregated features
        stats: Normalization statistics
        dct_coeffs: Optional dictionary with DCT coefficients {
            'd_atr': np.ndarray,
            'ofi_60s': np.ndarray,
            'barrier_delta_liq_log': np.ndarray,
            'tape_imbalance': np.ndarray
        }
        stratum: Optional stratum key for stratified normalization
    
    Returns:
        Dictionary with all stream values:
        - sigma_m, sigma_f, sigma_b, sigma_d, sigma_s (canonical)
        - sigma_p, sigma_r (merged)
        - alignment, divergence, alignment_adj (composites)
    """
    dct_coeffs = dct_coeffs or {}
    
    # Compute 5 canonical streams
    sigma_m = compute_momentum_stream(bar_row, stats, stratum)
    sigma_f = compute_flow_stream(
        bar_row, stats,
        dct_ofi=dct_coeffs.get('ofi_60s'),
        dct_tape=dct_coeffs.get('tape_imbalance'),
        stratum=stratum
    )
    sigma_b = compute_barrier_stream(bar_row, stats, stratum)
    sigma_d = compute_dealer_stream(bar_row, stats, stratum)
    sigma_s = compute_setup_stream(
        bar_row, stats,
        dct_d_atr=dct_coeffs.get('d_atr'),
        stratum=stratum
    )
    
    # Compute merged streams per STREAMS.md Section 5.1-5.2
    # Σ_P: PRESSURE (Momentum + Flow)
    sigma_p = float(np.tanh(0.55 * sigma_m + 0.45 * sigma_f))
    
    # Σ_R: STRUCTURE (Barrier + Setup)
    sigma_r = float(np.tanh(0.70 * sigma_b + 0.30 * sigma_s))
    
    # Compute composites per STREAMS.md Section 5.3
    # Directional alignment (exclude dealer, weight by setup)
    a_dir = (sigma_m + sigma_f + sigma_b) / 3.0
    alignment = a_dir * (0.6 + 0.4 * (0.5 + 0.5 * sigma_s))
    
    # Divergence (standard deviation of directional streams)
    divergence = float(np.std([sigma_m, sigma_f, sigma_b]))
    
    # Dealer-adjusted alignment per STREAMS.md Section 5.4
    k_d = 0.35
    alignment_adj = float(np.clip(alignment * (1.0 + k_d * sigma_d), -1, 1))
    
    return {
        'sigma_m': sigma_m,
        'sigma_f': sigma_f,
        'sigma_b': sigma_b,
        'sigma_d': sigma_d,
        'sigma_s': sigma_s,
        'sigma_p': sigma_p,
        'sigma_r': sigma_r,
        'alignment': alignment,
        'divergence': divergence,
        'alignment_adj': alignment_adj,
    }


def compute_derivatives(
    stream_history: pd.DataFrame,
    stream_name: str,
    halflife_bars: float = 3.0
) -> Dict[str, float]:
    """
    Compute derivatives for a stream per STREAMS.md Section 4.
    
    Args:
        stream_history: DataFrame with stream values over time
        stream_name: Name of stream column (e.g., 'sigma_p')
        halflife_bars: EMA halflife for smoothing (default 3)
    
    Returns:
        Dictionary with smoothed value, slope, curvature, jerk
    """
    if len(stream_history) < 3:
        return {
            'smooth': 0.0,
            'slope': 0.0,
            'curvature': 0.0,
            'jerk': 0.0
        }
    
    # Apply EMA smoothing
    smooth_col = f'{stream_name}_smooth'
    stream_history[smooth_col] = stream_history[stream_name].ewm(halflife=halflife_bars).mean()
    
    # Compute derivatives
    smooth_vals = stream_history[smooth_col].values
    
    # Slope (1st difference)
    slope = smooth_vals[-1] - smooth_vals[-2] if len(smooth_vals) >= 2 else 0.0
    
    # Curvature (2nd difference)
    if len(smooth_vals) >= 3:
        slope_prev = smooth_vals[-2] - smooth_vals[-3]
        curvature = slope - slope_prev
    else:
        curvature = 0.0
    
    # Jerk (3rd difference)
    if len(smooth_vals) >= 4:
        slope_prev2 = smooth_vals[-3] - smooth_vals[-4]
        curvature_prev = slope_prev - slope_prev2
        jerk = curvature - curvature_prev
    else:
        jerk = 0.0
    
    return {
        'smooth': float(smooth_vals[-1]),
        'slope': float(slope),
        'curvature': float(curvature),
        'jerk': float(jerk)
    }

