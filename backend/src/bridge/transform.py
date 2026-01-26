import numpy as np

# Tuning constants (V1)
W_LOG_MAX = 10.0  # Log(1 + large_qty) -> e.g. ln(20000) ~ 9.9
E_LOG_MAX = 5.0   # Log(1 + erosion)

def compute_wall_intensity(depth_qty_rest: float) -> float:
    """
    Intensity = clamp(log1p(depth) / W_LOG_MAX, 0, 1)
    """
    val = np.log1p(depth_qty_rest)
    return np.clip(val / W_LOG_MAX, 0.0, 1.0)

def compute_wall_erosion(d1_depth_qty: float, depth_start: float = 1.0) -> float:
    """
    If erosion not provided: max(-d1, 0)
    Then normalize.
    """
    # Raw erosion rate
    erosion_raw = max(-d1_depth_qty, 0.0)
    # Normalized relative to depth (if we wanted relative) or absolute?
    # IMPLEMENT.md says: wall_erosion_norm = clamp(log(1 + erosion_raw / (depth_start + EPS)) / E_LOG_MAX, 0, 1)
    
    eps = 1.0
    val = np.log1p(erosion_raw / (depth_start + eps))
    return np.clip(val / E_LOG_MAX, 0.0, 1.0)

def normalize_vacuum(vac_score: float) -> float:
    return float(np.clip(vac_score, 0.0, 1.0))

def normalize_physics(score_signed: float) -> float:
    # Passed as-is, just float cast
    return float(score_signed)
