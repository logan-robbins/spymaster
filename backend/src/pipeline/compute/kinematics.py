"""
Shared kinematics computation functions.

Computes velocity, acceleration, jerk using Savitzky-Golay filters.
Can be signed (relative to direction) or unsigned (absolute).
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Optional
from scipy.signal import convolve


def get_trailing_savgol_coeffs(window_len: int, poly_order: int, deriv: int = 0) -> np.ndarray:
    """
    Compute Savitzky-Golay coefficients for the END POINT of a trailing window.
    """
    t = np.arange(-(window_len - 1), 1)
    X = np.vander(t, N=poly_order + 1, increasing=True)
    X_pinv = np.linalg.pinv(X)
    
    if deriv >= X_pinv.shape[0]:
        return np.zeros(window_len)
        
    weights = X_pinv[deriv, :] * float(math.factorial(deriv))
    return weights[::-1]


def compute_kinematics_series(
    price_arr: np.ndarray,
    windows_minutes: List[int],
    dt: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Compute continuous kinematics series from price array.
    
    Args:
        price_arr: Array of prices (10s grid)
        windows_minutes: List of window sizes in minutes
        dt: Time step in seconds (default 10s)
    
    Returns:
        Dictionary of physics series (velocity, acceleration, jerk)
    """
    physics_series = {}
    
    for w_min in windows_minutes:
        n_bars = int(w_min * 60 / dt)
        if n_bars < 3:
            n_bars = 3
        
        # Velocity (Linear Slope)
        w_vel = get_trailing_savgol_coeffs(n_bars, poly_order=1, deriv=1)
        vel_raw = convolve(price_arr, w_vel, mode='valid')
        vel_aligned = np.full_like(price_arr, np.nan)
        vel_aligned[n_bars-1:] = vel_raw
        physics_series[f'velocity_{w_min}min'] = vel_aligned / dt
        
        # Acceleration (Quadratic)
        w_acc = get_trailing_savgol_coeffs(n_bars, poly_order=2, deriv=2)
        acc_raw = convolve(price_arr, w_acc, mode='valid')
        acc_aligned = np.full_like(price_arr, np.nan)
        acc_aligned[n_bars-1:] = acc_raw
        physics_series[f'acceleration_{w_min}min'] = acc_aligned / (dt**2)
        
        # Jerk (Cubic) - only for small windows
        if w_min <= 5:
            w_jerk = get_trailing_savgol_coeffs(n_bars, poly_order=3, deriv=3)
            jerk_raw = convolve(price_arr, w_jerk, mode='valid')
            jerk_aligned = np.full_like(price_arr, np.nan)
            jerk_aligned[n_bars-1:] = jerk_raw
            physics_series[f'jerk_{w_min}min'] = jerk_aligned / (dt**3)
    
    return physics_series


def compute_kinematics_windows(
    signal_ts: np.ndarray,
    ohlcv_df: pd.DataFrame,
    windows_minutes: List[int],
    directions: Optional[np.ndarray] = None,
    signed: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Compute kinematics at signal timestamps.
    
    Args:
        signal_ts: Array of signal timestamps (ns)
        ohlcv_df: OHLCV DataFrame (10s)
        windows_minutes: List of window sizes in minutes
        directions: Array of directions ('UP'/'DOWN') for signed mode
        signed: If True, multiply by direction sign. If False, return absolute values.
    
    Returns:
        Dictionary of kinematics features
    """
    result = {}
    n_signals = len(signal_ts)
    
    if n_signals == 0 or ohlcv_df.empty:
        for w_min in windows_minutes:
            result[f'velocity_{w_min}min'] = np.array([])
            result[f'acceleration_{w_min}min'] = np.array([])
            if w_min <= 5:
                result[f'jerk_{w_min}min'] = np.array([])
        return result
    
    # Prepare dense price series
    ohlcv = ohlcv_df.copy()
    if 'timestamp' not in ohlcv.columns and isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv = ohlcv.reset_index()
    if 'timestamp' not in ohlcv.columns:
        ohlcv = ohlcv.rename(columns={'index': 'timestamp'})
    
    ohlcv = ohlcv.sort_values('timestamp').set_index('timestamp')
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep='first')]
    
    dense_close = ohlcv['close'].resample('10s').ffill(limit=1)
    dense_close = dense_close.interpolate(method='linear', limit=3, limit_area='inside')
    
    dense_ts_ns = dense_close.index.values.astype(np.int64)
    price_arr = dense_close.values.astype(np.float64)
    
    # Compute continuous physics
    physics_series = compute_kinematics_series(price_arr, windows_minutes)
    
    # Lookup values at signal timestamps
    idx_lookup = np.searchsorted(dense_ts_ns, signal_ts, side='right') - 1
    idx_lookup = np.clip(idx_lookup, 0, len(dense_ts_ns) - 1)
    
    # Direction sign
    if signed and directions is not None:
        dir_sign = np.where(directions == 'UP', 1.0, -1.0)
    else:
        dir_sign = np.ones(n_signals)
    
    for feat_name, arr in physics_series.items():
        base_values = arr[idx_lookup]
        
        if signed:
            result[feat_name] = base_values * dir_sign
        else:
            # Unsigned (absolute) for global pipeline
            result[f'{feat_name}_abs'] = np.abs(base_values)
            result[feat_name] = base_values  # Also keep signed version
        
        # Momentum trend (legacy)
        if 'acceleration' in feat_name:
            trend_name = feat_name.replace('acceleration', 'momentum_trend')
            result[trend_name] = result[feat_name]
    
    return result

