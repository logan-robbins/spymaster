import pandas as pd
import numpy as np
import math
from typing import List, Dict, Any
from scipy.signal import convolve

from src.pipeline.core.stage import BaseStage, StageContext
from src.common.config import CONFIG


def get_trailing_savgol_coeffs(window_len: int, poly_order: int, deriv: int = 0) -> np.ndarray:
    """
    Compute Savitzky-Golay coefficients for the END POINT of a trailing window.
    
    Standard savgol_coeffs are for the midpoint. We need the coefficients 
    that estimating the derivative at t=0 given history t=[-N+1, ... 0].
    
    y = X * c
    c = (X^T * X)^-1 * X^T * y
    
    We want the k-th derivative at t=0.
    This corresponds to c[k] * factorial(k).
    
    So weights w = kth_row_of_inverse * X^T
    """
    # Time vector: [-(N-1), ..., -1, 0]
    t = np.arange(-(window_len - 1), 1)
    
    # Vandermonde matrix: X[i, j] = t[i] ^ j
    # shape (N, poly_order + 1)
    # columns: 1, t, t^2, ...
    X = np.vander(t, N=poly_order + 1, increasing=True)
    
    # Least squares pseudo-inverse: (X^T X)^-1 X^T
    # We can use pinv
    X_pinv = np.linalg.pinv(X)
    
    # We want the coefficient for t^deriv (c_deriv)
    # The derivative k of the polynomial at t=0 is k! * c_k
    # So we take the row 'deriv' from X_pinv and multiply by factorial(deriv)
    if deriv >= X_pinv.shape[0]:
        return np.zeros(window_len)
        
    weights = X_pinv[deriv, :] * float(math.factorial(deriv))
    
    # Note: convolutions flip the kernel. 
    # If we do y[t] = sum(w[i] * x[t-i]), this matches the definition of dot product with history.
    # However, scipy.signal.convolve(data, kernel, mode='valid') applies kernel sliding.
    # If kernel is [w_0, w_1...], it aligns such that end of kernel hits end of text for 'valid'?
    # Let's verify orientation.
    # convolution: (f * g)[n] = sum(f[m] g[n-m])
    # We want y[t] = w_0*x[t-(N-1)] + ... + w_{N-1}*x[t]
    # So we need to reverse the weights for using in convolve/correlate.
    # Actually, let's just use the weights as is and check which function applies dot product.
    # We'll use result[i] = dot(weights, x[i-N+1:i+1]).
    # This means weights corresponds to x relative to current t.
    # weights[0] corresponds to t-(N-1), weights[-1] to t.
    # Scipy convolve flips kernel. So if we pass weights[::-1], it applies them "forward".
    return weights[::-1]


def compute_multiwindow_kinematics(
    signals_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    windows_minutes: List[int] = None
) -> pd.DataFrame:
    """
    Compute kinematics at multiple lookback windows using Vectorized Convolutions.
    
    Optimization Strategy:
    - Resample OHLCV to strict 10s grid (dense).
    - For each window size:
      - Compute Trailing Savitzky-Golay kernels for Vel/Accel/Jerk.
      - Convolve Price series with kernels to get continuous physics.
      - Lookup values at Signal timestamps O(1).
    
    Args:
        signals_df: DataFrame with signals
        ohlcv_df: OHLCV DataFrame (10s)
        windows_minutes: List of lookback windows (default: [1, 3, 5, 10, 20])
    
    Returns:
        DataFrame with kinematics
    """
    if windows_minutes is None:
        windows_minutes = [1, 2, 3, 5, 10, 20]
    
    if signals_df.empty or ohlcv_df.empty:
        return signals_df
        
    # 1. Prepare Dense Price Series (10s grid)
    # ----------------------------------------
    ohlcv = ohlcv_df.copy()
    if 'timestamp' not in ohlcv.columns and isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv = ohlcv.reset_index()
    if 'timestamp' not in ohlcv.columns:
         ohlcv = ohlcv.rename(columns={'index': 'timestamp'})
         
    # Ensure sorted unique index
    ohlcv = ohlcv.sort_values('timestamp').set_index('timestamp')
    # Remove duplicates
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep='first')]
    
    # Resample to 10s to ensure fixed grid
    # We assume 'close' is the price
    # ffill to handle gaps
    dense_close = ohlcv['close'].resample('10s').ffill()
    dense_ts = dense_close.index
    dense_ts_ns = dense_ts.values.astype(np.int64)
    price_arr = dense_close.values.astype(np.float64)
    
    # 2. Compute Continuous Physics via Convolution
    # --------------------------------------------
    # We store the "Physics Series" in a dictionary
    physics_series = {} 
    
    # Pre-calculate kernels for all windows
    # Unit of time T is "steps" (10s).
    # Velocity output will be price/10s. We need to scale to price/sec?
    # Original code: t in seconds, so slope is price/sec.
    # If we use steps, slope is price/10s.
    # So we must divide Velocity by 10.
    # Accel by 100.
    # Jerk by 1000.
    dt = 10.0 
    
    for w_min in windows_minutes:
        n_bars = int(w_min * 60 / dt)
        if n_bars < 3:
            n_bars = 3 # Minimum for polyfit
        
        # Poly orders:
        # Vel: 1 (Linear fit)
        # Accel: 2 (Quadratic fit)
        # Jerk: 3 (Cubic fit)
        
        # Velocity (Linear Slope)
        w_vel = get_trailing_savgol_coeffs(n_bars, poly_order=1, deriv=1)
        # Convolve (valid mode returns len(arr) - len(kernel) + 1)
        # We need output aligned with input.
        # "full" mode gives (N+K-1).
        # We want the value at index T to depend on T-N+1...T.
        # Scipy convolve with 'full': result[i] corresponds to overlap centered at i?
        # Let's use 'valid'. result[0] corresponds to index N-1 of input.
        # So we pad result with N-1 nans at start to align.
        
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
        
        # Jerk (Cubic) - only for small windows per schema
        if w_min <= 5:
            w_jerk = get_trailing_savgol_coeffs(n_bars, poly_order=3, deriv=3)
            jerk_raw = convolve(price_arr, w_jerk, mode='valid')
            jerk_aligned = np.full_like(price_arr, np.nan)
            jerk_aligned[n_bars-1:] = jerk_raw
            physics_series[f'jerk_{w_min}min'] = jerk_aligned / (dt**3)
            
    # 3. Lookup Values for Signals
    # ----------------------------
    result = signals_df.copy()
    sig_ts = result['ts_ns'].values.astype(np.int64)
    directions = result['direction'].values
    dir_sign = np.where(directions == 'UP', 1.0, -1.0)
    
    # Find nearest index in dense series
    # searchsorted returns index where value would be inserted
    # dense_ts_ns is convex (sorted)
    # We want the index corresponding to the signal timestamp (or immediately preceding)
    # Because we ffilled, we want 'right' - 1?
    # Actually, we want the exact bar containing the timestamp.
    # Since dense is 10s grid, we can just compute index?
    # No, gaps might exist if we didn't reindex perfectly or if day starts late.
    # searchsorted is safest.
    
    idx_lookup = np.searchsorted(dense_ts_ns, sig_ts, side='right') - 1
    # Check bounds
    idx_lookup = np.clip(idx_lookup, 0, len(dense_ts_ns) - 1)
    
    # Validate timestamp match (optional, ensure we aren't looking up too far away)
    # retrieved_ts = dense_ts_ns[idx_lookup]
    # diff = np.abs(sig_ts - retrieved_ts)
    # 10s = 1e10 ns.
    # mask_valid = diff <= 2e10 # Allow slight misalignment
    
    for feat_name, arr in physics_series.items():
        base_values = arr[idx_lookup]
        # Physics is relative to approach direction
        # velocity UP means price increasing.
        # If direction is UP (approaching from below), positive price velocity = approach velocity.
        # If direction is DOWN (approaching from above), negative price velocity = approach velocity.
        # So: feat = dir_sign * val
        
        result[feat_name] = base_values * dir_sign
        
        # Handle Momentum Trend (legacy/proxy)
        if 'acceleration' in feat_name:
            # momentum_trend was proxy for acceleration
            trend_name = feat_name.replace('acceleration', 'momentum_trend')
            result[trend_name] = result[feat_name]

    return result


class ComputeMultiWindowKinematicsStage(BaseStage):
    """Compute kinematics at multiple lookback windows using vectorized convolutions."""
    
    @property
    def name(self) -> str:
        return "compute_multiwindow_kinematics"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df', 'ohlcv_10s']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        ohlcv_df = ctx.data['ohlcv_10s']
        
        signals_df = compute_multiwindow_kinematics(
            signals_df=signals_df,
            ohlcv_df=ohlcv_df,
            windows_minutes=[1, 2, 3, 5, 10, 20]
        )
        
        return {'signals_df': signals_df}
