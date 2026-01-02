"""
Compute F=ma consistency checks (physics validation features).

Cross-validate force and mass proxies
with observed acceleration.

F = ma analogy:
- Mass: Liquidity depth (barrier resistance)
- Force: Order flow imbalance + tape pressure
- Acceleration: Price movement (kinematics)

Residual = actual_accel - predicted_accel reveals:
- Positive residual: "Hidden momentum" (price moved more than liquidity suggests)
- Negative residual: "Absorption" (liquidity absorbed the pressure)
"""

from typing import Any, Dict, List
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext


def compute_force_mass_features(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute F=ma consistency features.
    
    Required inputs (from prior stages):
    - acceleration: From kinematics stage
    - integrated_ofi: From OFI stage (force proxy)
    - barrier_delta_liq: From physics stage (mass proxy, defending liquidity)
    - tape_imbalance: From physics stage (additional force)
    
    Outputs:
    - predicted_accel: Force / Mass (scaled)
    - accel_residual: actual_accel - predicted_accel
    - force_mass_ratio: Diagnostic ratio
    
    Args:
        signals_df: DataFrame with physics features
    
    Returns:
        DataFrame with F=ma features added
    """
    if signals_df.empty:
        result = signals_df.copy()
        result['predicted_accel'] = 0.0
        result['accel_residual'] = 0.0
        result['force_mass_ratio'] = 0.0
        return result
    
    # Determine acceleration column (single-window or multi-window)
    if 'acceleration' in signals_df.columns:
        accel_col = 'acceleration'
    elif 'acceleration_1min' in signals_df.columns:
        accel_col = 'acceleration_1min'
    else:
        raise ValueError("signals_df missing acceleration column (acceleration_1min)")
    
    # Build force proxy (combination of OFI + tape)
    force = np.zeros(len(signals_df), dtype=np.float64)
    
    if 'ofi_60s' in signals_df.columns:
        force += signals_df['ofi_60s'].fillna(0).values
    
    if 'tape_imbalance' in signals_df.columns:
        # Scale tape imbalance to consistent units
        force += signals_df['tape_imbalance'].fillna(0).values * 10.0
    
    # Build mass proxy (liquidity depth)
    mass = np.ones(len(signals_df), dtype=np.float64)  # Default mass = 1
    
    if 'barrier_delta_liq' in signals_df.columns:
        # Absolute value of delta_liq represents liquidity magnitude
        mass = np.abs(signals_df['barrier_delta_liq'].fillna(100).values)
        mass = np.maximum(mass, 1.0)  # Avoid division by zero
    
    # Compute predicted acceleration
    # predicted_accel = Force / Mass (with scaling to match units)
    scale_factor = 0.01  # Tune this to match acceleration units
    predicted_accel = (force / mass) * scale_factor
    
    # Actual acceleration
    actual_accel = signals_df[accel_col].fillna(0).values
    
    # Residual (unexplained acceleration)
    accel_residual = actual_accel - predicted_accel
    
    # Force/Mass ratio (diagnostic)
    force_mass_ratio = force / mass
    
    result = signals_df.copy()
    result['predicted_accel'] = predicted_accel
    result['accel_residual'] = accel_residual
    result['force_mass_ratio'] = force_mass_ratio
    result['force_proxy'] = force
    result['mass_proxy'] = mass
    
    # Flow alignment: OFI aligned with approach direction
    # Per EPISODE_VECTOR_SCHEMA.md Section D, index 110
    # Positive when OFI matches the approach direction, negative when opposing
    if 'ofi_60s' in signals_df.columns and 'direction' in signals_df.columns:
        ofi_60s = signals_df['ofi_60s'].fillna(0).values
        # Direction sign: UP = 1 (approaching from below), DOWN = -1 (approaching from above)
        direction_sign = np.where(signals_df['direction'] == 'UP', 1, -1)
        result['flow_alignment'] = ofi_60s * direction_sign
    elif 'ofi_60s' in signals_df.columns and 'distance_signed' in signals_df.columns:
        # Alternative: use signed distance if direction not available
        ofi_60s = signals_df['ofi_60s'].fillna(0).values
        distance_sign = np.sign(signals_df['distance_signed'].fillna(0).values)
        result['flow_alignment'] = ofi_60s * distance_sign
    else:
        result['flow_alignment'] = 0.0
    
    return result


class ComputeForceMassStage(BaseStage):
    """
    Compute F=ma physics validation features.
    
    Cross-validate force/mass with observed kinematics.
    
    Outputs:
        signals_df: Updated with F=ma features
    """
    
    @property
    def name(self) -> str:
        return "compute_force_mass"
    
    @property
    def required_inputs(self) -> List[str]:
        return ['signals_df']
    
    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        signals_df = ctx.data['signals_df']
        
        signals_df = compute_force_mass_features(signals_df)
        
        return {'signals_df': signals_df}
