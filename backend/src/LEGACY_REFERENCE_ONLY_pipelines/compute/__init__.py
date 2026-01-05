"""
Shared compute functions for pipeline stages.

These functions contain the core computation logic that can be used by both:
- Level-specific pipeline (with level_price parameter)
- Global market pipeline (with level_price=None)
"""

from .ofi import compute_event_ofi, compute_ofi_windows
from .kinematics import compute_kinematics_windows
from .microstructure import compute_market_microstructure
from .gex import compute_gex_features

__all__ = [
    'compute_event_ofi',
    'compute_ofi_windows',
    'compute_kinematics_windows',
    'compute_market_microstructure',
    'compute_gex_features',
]

