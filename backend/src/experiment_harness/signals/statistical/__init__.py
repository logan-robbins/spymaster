"""Statistical (pure-arithmetic) signal implementations.

Importing this package triggers registration of all statistical signal
classes into the global SIGNAL_REGISTRY via register_signal() calls at
module level in each signal module.
"""
from __future__ import annotations

from src.experiment_harness.signals.statistical import (  # noqa: F401
    ads,
    ads_pfp_svac,
    erd,
    iirc,
    jad,
    msd,
    perm_derivative,
    pfp,
    spg,
)
