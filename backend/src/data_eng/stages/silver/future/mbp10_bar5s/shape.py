from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON


def compute_shape_fractions(sizes: NDArray[np.float64]) -> NDArray[np.float64]:
    total = sizes.sum() + EPSILON
    return sizes / total

