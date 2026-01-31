from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.serving.forecast_calibration import confidence_calibration, map_coeffs_to_params


def test_map_coeffs_to_params_clipping() -> None:
    beta, gamma = map_coeffs_to_params(a=-0.5, b=-1.0)
    assert beta == 0.0
    assert gamma == 1.0

    beta, gamma = map_coeffs_to_params(a=0.5, b=2.0)
    assert beta == 2.0
    assert gamma == 0.5

    beta, gamma = map_coeffs_to_params(a=2.0, b=1.0)
    assert beta == 1.0
    assert gamma == 0.0


def test_confidence_calibration_monotonic() -> None:
    confidences = [0.05, 0.10, 0.20, 0.40]
    hits = [0, 0, 1, 1]
    result = confidence_calibration(confidences, hits, n_bins=2)
    assert result["monotonic"] is True
