from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.experiment_harness.dataset_registry import DatasetRegistry
from src.experiment_harness.eval_engine import EvalEngine


def _write_dataset(
    root: Path,
    dataset_id: str,
    *,
    include_support: bool,
) -> None:
    dataset_dir = root / "research" / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    n_bins = 3
    k_vals = np.arange(-50, 51, dtype=np.int32)

    bins_df = pd.DataFrame(
        {
            "bin_seq": [0, 1, 2],
            "ts_ns": [1_000, 2_000, 3_000],
            "mid_price": [100.0, 100.25, 100.5],
        }
    )
    bins_df.to_parquet(dataset_dir / "bins.parquet", index=False)

    rows: list[dict[str, float | int]] = []
    for bin_seq in range(n_bins):
        for k in k_vals:
            row: dict[str, float | int] = {"bin_seq": bin_seq, "k": int(k)}
            if include_support:
                row["composite_d1"] = float(np.sin((bin_seq + 1) * k / 50.0))
                row["composite_d2"] = float(np.cos((bin_seq + 1) * k / 60.0))
                row["composite_d3"] = float(np.sin((bin_seq + 1) * k / 70.0))
            rows.append(row)

    grid_df = pd.DataFrame(rows)
    grid_df.to_parquet(dataset_dir / "grid_clean.parquet", index=False)
    (dataset_dir / "manifest.json").write_text("{}", encoding="utf-8")


def test_load_dataset_derives_flow_columns_when_absent(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "ds_flow_derive", include_support=True)

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    result = engine.load_dataset(
        "ds_flow_derive",
        ["flow_score", "flow_state_code"],
        registry,
    )

    score = result["flow_score"]
    state = result["flow_state_code"]
    assert score.shape == (3, 101)
    assert state.shape == (3, 101)
    assert np.all(np.isfinite(score))
    assert np.all((-1 <= state) & (state <= 1))


def test_load_dataset_fails_when_flow_columns_missing_and_no_support(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "ds_flow_fail", include_support=False)

    registry = DatasetRegistry(tmp_path)
    engine = EvalEngine()
    with pytest.raises(KeyError, match="missing columns needed to derive flow fields"):
        engine.load_dataset(
            "ds_flow_fail",
            ["flow_score"],
            registry,
        )
