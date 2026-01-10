from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..io import is_partition_complete, partition_ref, read_partition
from ..stages.bronze.future_mbo.ingest_preview import BronzeIngestMboPreview
from ..stages.silver.future_mbo.compute_level_vacuum_5s import (
    BASE_FEATURES,
    DERIV_FEATURES,
    OUTPUT_COLUMNS,
    UP_DERIV_FEATURES,
    UP_FEATURES,
    TICK_INT,
    WINDOW_NS,
    compute_mbo_level_vacuum_5s,
)

P_REF = 4770.0
SYMBOL = "ESZ2"
DT = "2022-01-02"


def main() -> None:
    repo_root = Path.cwd()
    cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")

    _ensure_bronze(cfg, repo_root, SYMBOL, DT)
    bronze = _load_bronze(cfg, SYMBOL, DT)
    if len(bronze) == 0:
        raise ValueError("Preview data produced no rows")

    template = bronze.iloc[0].to_dict()
    base_ts = int(bronze["ts_event"].min())

    preview_out = compute_mbo_level_vacuum_5s(bronze, P_REF, SYMBOL)
    _assert_columns(preview_out)

    all_actions = _build_all_actions_case(template, base_ts)
    all_actions_out = compute_mbo_level_vacuum_5s(all_actions, P_REF, SYMBOL)
    _assert_columns(all_actions_out)
    _assert_derivatives_zero(all_actions_out)
    _assert_finite(all_actions_out)

    missing_orders = _build_missing_orders_case(template, base_ts)
    missing_orders_out = compute_mbo_level_vacuum_5s(missing_orders, P_REF, SYMBOL)
    _assert_columns(missing_orders_out)
    _assert_finite(missing_orders_out)

    reset_case = _build_reset_case(template, base_ts)
    reset_out = compute_mbo_level_vacuum_5s(reset_case, P_REF, SYMBOL)
    _assert_columns(reset_out)
    _assert_reset_window(reset_out, base_ts)

    noop_case = _build_noop_case(template, base_ts)
    noop_out = compute_mbo_level_vacuum_5s(noop_case, P_REF, SYMBOL)
    if len(noop_out) != 0:
        raise ValueError("No-op case should produce no rows")

    print("OK")


def _ensure_bronze(cfg, repo_root: Path, symbol: str, dt: str) -> None:
    stage = BronzeIngestMboPreview()
    stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt)

    ref = partition_ref(cfg, "bronze.future_mbo.mbo", symbol, dt)
    if not is_partition_complete(ref):
        raise ValueError("Bronze preview partition missing")


def _load_bronze(cfg, symbol: str, dt: str) -> pd.DataFrame:
    ref = partition_ref(cfg, "bronze.future_mbo.mbo", symbol, dt)
    return read_partition(ref)


def _build_all_actions_case(template: dict, base_ts: int) -> pd.DataFrame:
    p_ref_int = int(round(P_REF / 1e-9))
    rows = []
    seq = 1

    def add_event(offset_ns: int, action: str, side: str, price_int: int, size: int, order_id: int) -> None:
        nonlocal seq
        row = dict(template)
        ts_event = base_ts + offset_ns
        row.update(
            {
                "ts_event": ts_event,
                "ts_recv": ts_event,
                "action": action,
                "side": side,
                "price": price_int,
                "size": size,
                "order_id": order_id,
                "sequence": seq,
                "symbol": SYMBOL,
            }
        )
        rows.append(row)
        seq += 1

    add_event(0, "N", "N", 0, 0, 999000)
    add_event(100_000_000, "A", "A", p_ref_int + TICK_INT, 2, 1)
    add_event(700_000_000, "M", "A", p_ref_int + 10 * TICK_INT, 2, 1)
    add_event(1_300_000_000, "M", "A", p_ref_int + TICK_INT, 2, 1)
    add_event(2_000_000_000, "M", "A", p_ref_int, 2, 1)

    add_event(2_200_000_000, "A", "B", p_ref_int - TICK_INT, 3, 2)
    add_event(2_400_000_000, "F", "B", p_ref_int - TICK_INT, 1, 2)
    add_event(3_200_000_000, "C", "B", p_ref_int - TICK_INT, 0, 2)

    add_event(3_500_000_000, "A", "A", p_ref_int + 18 * TICK_INT, 1, 3)
    add_event(4_100_000_000, "M", "A", p_ref_int + 18 * TICK_INT, 3, 3)
    add_event(4_200_000_000, "T", "A", p_ref_int + 18 * TICK_INT, 0, 3)

    add_event(WINDOW_NS + 100_000_000, "A", "B", p_ref_int - 10 * TICK_INT, 4, 4)
    add_event(WINDOW_NS + 700_000_000, "M", "B", p_ref_int - 18 * TICK_INT, 4, 4)
    add_event(WINDOW_NS + 1_500_000_000, "C", "B", p_ref_int - 18 * TICK_INT, 0, 4)

    return pd.DataFrame(rows)


def _build_missing_orders_case(template: dict, base_ts: int) -> pd.DataFrame:
    rows = []
    seq = 1
    for action in ("C", "M", "F", "T"):
        row = dict(template)
        ts_event = base_ts + seq * 100_000_000
        row.update(
            {
                "ts_event": ts_event,
                "ts_recv": ts_event,
                "action": action,
                "side": "A",
                "price": int(round(P_REF / 1e-9)) + TICK_INT,
                "size": 1,
                "order_id": 999100 + seq,
                "sequence": seq,
                "symbol": SYMBOL,
            }
        )
        rows.append(row)
        seq += 1
    return pd.DataFrame(rows)


def _build_reset_case(template: dict, base_ts: int) -> pd.DataFrame:
    p_ref_int = int(round(P_REF / 1e-9))
    rows = []
    seq = 1

    def add_event(offset_ns: int, action: str, side: str, price_int: int, size: int, order_id: int) -> None:
        nonlocal seq
        row = dict(template)
        ts_event = base_ts + offset_ns
        row.update(
            {
                "ts_event": ts_event,
                "ts_recv": ts_event,
                "action": action,
                "side": side,
                "price": price_int,
                "size": size,
                "order_id": order_id,
                "sequence": seq,
                "symbol": SYMBOL,
            }
        )
        rows.append(row)
        seq += 1

    add_event(0, "R", "N", 0, 0, 0)
    add_event(200_000_000, "A", "A", p_ref_int + TICK_INT, 1, 50)
    add_event(WINDOW_NS + 100_000_000, "A", "B", p_ref_int - TICK_INT, 1, 51)

    return pd.DataFrame(rows)


def _build_noop_case(template: dict, base_ts: int) -> pd.DataFrame:
    rows = []
    for i in range(3):
        row = dict(template)
        ts_event = base_ts + i * 100_000_000
        row.update(
            {
                "ts_event": ts_event,
                "ts_recv": ts_event,
                "action": "N",
                "side": "N",
                "price": 0,
                "size": 0,
                "order_id": 800000 + i,
                "sequence": i + 1,
                "symbol": SYMBOL,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _assert_columns(df: pd.DataFrame) -> None:
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _assert_derivatives_zero(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("No rows for derivative check")
    first = df.iloc[0]
    for name in DERIV_FEATURES + UP_DERIV_FEATURES:
        if not math.isclose(float(first[name]), 0.0, abs_tol=1e-12):
            raise ValueError(f"Derivative {name} expected 0")


def _assert_finite(df: pd.DataFrame) -> None:
    if len(df) == 0:
        return
    subset = df[BASE_FEATURES + DERIV_FEATURES + UP_FEATURES + UP_DERIV_FEATURES]
    if not subset.replace([float("inf"), float("-inf")], pd.NA).notna().all().all():
        raise ValueError("Non-finite feature values")


def _assert_reset_window(df: pd.DataFrame, base_ts: int) -> None:
    if len(df) == 0:
        raise ValueError("Reset case produced no rows")
    window0_start = (base_ts // WINDOW_NS) * WINDOW_NS
    if int(df.iloc[0]["window_start_ts_ns"]) == int(window0_start):
        raise ValueError("Reset window was not skipped")


if __name__ == "__main__":
    main()
