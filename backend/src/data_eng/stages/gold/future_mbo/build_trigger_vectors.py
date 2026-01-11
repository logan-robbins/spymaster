from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...base import Stage, StageIO
from ....config import AppConfig
from ....contracts import enforce_contract, load_avro_contract
from ....io import (
    is_partition_complete,
    partition_ref,
    read_manifest_hash,
    read_partition,
    write_partition,
)
from ...silver.future_mbo.compute_level_vacuum_5s import TICK_INT

BAR_NS = 120_000_000_000
N_BARS = 6
THRESH_TICKS = 8
LOOKBACK_WINDOWS = 24
TRADE_ACTION = "T"

F_DOWN = [
    "f1_ask_com_disp_log",
    "f1_ask_slope_convex_log",
    "f1_ask_near_share_delta",
    "f1_ask_reprice_away_share_rest",
    "f2_ask_pull_add_log_rest",
    "f2_ask_pull_intensity_rest",
    "f2_ask_near_pull_share_rest",
    "f3_bid_com_disp_log",
    "f3_bid_slope_convex_log",
    "f3_bid_near_share_delta",
    "f3_bid_reprice_away_share_rest",
    "f4_bid_pull_add_log_rest",
    "f4_bid_pull_intensity_rest",
    "f4_bid_near_pull_share_rest",
    "f5_vacuum_expansion_log",
    "f6_vacuum_decay_log",
    "f7_vacuum_total_log",
]

F_UP = [
    "u1_ask_com_disp_log",
    "u2_ask_slope_convex_log",
    "u3_ask_near_share_decay",
    "u4_ask_reprice_away_share_rest",
    "u5_ask_pull_add_log_rest",
    "u6_ask_pull_intensity_rest",
    "u7_ask_near_pull_share_rest",
    "u8_bid_com_approach_log",
    "u9_bid_slope_support_log",
    "u10_bid_near_share_rise",
    "u11_bid_reprice_toward_share_rest",
    "u12_bid_add_pull_log_rest",
    "u13_bid_add_intensity",
    "u14_bid_far_pull_share_rest",
    "u15_up_expansion_log",
    "u16_up_flow_log",
    "u17_up_total_log",
]

X_COLUMNS = [
    col
    for name in F_DOWN
    for col in (name, f"d1_{name}", f"d2_{name}", f"d3_{name}")
] + [
    col
    for name in F_UP
    for col in (name, f"d1_{name}", f"d2_{name}", f"d3_{name}")
]

OUTPUT_COLUMNS = [
    "vector_id",
    "ts_end_ns",
    "trigger_bar_id",
    "trigger_candle_id",
    "horizon_end_ts_h0",
    "horizon_end_ts_h1",
    "horizon_end_ts_h2",
    "horizon_end_ts_h3",
    "horizon_end_ts_h4",
    "horizon_end_ts_h5",
    "horizon_end_ts_h6",
    "session_date",
    "symbol",
    "level_id",
    "P_ref",
    "P_REF_INT",
    "approach_dir",
    "first_hit",
    "first_hit_ts",
    "first_hit_bar_offset",
    "whipsaw_flag",
    "second_hit_ts",
    "second_hit_bar_offset",
    "true_outcome",
    "true_outcome_h0",
    "true_outcome_h1",
    "true_outcome_h2",
    "true_outcome_h3",
    "true_outcome_h4",
    "true_outcome_h5",
    "true_outcome_h6",
    "mfe_up_ticks",
    "mfe_down_ticks",
    "mae_before_upper_ticks",
    "mae_before_lower_ticks",
    "vector",
    "vector_dim",
]


@dataclass
class HitInfo:
    first_hit: str
    first_hit_ts: Optional[int]
    first_hit_bar_offset: Optional[int]
    whipsaw_flag: int
    second_hit_ts: Optional[int]
    second_hit_bar_offset: Optional[int]
    true_outcome: str
    true_outcome_h: List[str]
    horizon_end_ts: List[int]
    mfe_up_ticks: float
    mfe_down_ticks: float
    mae_before_upper_ticks: float
    mae_before_lower_ticks: float
    trigger_bar_id: int
    trigger_candle_id: int


class GoldBuildMboTriggerVectors(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_mbo_trigger_vectors",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.mbo_level_vacuum_5s",
                    "bronze.future_mbo.mbo",
                ],
                output="gold.future_mbo.mbo_trigger_vectors",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        vacuum_key = "silver.future_mbo.mbo_level_vacuum_5s"
        mbo_key = "bronze.future_mbo.mbo"

        vacuum_ref = partition_ref(cfg, vacuum_key, symbol, dt)
        mbo_ref = partition_ref(cfg, mbo_key, symbol, dt)

        if not is_partition_complete(vacuum_ref):
            raise FileNotFoundError(f"Input not ready: {vacuum_key} dt={dt}")
        if not is_partition_complete(mbo_ref):
            raise FileNotFoundError(f"Input not ready: {mbo_key} dt={dt}")

        vacuum_contract_path = repo_root / cfg.dataset(vacuum_key).contract
        vacuum_contract = load_avro_contract(vacuum_contract_path)
        df_vacuum = read_partition(vacuum_ref)
        df_vacuum = enforce_contract(df_vacuum, vacuum_contract)

        mbo_contract_path = repo_root / cfg.dataset(mbo_key).contract
        mbo_contract = load_avro_contract(mbo_contract_path)
        df_mbo = read_partition(mbo_ref)
        df_mbo = enforce_contract(df_mbo, mbo_contract)

        if len(df_vacuum) == 0 or len(df_mbo) == 0:
            df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            df_out = _build_trigger_vectors(df_vacuum, df_mbo, dt, symbol)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        if len(df_out) > 0:
            df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {
                "dataset": vacuum_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(vacuum_ref),
            },
            {
                "dataset": mbo_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(mbo_ref),
            },
        ]

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=out_contract_path,
            inputs=lineage,
            stage=self.name,
        )

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")


def _build_trigger_vectors(
    df_vacuum: pd.DataFrame,
    df_mbo: pd.DataFrame,
    dt: str,
    symbol: str,
) -> pd.DataFrame:
    level_id = _load_level_id()

    df_vacuum = df_vacuum.sort_values("window_end_ts_ns").reset_index(drop=True)
    p_ref = float(df_vacuum["P_ref"].iloc[0])
    p_ref_int = int(df_vacuum["P_REF_INT"].iloc[0])

    x_matrix = _build_x_matrix(df_vacuum)
    if x_matrix.shape[0] < LOOKBACK_WINDOWS:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    v_matrix = _build_v_matrix(x_matrix)
    window_end_ts = df_vacuum["window_end_ts_ns"].to_numpy(dtype=np.int64)
    approach_dir = df_vacuum["approach_dir"].to_numpy(dtype=object)

    eligible_mask = approach_dir[LOOKBACK_WINDOWS - 1 :] != "approach_none"
    if not np.any(eligible_mask):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    eligible_indices = np.where(eligible_mask)[0]
    v_matrix = v_matrix[eligible_indices]
    window_end_ts = window_end_ts[LOOKBACK_WINDOWS - 1 :][eligible_indices]
    approach_dir = approach_dir[LOOKBACK_WINDOWS - 1 :][eligible_indices]

    trade_ts, trade_px = _extract_trade_stream(df_mbo)
    session_start_ns = _session_start_ns(dt)
    session_start_bar_id = session_start_ns // BAR_NS

    upper_barrier_int = p_ref_int + THRESH_TICKS * TICK_INT
    lower_barrier_int = p_ref_int - THRESH_TICKS * TICK_INT

    rows: List[Dict[str, object]] = []
    for trigger_ts, approach, vector in zip(window_end_ts, approach_dir, v_matrix):
        hit_info = _label_trigger(
            trade_ts=trade_ts,
            trade_px=trade_px,
            trigger_ts=int(trigger_ts),
            approach_dir=approach,
            p_ref_int=p_ref_int,
            upper_barrier_int=upper_barrier_int,
            lower_barrier_int=lower_barrier_int,
            session_start_bar_id=session_start_bar_id,
        )

        rows.append(
            {
                "vector_id": int(trigger_ts),
                "ts_end_ns": int(trigger_ts),
                "trigger_bar_id": int(hit_info.trigger_bar_id),
                "trigger_candle_id": int(hit_info.trigger_candle_id),
                "horizon_end_ts_h0": int(hit_info.horizon_end_ts[0]),
                "horizon_end_ts_h1": int(hit_info.horizon_end_ts[1]),
                "horizon_end_ts_h2": int(hit_info.horizon_end_ts[2]),
                "horizon_end_ts_h3": int(hit_info.horizon_end_ts[3]),
                "horizon_end_ts_h4": int(hit_info.horizon_end_ts[4]),
                "horizon_end_ts_h5": int(hit_info.horizon_end_ts[5]),
                "horizon_end_ts_h6": int(hit_info.horizon_end_ts[6]),
                "session_date": dt,
                "symbol": symbol,
                "level_id": level_id,
                "P_ref": float(p_ref),
                "P_REF_INT": int(p_ref_int),
                "approach_dir": approach,
                "first_hit": hit_info.first_hit,
                "first_hit_ts": hit_info.first_hit_ts,
                "first_hit_bar_offset": hit_info.first_hit_bar_offset,
                "whipsaw_flag": int(hit_info.whipsaw_flag),
                "second_hit_ts": hit_info.second_hit_ts,
                "second_hit_bar_offset": hit_info.second_hit_bar_offset,
                "true_outcome": hit_info.true_outcome,
                "true_outcome_h0": hit_info.true_outcome_h[0],
                "true_outcome_h1": hit_info.true_outcome_h[1],
                "true_outcome_h2": hit_info.true_outcome_h[2],
                "true_outcome_h3": hit_info.true_outcome_h[3],
                "true_outcome_h4": hit_info.true_outcome_h[4],
                "true_outcome_h5": hit_info.true_outcome_h[5],
                "true_outcome_h6": hit_info.true_outcome_h[6],
                "mfe_up_ticks": float(hit_info.mfe_up_ticks),
                "mfe_down_ticks": float(hit_info.mfe_down_ticks),
                "mae_before_upper_ticks": float(hit_info.mae_before_upper_ticks),
                "mae_before_lower_ticks": float(hit_info.mae_before_lower_ticks),
                "vector": vector.tolist(),
                "vector_dim": int(vector.size),
            }
        )

    df_out = pd.DataFrame(rows)
    return df_out.loc[:, OUTPUT_COLUMNS]


def _load_level_id() -> str:
    level_id = os.environ.get("LEVEL_ID")
    if level_id is None:
        raise ValueError("Missing LEVEL_ID env var")
    level_id = level_id.strip()
    if not level_id:
        raise ValueError("LEVEL_ID env var is empty")
    return level_id


def _build_x_matrix(df_vacuum: pd.DataFrame) -> np.ndarray:
    x = df_vacuum[X_COLUMNS].to_numpy(dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _build_v_matrix(x_matrix: np.ndarray) -> np.ndarray:
    n_rows, dims = x_matrix.shape
    if n_rows < LOOKBACK_WINDOWS:
        return np.empty((0, dims * 7), dtype=np.float64)

    cumsum = np.vstack([np.zeros((1, dims)), np.cumsum(x_matrix, axis=0)])
    sum_3 = cumsum[3:] - cumsum[:-3]
    sum_9 = cumsum[9:] - cumsum[:-9]
    sum_24 = cumsum[24:] - cumsum[:-24]

    start = LOOKBACK_WINDOWS - 1
    r1 = x_matrix[start:]
    r2 = sum_3[start - 2 :] / 3.0
    r3 = (x_matrix[start:] - x_matrix[start - 2 : -2]) / 2.0
    r4 = sum_9[start - 8 :] / 9.0
    r5 = (x_matrix[start:] - x_matrix[start - 8 : -8]) / 8.0
    r6 = sum_24 / 24.0
    r7 = (x_matrix[start:] - x_matrix[start - 23 : -23]) / 23.0

    stacked = np.stack([r1, r2, r3, r4, r5, r6, r7], axis=2)
    return stacked.reshape(n_rows - start, dims * 7)


def _extract_trade_stream(df_mbo: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    trades = df_mbo.loc[df_mbo["action"] == TRADE_ACTION].copy()
    if len(trades) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    trades = trades.sort_values(["ts_event", "sequence"], ascending=[True, True])
    ts_event = trades["ts_event"].to_numpy(dtype=np.int64)
    price = trades["price"].to_numpy(dtype=np.int64)
    return ts_event, price



def _session_start_ns(dt: str) -> int:
    ts_local = pd.Timestamp(f"{dt} 08:30:00", tz="America/New_York")
    return int(ts_local.tz_convert("UTC").value)


def _label_trigger(
    trade_ts: np.ndarray,
    trade_px: np.ndarray,
    trigger_ts: int,
    approach_dir: str,
    p_ref_int: int,
    upper_barrier_int: int,
    lower_barrier_int: int,
    session_start_bar_id: int,
) -> HitInfo:
    trigger_bar_id = trigger_ts // BAR_NS
    trigger_candle_id = int(trigger_bar_id - session_start_bar_id)
    horizon_end_ts = [
        int((trigger_bar_id + h + 1) * BAR_NS) for h in range(N_BARS + 1)
    ]

    if trade_ts.size == 0:
        first_hit = "NONE"
        true_outcome = _map_outcome(approach_dir, first_hit)
        true_outcome_h = [true_outcome for _ in range(N_BARS + 1)]
        return HitInfo(
            first_hit=first_hit,
            first_hit_ts=None,
            first_hit_bar_offset=None,
            whipsaw_flag=0,
            second_hit_ts=None,
            second_hit_bar_offset=None,
            true_outcome=true_outcome,
            true_outcome_h=true_outcome_h,
            horizon_end_ts=horizon_end_ts,
            mfe_up_ticks=0.0,
            mfe_down_ticks=0.0,
            mae_before_upper_ticks=0.0,
            mae_before_lower_ticks=0.0,
            trigger_bar_id=int(trigger_bar_id),
            trigger_candle_id=trigger_candle_id,
        )

    start_idx = np.searchsorted(trade_ts, trigger_ts, side="right")
    end_idx = np.searchsorted(trade_ts, horizon_end_ts[-1], side="right")
    if start_idx >= end_idx:
        first_hit = "NONE"
        true_outcome = _map_outcome(approach_dir, first_hit)
        true_outcome_h = [true_outcome for _ in range(N_BARS + 1)]
        return HitInfo(
            first_hit=first_hit,
            first_hit_ts=None,
            first_hit_bar_offset=None,
            whipsaw_flag=0,
            second_hit_ts=None,
            second_hit_bar_offset=None,
            true_outcome=true_outcome,
            true_outcome_h=true_outcome_h,
            horizon_end_ts=horizon_end_ts,
            mfe_up_ticks=0.0,
            mfe_down_ticks=0.0,
            mae_before_upper_ticks=0.0,
            mae_before_lower_ticks=0.0,
            trigger_bar_id=int(trigger_bar_id),
            trigger_candle_id=trigger_candle_id,
        )

    slice_ts = trade_ts[start_idx:end_idx]
    slice_px = trade_px[start_idx:end_idx]

    upper_mask = slice_px >= upper_barrier_int
    lower_mask = slice_px <= lower_barrier_int

    t_hit_upper, idx_upper = _first_hit_ts(slice_ts, upper_mask)
    t_hit_lower, idx_lower = _first_hit_ts(slice_ts, lower_mask)

    first_hit, first_hit_ts = _resolve_first_hit(t_hit_upper, t_hit_lower)
    whipsaw_flag = int(t_hit_upper is not None and t_hit_lower is not None)

    first_hit_bar_offset = None
    if first_hit_ts is not None:
        first_hit_bar_offset = int((first_hit_ts // BAR_NS) - trigger_bar_id)

    second_hit_ts = None
    second_hit_bar_offset = None
    if t_hit_upper is not None and t_hit_lower is not None and t_hit_upper != t_hit_lower:
        second_hit_ts = int(max(t_hit_upper, t_hit_lower))
        second_hit_bar_offset = int((second_hit_ts // BAR_NS) - trigger_bar_id)

    true_outcome = _map_outcome(approach_dir, first_hit)
    true_outcome_h = [
        _map_outcome(
            approach_dir,
            _first_hit_by_h(t_hit_upper, t_hit_lower, horizon_end),
        )
        for horizon_end in horizon_end_ts
    ]

    min_px = int(np.min(slice_px))
    max_px = int(np.max(slice_px))
    mfe_up_ticks = max(0.0, (max_px - p_ref_int) / float(TICK_INT))
    mfe_down_ticks = max(0.0, (p_ref_int - min_px) / float(TICK_INT))

    mae_before_upper_ticks = 0.0
    mae_before_lower_ticks = 0.0

    if first_hit == "UPPER" and idx_upper is not None:
        if idx_upper > 0:
            min_before = int(np.min(slice_px[:idx_upper]))
            mae_before_upper_ticks = max(0.0, (p_ref_int - min_before) / float(TICK_INT))
    elif first_hit == "LOWER" and idx_lower is not None:
        if idx_lower > 0:
            max_before = int(np.max(slice_px[:idx_lower]))
            mae_before_lower_ticks = max(0.0, (max_before - p_ref_int) / float(TICK_INT))
    else:
        mae_before_upper_ticks = max(0.0, (p_ref_int - min_px) / float(TICK_INT))
        mae_before_lower_ticks = max(0.0, (max_px - p_ref_int) / float(TICK_INT))

    return HitInfo(
        first_hit=first_hit,
        first_hit_ts=first_hit_ts,
        first_hit_bar_offset=first_hit_bar_offset,
        whipsaw_flag=whipsaw_flag,
        second_hit_ts=second_hit_ts,
        second_hit_bar_offset=second_hit_bar_offset,
        true_outcome=true_outcome,
        true_outcome_h=true_outcome_h,
        horizon_end_ts=horizon_end_ts,
        mfe_up_ticks=mfe_up_ticks,
        mfe_down_ticks=mfe_down_ticks,
        mae_before_upper_ticks=mae_before_upper_ticks,
        mae_before_lower_ticks=mae_before_lower_ticks,
        trigger_bar_id=int(trigger_bar_id),
        trigger_candle_id=trigger_candle_id,
    )


def _first_hit_ts(
    slice_ts: np.ndarray,
    hit_mask: np.ndarray,
) -> tuple[Optional[int], Optional[int]]:
    if not np.any(hit_mask):
        return None, None
    idx = int(np.argmax(hit_mask))
    return int(slice_ts[idx]), idx


def _resolve_first_hit(
    t_hit_upper: Optional[int],
    t_hit_lower: Optional[int],
) -> tuple[str, Optional[int]]:
    if t_hit_upper is None and t_hit_lower is None:
        return "NONE", None
    if t_hit_upper is not None and t_hit_lower is None:
        return "UPPER", t_hit_upper
    if t_hit_upper is None and t_hit_lower is not None:
        return "LOWER", t_hit_lower
    if t_hit_upper == t_hit_lower:
        return "WHIPSAW", t_hit_upper
    if t_hit_upper < t_hit_lower:
        return "UPPER", t_hit_upper
    return "LOWER", t_hit_lower


def _first_hit_by_h(
    t_hit_upper: Optional[int],
    t_hit_lower: Optional[int],
    horizon_end_ts: int,
) -> str:
    upper_hit = t_hit_upper is not None and t_hit_upper <= horizon_end_ts
    lower_hit = t_hit_lower is not None and t_hit_lower <= horizon_end_ts

    if not upper_hit and not lower_hit:
        return "NONE"
    if upper_hit and not lower_hit:
        return "UPPER"
    if lower_hit and not upper_hit:
        return "LOWER"
    if t_hit_upper == t_hit_lower:
        return "WHIPSAW"
    if t_hit_upper < t_hit_lower:
        return "UPPER"
    return "LOWER"


def _map_outcome(approach_dir: str, first_hit: str) -> str:
    if approach_dir == "approach_up":
        if first_hit == "UPPER":
            return "BREAK_UP"
        if first_hit == "LOWER":
            return "REJECT_DOWN"
        if first_hit == "WHIPSAW":
            return "WHIPSAW"
        return "CHOP"
    if approach_dir == "approach_down":
        if first_hit == "LOWER":
            return "BREAK_DOWN"
        if first_hit == "UPPER":
            return "REJECT_UP"
        if first_hit == "WHIPSAW":
            return "WHIPSAW"
        return "CHOP"
    raise ValueError(f"Unexpected approach_dir: {approach_dir}")
