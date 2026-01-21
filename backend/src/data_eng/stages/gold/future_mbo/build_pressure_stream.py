from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import math
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
from ....retrieval.trigger_engine import EpisodeGate, H_FIRE
from .build_trigger_vectors import _load_level_id

WATCH_VACUUM = 0.60
ARMED_VACUUM = 0.70

OUTPUT_COLUMNS = [
    "ts_end_ns",
    "symbol",
    "session_date",
    "level_id",
    "p_ref",
    "approach_dir",
    "pressure_above_retreat",
    "pressure_above_decay_or_build",
    "pressure_above_localization",
    "pressure_above_shock",
    "pressure_above_score",
    "pressure_below_retreat_or_recede",
    "pressure_below_decay_or_build",
    "pressure_below_localization",
    "pressure_below_shock",
    "pressure_below_score",
    "vacuum_score",
    "retrieval_h_fire",
    "p_break",
    "p_reject",
    "p_chop",
    "margin",
    "risk_q80_ticks",
    "resolve_rate",
    "whipsaw_rate",
    "signal_state",
    "fire_flag",
    "signal",
    "episode_id",
]


class GoldBuildMboPressureStream(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_mbo_pressure_stream",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.mbo_level_vacuum_5s",
                    "gold.future_mbo.mbo_trigger_signals",
                ],
                output="gold.future_mbo.mbo_pressure_stream",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        vacuum_key = "silver.future_mbo.mbo_level_vacuum_5s"
        trigger_key = "gold.future_mbo.mbo_trigger_signals"

        vacuum_ref = partition_ref(cfg, vacuum_key, symbol, dt)
        trigger_ref = partition_ref(cfg, trigger_key, symbol, dt)

        if not is_partition_complete(vacuum_ref):
            raise FileNotFoundError(f"Input not ready: {vacuum_key} dt={dt}")
        if not is_partition_complete(trigger_ref):
            raise FileNotFoundError(f"Input not ready: {trigger_key} dt={dt}")

        vacuum_contract_path = repo_root / cfg.dataset(vacuum_key).contract
        vacuum_contract = load_avro_contract(vacuum_contract_path)
        df_vacuum = read_partition(vacuum_ref)
        df_vacuum = enforce_contract(df_vacuum, vacuum_contract)

        trigger_contract_path = repo_root / cfg.dataset(trigger_key).contract
        trigger_contract = load_avro_contract(trigger_contract_path)
        df_triggers = read_partition(trigger_ref)
        df_triggers = enforce_contract(df_triggers, trigger_contract)

        if len(df_vacuum) == 0:
            df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            df_out = _build_pressure_stream(
                df_vacuum=df_vacuum,
                df_triggers=df_triggers,
                symbol=symbol,
                session_date=dt,
            )

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
                "dataset": trigger_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(trigger_ref),
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


def _build_pressure_stream(
    df_vacuum: pd.DataFrame,
    df_triggers: pd.DataFrame,
    symbol: str,
    session_date: str,
) -> pd.DataFrame:
    df_vacuum = df_vacuum.sort_values("window_end_ts_ns").reset_index(drop=True)
    trigger_by_ts: Dict[int, pd.Series] = {}
    for row in df_triggers.itertuples(index=False):
        trigger_by_ts[int(getattr(row, "trigger_ts"))] = row

    level_id = _load_level_id()
    gate = EpisodeGate()

    rows: List[Dict[str, object]] = []
    for idx, vac in enumerate(df_vacuum.itertuples(index=False)):
        approach_dir = str(getattr(vac, "approach_dir"))
        ts_end = int(getattr(vac, "window_end_ts_ns"))
        p_ref = float(getattr(vac, "P_ref"))

        episode_id, _ = gate.step(idx, approach_dir)
        pressure = _pressure_scores(vac, approach_dir)

        trigger = trigger_by_ts.get(ts_end)
        if trigger is None:
            retrieval = _default_retrieval()
            fire_flag = 0
            signal = "NONE"
            margin = retrieval["margin"]
        else:
            retrieval = {
                "p_break": float(getattr(trigger, "p_break")),
                "p_reject": float(getattr(trigger, "p_reject")),
                "p_chop": float(getattr(trigger, "p_chop")),
                "margin": float(getattr(trigger, "margin")),
                "risk_q80_ticks": float(getattr(trigger, "risk_q80_ticks")),
                "resolve_rate": float(getattr(trigger, "resolve_rate")),
                "whipsaw_rate": float(getattr(trigger, "whipsaw_rate")),
            }
            fire_flag = int(getattr(trigger, "fire_flag"))
            signal = str(getattr(trigger, "signal"))
            margin = retrieval["margin"]

        if approach_dir == "approach_none":
            state = "IDLE"
            pressure = _null_pressure()
            retrieval = _null_retrieval()
            fire_flag = 0
            signal = "NONE"
        else:
            vacuum_score = pressure["vacuum_score"]
            if fire_flag == 1:
                state = "FIRE"
            elif vacuum_score is not None and vacuum_score >= ARMED_VACUUM and margin >= 0.20:
                state = "ARMED"
            elif vacuum_score is not None and vacuum_score >= WATCH_VACUUM:
                state = "WATCH"
            else:
                state = "TRACK"

        rows.append(
            {
                "ts_end_ns": int(ts_end),
                "symbol": str(symbol),
                "session_date": str(session_date),
                "level_id": str(level_id),
                "p_ref": float(p_ref),
                "approach_dir": approach_dir,
                "pressure_above_retreat": pressure["above_retreat"],
                "pressure_above_decay_or_build": pressure["above_decay"],
                "pressure_above_localization": pressure["above_local"],
                "pressure_above_shock": pressure["above_shock"],
                "pressure_above_score": pressure["above_score"],
                "pressure_below_retreat_or_recede": pressure["below_retreat"],
                "pressure_below_decay_or_build": pressure["below_decay"],
                "pressure_below_localization": pressure["below_local"],
                "pressure_below_shock": pressure["below_shock"],
                "pressure_below_score": pressure["below_score"],
                "vacuum_score": pressure["vacuum_score"],
                "retrieval_h_fire": int(H_FIRE),
                "p_break": retrieval["p_break"],
                "p_reject": retrieval["p_reject"],
                "p_chop": retrieval["p_chop"],
                "margin": retrieval["margin"],
                "risk_q80_ticks": retrieval["risk_q80_ticks"],
                "resolve_rate": retrieval["resolve_rate"],
                "whipsaw_rate": retrieval["whipsaw_rate"],
                "signal_state": str(state),
                "fire_flag": int(fire_flag),
                "signal": str(signal),
                "episode_id": int(episode_id),
            }
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df_out = pd.DataFrame(rows)
    return df_out.loc[:, OUTPUT_COLUMNS]


def _default_retrieval() -> Dict[str, float]:
    return {
        "p_break": 0.0,
        "p_reject": 0.0,
        "p_chop": 0.0,
        "margin": 0.0,
        "risk_q80_ticks": 0.0,
        "resolve_rate": 0.0,
        "whipsaw_rate": 0.0,
    }


def _null_retrieval() -> Dict[str, object]:
    return {
        "p_break": None,
        "p_reject": None,
        "p_chop": None,
        "margin": None,
        "risk_q80_ticks": None,
        "resolve_rate": None,
        "whipsaw_rate": None,
    }


def _null_pressure() -> Dict[str, object]:
    return {
        "above_retreat": None,
        "above_decay": None,
        "above_local": None,
        "above_shock": None,
        "above_score": None,
        "below_retreat": None,
        "below_decay": None,
        "below_local": None,
        "below_shock": None,
        "below_score": None,
        "vacuum_score": None,
    }


def _pressure_scores(vac: object, approach_dir: str) -> Dict[str, float | None]:
    if approach_dir == "approach_up":
        # Gate Retreat by BBO distance: u18_ask_bbo_dist_ticks
        # If bbo dist < 1.0 (tight), gate ~ 0.5 or less.
        # If bbo dist > 1.0 (open), gate -> 1.0.
        # Using sigmoid(dist - 1.5) -> sigmoid(-1.5) = 0.18, sigmoid(0.5) = 0.62?
        # Let's use sigmoid(dist - 1.0). At 0 dist -> sigmoid(-1) = 0.26. At 1 dist -> sigmoid(0) = 0.5. At 2 dist -> 0.73.
        # This penalizes tight BBO significantly.
        bbo_gate_above = _sigmoid(_safe_float(getattr(vac, "u18_ask_bbo_dist_ticks")) - 1.0)
        
        above_retreat = _sigmoid(_safe_float(getattr(vac, "u1_ask_com_disp_log"))) * bbo_gate_above
        above_decay = _sigmoid(_safe_float(getattr(vac, "u5_ask_pull_add_log_rest")))
        above_local = _sigmoid(_safe_float(getattr(vac, "u7_ask_near_pull_share_rest")) - 0.5)
        above_shock = _sigmoid(_safe_float(getattr(vac, "d2_u5_ask_pull_add_log_rest")))

        below_retreat = _sigmoid(_safe_float(getattr(vac, "u8_bid_com_approach_log")))
        below_decay = _sigmoid(_safe_float(getattr(vac, "u12_bid_add_pull_log_rest")))
        below_local = _sigmoid(_safe_float(getattr(vac, "u10_bid_near_share_rise")))
        below_shock = _sigmoid(_safe_float(getattr(vac, "d2_u12_bid_add_pull_log_rest")))
    elif approach_dir == "approach_down":
        above_retreat = _sigmoid(_safe_float(getattr(vac, "f1_ask_com_disp_log")))
        above_decay = _sigmoid(_safe_float(getattr(vac, "f2_ask_pull_add_log_rest")))
        above_local = _sigmoid(_safe_float(getattr(vac, "f2_ask_near_pull_share_rest")) - 0.5)
        above_shock = _sigmoid(_safe_float(getattr(vac, "d2_f2_ask_pull_add_log_rest")))

        # Gate Retreat by BBO distance: f9_bid_bbo_dist_ticks
        bbo_gate_below = _sigmoid(_safe_float(getattr(vac, "f9_bid_bbo_dist_ticks")) - 1.0)
        
        below_retreat = _sigmoid(_safe_float(getattr(vac, "f3_bid_com_disp_log"))) * bbo_gate_below
        below_decay = _sigmoid(_safe_float(getattr(vac, "f4_bid_pull_add_log_rest")))
        below_local = _sigmoid(_safe_float(getattr(vac, "f4_bid_near_pull_share_rest")) - 0.5)
        below_shock = _sigmoid(_safe_float(getattr(vac, "d2_f4_bid_pull_add_log_rest")))
    else:
        return _null_pressure()

    above_score = _mean4(above_retreat, above_decay, above_local, above_shock)
    below_score = _mean4(below_retreat, below_decay, below_local, below_shock)
    vacuum_score = _mean2(above_score, below_score)

    return {
        "above_retreat": above_retreat,
        "above_decay": above_decay,
        "above_local": above_local,
        "above_shock": above_shock,
        "above_score": above_score,
        "below_retreat": below_retreat,
        "below_decay": below_decay,
        "below_local": below_local,
        "below_shock": below_shock,
        "below_score": below_score,
        "vacuum_score": vacuum_score,
    }


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _safe_float(val: object) -> float:
    if val is None:
        return 0.0
    if isinstance(val, str):
        return 0.0
    try:
        out = float(val)
        if math.isnan(out):
            return 0.0
        return out
    except (TypeError, ValueError):
        return 0.0


def _mean4(a: float, b: float, c: float, d: float) -> float:
    return (a + b + c + d) / 4.0


def _mean2(a: float, b: float) -> float:
    return (a + b) / 2.0
