from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

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
from ....retrieval.query import TriggerVectorRetriever
from ....retrieval.trigger_engine import (
    COOLDOWN_WINDOWS,
    H_FIRE,
    K,
    K_RAW,
    MARGIN_MIN,
    MAX_WHIPSAW_RATE,
    MIN_GAP_WINDOWS,
    MIN_RESOLVE_RATE,
    P_CHOP_MAX,
    P_MIN,
    STOP_TICKS,
    EpisodeGate,
    TriggerEngine,
    TriggerThresholds,
)

OUTPUT_COLUMNS = [
    "trigger_ts",
    "session_date",
    "symbol",
    "level_id",
    "approach_dir",
    "episode_id",
    "h_fire",
    "p_break",
    "p_reject",
    "p_chop",
    "p_top1",
    "margin",
    "risk_q80_ticks",
    "resolve_rate",
    "whipsaw_rate",
    "c_top1",
    "signal",
    "fire_flag",
]


class GoldBuildMboTriggerSignals(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_mbo_trigger_signals",
            io=StageIO(
                inputs=[
                    "silver.future_mbo.mbo_level_vacuum_5s",
                    "gold.future_mbo.mbo_trigger_vectors",
                ],
                output="gold.future_mbo.mbo_trigger_signals",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        vacuum_key = "silver.future_mbo.mbo_level_vacuum_5s"
        trigger_key = "gold.future_mbo.mbo_trigger_vectors"

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

        if len(df_vacuum) == 0 or len(df_triggers) == 0:
            df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            index_dir = _load_index_dir()
            retriever = TriggerVectorRetriever(index_dir)
            df_out = _build_trigger_signals(
                df_vacuum=df_vacuum,
                df_triggers=df_triggers,
                retriever=retriever,
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


def _build_trigger_signals(
    df_vacuum: pd.DataFrame,
    df_triggers: pd.DataFrame,
    retriever: TriggerVectorRetriever,
) -> pd.DataFrame:
    df_vacuum = df_vacuum.sort_values("window_end_ts_ns").reset_index(drop=True)
    df_triggers = df_triggers.sort_values("ts_end_ns").reset_index(drop=True)

    trigger_by_ts: Dict[int, pd.Series] = {}
    for row in df_triggers.itertuples(index=False):
        ts_end = int(getattr(row, "ts_end_ns"))
        if ts_end in trigger_by_ts:
            raise ValueError(f"Duplicate trigger window: {ts_end}")
        trigger_by_ts[ts_end] = row

    engine = TriggerEngine(retriever=retriever, k=K, k_raw=K_RAW)
    thresholds = TriggerThresholds(
        p_min=P_MIN,
        margin_min=MARGIN_MIN,
        p_chop_max=P_CHOP_MAX,
        stop_ticks=STOP_TICKS,
        min_resolve_rate=MIN_RESOLVE_RATE,
        max_whipsaw_rate=MAX_WHIPSAW_RATE,
    )
    gate = EpisodeGate(
        cooldown_windows=COOLDOWN_WINDOWS,
        min_gap_windows=MIN_GAP_WINDOWS,
    )

    rows: List[Dict[str, object]] = []
    unused = set(trigger_by_ts.keys())
    for idx, vac in enumerate(df_vacuum.itertuples(index=False)):
        approach_dir = str(getattr(vac, "approach_dir"))
        ts_end = int(getattr(vac, "window_end_ts_ns"))
        episode_id, blocked = gate.step(idx, approach_dir)

        trigger = trigger_by_ts.get(ts_end)
        if trigger is None:
            continue
        unused.discard(ts_end)

        if approach_dir == "approach_none":
            raise ValueError(f"Trigger window has approach_none: {ts_end}")

        trigger_dir = str(getattr(trigger, "approach_dir"))
        if trigger_dir != approach_dir:
            raise ValueError(f"Approach dir mismatch at {ts_end}")

        vector = np.array(getattr(trigger, "vector"), dtype=np.float64)
        metrics = engine.score_vector(
            level_id=str(getattr(trigger, "level_id")),
            approach_dir=approach_dir,
            vector=vector,
        )
        decision = engine.decide(metrics, thresholds)
        fire_flag = decision.fire_flag
        signal = decision.signal
        if blocked:
            fire_flag = 0
            signal = "NONE"
        if fire_flag == 1:
            gate.register_fire(episode_id)

        rows.append(
            {
                "trigger_ts": int(ts_end),
                "session_date": str(getattr(trigger, "session_date")),
                "symbol": str(getattr(trigger, "symbol")),
                "level_id": str(getattr(trigger, "level_id")),
                "approach_dir": approach_dir,
                "episode_id": int(episode_id),
                "h_fire": int(H_FIRE),
                "p_break": float(metrics.p_break),
                "p_reject": float(metrics.p_reject),
                "p_chop": float(metrics.p_chop),
                "p_top1": float(metrics.p_top1),
                "margin": float(metrics.margin),
                "risk_q80_ticks": float(metrics.risk_q80_ticks),
                "resolve_rate": float(metrics.resolve_rate),
                "whipsaw_rate": float(metrics.whipsaw_rate),
                "c_top1": str(metrics.c_top1),
                "signal": str(signal),
                "fire_flag": int(fire_flag),
            }
        )

    if unused:
        sample = sorted(list(unused))[:5]
        raise ValueError(f"Trigger windows missing in vacuum: {sample}")

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df_out = pd.DataFrame(rows)
    return df_out.loc[:, OUTPUT_COLUMNS]


def _load_index_dir() -> Path:
    value = os.environ.get("MBO_INDEX_DIR")
    if value is None:
        raise ValueError("Missing MBO_INDEX_DIR env var")
    value = value.strip()
    if not value:
        raise ValueError("MBO_INDEX_DIR env var is empty")
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"MBO_INDEX_DIR not found: {path}")
    return path
