from __future__ import annotations

from pathlib import Path
from typing import List

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

GEX_BASE_FEATURES = [
    "gex_call_above_1", "gex_call_above_2", "gex_call_above_3", "gex_call_above_4", "gex_call_above_5",
    "gex_put_above_1", "gex_put_above_2", "gex_put_above_3", "gex_put_above_4", "gex_put_above_5",
    "gex_call_below_1", "gex_call_below_2", "gex_call_below_3", "gex_call_below_4", "gex_call_below_5",
    "gex_put_below_1", "gex_put_below_2", "gex_put_below_3", "gex_put_below_4", "gex_put_below_5",
    "gex_net_above", "gex_net_below", "gex_imbalance_ratio", "gex_total",
    "flow_call_add_above", "flow_call_pull_above", "flow_put_add_above", "flow_put_pull_above",
    "flow_call_add_below", "flow_call_pull_below", "flow_put_add_below", "flow_put_pull_below",
    "flow_net_above", "flow_net_below",
]

GEX_DERIVED_FEATURES: List[str] = []
for prefix in ["d1_", "d2_", "d3_"]:
    for feat in GEX_BASE_FEATURES:
        GEX_DERIVED_FEATURES.append(prefix + feat)

GEX_ALL_FEATURES = GEX_BASE_FEATURES + GEX_DERIVED_FEATURES

TRIGGER_VECTOR_COLUMNS = [
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

OUTPUT_COLUMNS = TRIGGER_VECTOR_COLUMNS + GEX_ALL_FEATURES


class GoldBuildGexEnrichedTriggerVectors(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_build_gex_enriched_trigger_vectors",
            io=StageIO(
                inputs=[
                    "gold.future_mbo.mbo_trigger_vectors",
                    "silver.future_option_mbo.gex_5s",
                ],
                output="gold.future_option_mbo.gex_enriched_trigger_vectors",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(out_ref):
            return

        trigger_key = "gold.future_mbo.mbo_trigger_vectors"
        gex_key = "silver.future_option_mbo.gex_5s"

        trigger_ref = partition_ref(cfg, trigger_key, symbol, dt)
        gex_ref = partition_ref(cfg, gex_key, symbol, dt)

        if not is_partition_complete(trigger_ref):
            raise FileNotFoundError(f"Input not ready: {trigger_key} symbol={symbol} dt={dt}")
        if not is_partition_complete(gex_ref):
            raise FileNotFoundError(f"Input not ready: {gex_key} symbol={symbol} dt={dt}")

        trigger_contract_path = repo_root / cfg.dataset(trigger_key).contract
        trigger_contract = load_avro_contract(trigger_contract_path)
        df_trigger = read_partition(trigger_ref)
        df_trigger = enforce_contract(df_trigger, trigger_contract)

        gex_contract_path = repo_root / cfg.dataset(gex_key).contract
        gex_contract = load_avro_contract(gex_contract_path)
        df_gex = read_partition(gex_ref)
        df_gex = enforce_contract(df_gex, gex_contract)

        if len(df_trigger) == 0 or len(df_gex) == 0:
            df_out = _create_empty_output()
        else:
            df_out = _join_trigger_with_gex(df_trigger, df_gex)

        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        if len(df_out) > 0:
            df_out = enforce_contract(df_out, out_contract)

        lineage = [
            {
                "dataset": trigger_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(trigger_ref),
            },
            {
                "dataset": gex_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(gex_ref),
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


def _create_empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _join_trigger_with_gex(df_trigger: pd.DataFrame, df_gex: pd.DataFrame) -> pd.DataFrame:
    df_gex_subset = df_gex[["window_end_ts_ns"] + GEX_ALL_FEATURES].copy()
    df_gex_subset = df_gex_subset.rename(columns={"window_end_ts_ns": "ts_end_ns"})

    df_merged = pd.merge(
        df_trigger,
        df_gex_subset,
        on="ts_end_ns",
        how="left",
    )

    for col in GEX_ALL_FEATURES:
        if col not in df_merged.columns:
            df_merged[col] = 0.0
        df_merged[col] = df_merged[col].fillna(0.0)

    return df_merged[OUTPUT_COLUMNS]
