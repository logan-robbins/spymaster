"""Gold stage: Extract setup vectors from 2-minute candle signatures.

Reads from all 4 approach2m datasets and produces a single setup_vectors dataset
containing concatenated candle signatures ready for FAISS similarity search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ...base import Stage, StageIO
from ....io import partition_ref, is_partition_complete, read_partition, write_partition, read_manifest_hash


PRE_WINDOW_CANDLES = 5
CANDLE_POSITIONS = list(range(-(PRE_WINDOW_CANDLES - 1), 1))

SIGNATURE_COLS = [
    "bar2m_time_in_zone_frac",
    "bar2m_time_far_side_frac",
    "bar2m_late_time_far_side_frac",
    "bar2m_close_side",
    "bar2m_sig_u_start",
    "bar2m_sig_u_end",
    "bar2m_sig_u_min",
    "bar2m_sig_u_max",
    "bar2m_sig_u_mean",
    "bar2m_sig_u_std",
    "bar2m_sig_u_slope",
    "bar2m_sig_u_energy",
    "bar2m_sig_u_sign_flip_cnt",
    "bar2m_sig_u_burst_frac",
    "bar2m_sig_u_mean_early",
    "bar2m_sig_u_mean_mid",
    "bar2m_sig_u_mean_late",
    "bar2m_sig_u_energy_early",
    "bar2m_sig_u_energy_late",
    "bar2m_sig_u_late_minus_early",
    "bar2m_sig_u_late_over_early",
    "bar2m_sig_pressure_start",
    "bar2m_sig_pressure_end",
    "bar2m_sig_pressure_min",
    "bar2m_sig_pressure_max",
    "bar2m_sig_pressure_mean",
    "bar2m_sig_pressure_std",
    "bar2m_sig_pressure_slope",
    "bar2m_sig_pressure_energy",
    "bar2m_sig_pressure_sign_flip_cnt",
    "bar2m_sig_pressure_burst_frac",
    "bar2m_sig_pressure_mean_early",
    "bar2m_sig_pressure_mean_mid",
    "bar2m_sig_pressure_mean_late",
    "bar2m_sig_pressure_energy_early",
    "bar2m_sig_pressure_energy_late",
    "bar2m_sig_pressure_late_minus_early",
    "bar2m_sig_pressure_late_over_early",
    "bar2m_comp_obi0_lin_mean",
    "bar2m_comp_obi10_lin_mean",
    "bar2m_comp_cdi_lin_mean",
    "bar2m_comp_flow_norm_mean",
    "bar2m_comp_trade_imbal_mean",
    "bar2m_comp_wall_support_mean",
    "bar2m_comp_wall_dist_support_mean",
    "bar2m_comp_gap_spread_mean",
    "bar2m_comp_trade_activity_mean",
]

DERIVATIVE_SIGNALS = [
    "bar2m_sig_u_mean",
    "bar2m_sig_pressure_mean",
    "bar2m_comp_obi0_lin_mean",
    "bar2m_comp_flow_norm_mean",
    "bar2m_comp_trade_imbal_mean",
    "bar2m_comp_wall_support_mean",
    "bar2m_comp_gap_spread_mean",
]

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]


def _build_vector_for_episode(df_episode: pd.DataFrame) -> Optional[Dict]:
    """Build a setup vector from an episode's candle rows."""
    trigger_rows = df_episode[df_episode["is_trigger_candle"] == True]
    if len(trigger_rows) == 0:
        return None
    trigger = trigger_rows.iloc[0]

    pre_window_df = df_episode[
        (df_episode["bars_to_trigger"] >= -(PRE_WINDOW_CANDLES - 1)) &
        (df_episode["bars_to_trigger"] <= 0)
    ].sort_values("bars_to_trigger")

    if len(pre_window_df) == 0:
        return None

    candle_map = {}
    for _, row in pre_window_df.iterrows():
        pos = int(row["bars_to_trigger"])
        candle_map[pos] = row

    vector_parts = []
    for pos in CANDLE_POSITIONS:
        if pos in candle_map:
            row = candle_map[pos]
            for col in SIGNATURE_COLS:
                val = float(row.get(col, 0.0))
                val = 0.0 if np.isnan(val) or np.isinf(val) else val
                vector_parts.append(val)
        else:
            vector_parts.extend([0.0] * len(SIGNATURE_COLS))

    for signal in DERIVATIVE_SIGNALS:
        values_by_pos = []
        for pos in CANDLE_POSITIONS:
            if pos in candle_map:
                val = float(candle_map[pos].get(signal, 0.0))
                val = 0.0 if np.isnan(val) or np.isinf(val) else val
                values_by_pos.append(val)
            else:
                values_by_pos.append(0.0)

        for i in range(1, len(values_by_pos)):
            d1 = values_by_pos[i] - values_by_pos[i - 1]
            vector_parts.append(d1)

        for i in range(2, len(values_by_pos)):
            d2 = (values_by_pos[i] - values_by_pos[i - 1]) - (values_by_pos[i - 1] - values_by_pos[i - 2])
            vector_parts.append(d2)

    return {
        "episode_id": trigger["episode_id"],
        "level_type": trigger["level_type"],
        "level_price": float(trigger["level_price"]),
        "trigger_candle_ts": int(trigger["trigger_candle_ts"]),
        "approach_direction": int(trigger["approach_direction"]),
        "outcome": trigger["outcome"],
        "outcome_score": float(trigger["outcome_score"]),
        "is_premarket_context_truncated": bool(trigger["is_premarket_context_truncated"]),
        "vector": vector_parts,
        "vector_dim": len(vector_parts),
    }


class GoldExtractSetupVectors(Stage):
    """Extract setup vectors from 2-minute candle signatures."""

    def __init__(self) -> None:
        super().__init__(
            name="gold_extract_setup_vectors",
            io=StageIO(
                inputs=[
                    "silver.future.market_by_price_10_pm_high_approach2m",
                    "silver.future.market_by_price_10_pm_low_approach2m",
                    "silver.future.market_by_price_10_or_high_approach2m",
                    "silver.future.market_by_price_10_or_low_approach2m",
                ],
                output="gold.future.setup_vectors",
            ),
        )

    def run(self, cfg, repo_root: Path, symbol: str, dt: str) -> None:
        """Run the gold vector extraction stage."""
        output_ref = partition_ref(cfg, self.io.output, symbol, dt)
        if is_partition_complete(output_ref):
            return

        from src.common.utils.contract_selector import ContractSelector

        lake_root = repo_root / "lake"
        selector = ContractSelector(bronze_root=str(lake_root))
        try:
            selection = selector.select_front_month_mbp10(dt)
            front_month_symbol = selection.front_month_symbol
            dominance_ratio = selection.dominance_ratio
            roll_contaminated = selection.roll_contaminated
        except (FileNotFoundError, ValueError):
            front_month_symbol = symbol
            dominance_ratio = 1.0
            roll_contaminated = False

        is_front_month = symbol == front_month_symbol

        all_vectors: List[Dict] = []
        vector_counter = 0

        for level_type in LEVEL_TYPES:
            dataset_key = f"silver.future.market_by_price_10_{level_type}_approach2m"
            ref = partition_ref(cfg, dataset_key, symbol, dt)

            if not is_partition_complete(ref):
                continue

            try:
                df = read_partition(ref)
            except Exception:
                continue

            if df is None or len(df) == 0:
                continue

            episode_ids = df["episode_id"].unique()
            for episode_id in episode_ids:
                df_episode = df[df["episode_id"] == episode_id].copy()
                result = _build_vector_for_episode(df_episode)

                if result is None:
                    continue

                result["vector_id"] = f"{dt}_{symbol}_{level_type}_{vector_counter}"
                result["dt"] = dt
                result["symbol"] = symbol
                result["front_month_symbol"] = front_month_symbol
                result["dominance_ratio"] = dominance_ratio
                result["roll_contaminated"] = roll_contaminated
                result["is_front_month"] = is_front_month

                all_vectors.append(result)
                vector_counter += 1

        if len(all_vectors) == 0:
            df_out = pd.DataFrame(columns=[
                "vector_id", "episode_id", "dt", "symbol", "level_type", "level_price",
                "trigger_candle_ts", "approach_direction", "outcome", "outcome_score",
                "is_premarket_context_truncated", "front_month_symbol", "dominance_ratio",
                "roll_contaminated", "is_front_month", "vector", "vector_dim"
            ])
        else:
            df_out = pd.DataFrame(all_vectors)
            col_order = [
                "vector_id", "episode_id", "dt", "symbol", "level_type", "level_price",
                "trigger_candle_ts", "approach_direction", "outcome", "outcome_score",
                "is_premarket_context_truncated", "front_month_symbol", "dominance_ratio",
                "roll_contaminated", "is_front_month", "vector", "vector_dim"
            ]
            df_out = df_out[col_order]

        contract_path = repo_root / "src" / "data_eng" / "contracts" / "gold" / "future" / "setup_vectors.avsc"

        lineage = []
        for dataset_key in self.io.inputs:
            ref = partition_ref(cfg, dataset_key, symbol, dt)
            if is_partition_complete(ref):
                try:
                    lineage.append({
                        "dataset": dataset_key,
                        "dt": dt,
                        "manifest_sha256": read_manifest_hash(ref),
                    })
                except FileNotFoundError:
                    pass

        write_partition(
            cfg=cfg,
            dataset_key=self.io.output,
            symbol=symbol,
            dt=dt,
            df=df_out,
            contract_path=contract_path,
            inputs=lineage,
            stage=self.name,
        )
