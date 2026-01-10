"""Gold stage: Extract setup vectors from 2-minute candle signatures.

Reads from all 4 approach2m datasets and produces a single setup_vectors dataset
containing concatenated candle signatures ready for FAISS similarity search.

Compliant with COMPLIANCE_GPT.md and CHECK.md:
- Uses 6-bar lookback ending at confirmation candle
- Passes through all audit fields (candle IDs, window boundaries, flags, config)
- Dynamically detects all bar2m_ features for inclusion in vectors
- Derivatives already computed in silver stage (d1, d2 features)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd

from ...base import Stage, StageIO
from ....io import partition_ref, is_partition_complete, read_partition, write_partition, read_manifest_hash


LOOKBACK_BARS = 6
CANDLE_POSITIONS = list(range(-(LOOKBACK_BARS - 1), 1))

METADATA_COLS = {
    "bar_ts", "symbol", "episode_id", "touch_id", "level_type", "level_price",
    "candle_id", "trigger_candle_id", "confirm_candle_id", "infer_candle_id",
    "trigger_candle_ts", "confirm_candle_ts", "infer_ts",
    "lookback_start_id", "lookback_end_id", "lookfwd_start_id", "lookfwd_end_id",
    "bar_index_in_episode", "bar_index_in_touch", "bars_to_trigger", "bars_to_confirm",
    "is_pre_trigger", "is_trigger_candle", "is_confirm_candle", "is_post_confirm",
    "is_in_lookback", "is_in_lookfwd",
    "approach_direction", "is_standard_approach",
    "outcome", "outcome_score", "max_signed_dist", "min_signed_dist",
    "chop_flag", "failed_break_flag", "both_sides_hit_flag",
    "first_hit_side", "first_hit_offset_bars", "mae_pts", "mfe_pts",
    "bar2m_open", "bar2m_high", "bar2m_low", "bar2m_close",
    "bar2m_touched_in_zone", "bar2m_close_in_zone", "bar2m_first_touch_offset",
}

CONFIG_FIELDS = [
    "cfg_bar_interval_sec",
    "cfg_lookback_bars_infer",
    "cfg_confirm_bars",
    "cfg_lookfwd_bars_label",
    "cfg_cross_epsilon_pts",
    "cfg_approach_side_min_bars",
    "cfg_reset_min_bars",
    "cfg_reset_distance_pts",
    "cfg_break_threshold_pts",
    "cfg_reject_threshold_pts",
    "cfg_outcome_price_basis",
    "cfg_confirm_close_rule",
    "cfg_confirm_close_buffer_pts",
    "cfg_chop_retrace_to_trigger_open_pts",
    "cfg_chop_override_next_close_delta_pts",
    "cfg_failed_break_confirm_close_below_level_pts",
    "cfg_stop_buffer_pts",
    "cfg_max_adverse_excursion_horizon_bars",
    "cfg_min_bars_between_touches",
    "cfg_max_touches_per_level_per_session",
]

LEVEL_TYPES = ["pm_high", "pm_low", "or_high", "or_low"]


def _detect_signature_cols(df: pd.DataFrame) -> List[str]:
    """Dynamically detect all bar2m_ feature columns."""
    sig_cols = []
    for col in sorted(df.columns):
        if col.startswith("bar2m_") and col not in METADATA_COLS:
            sig_cols.append(col)
    return sig_cols


def _build_vector_for_episode(df_episode: pd.DataFrame, signature_cols: List[str]) -> Optional[Dict]:
    """Build a setup vector from an episode's candle rows."""
    confirm_rows = df_episode[df_episode["is_confirm_candle"] == True]
    if len(confirm_rows) == 0:
        trigger_rows = df_episode[df_episode["is_trigger_candle"] == True]
        if len(trigger_rows) == 0:
            return None
        confirm = trigger_rows.iloc[0]
    else:
        confirm = confirm_rows.iloc[0]

    lookback_df = df_episode[df_episode["is_in_lookback"] == True].sort_values("bars_to_confirm")

    if len(lookback_df) == 0:
        return None

    candle_map = {}
    for _, row in lookback_df.iterrows():
        pos = int(row["bars_to_confirm"])
        candle_map[pos] = row

    vector_parts = []
    for pos in CANDLE_POSITIONS:
        if pos in candle_map:
            row = candle_map[pos]
            for col in signature_cols:
                val = float(row.get(col, 0.0))
                val = 0.0 if np.isnan(val) or np.isinf(val) else val
                vector_parts.append(val)
        else:
            vector_parts.extend([0.0] * len(signature_cols))

    result = {
        "episode_id": confirm["episode_id"],
        "level_type": confirm["level_type"],
        "level_price": float(confirm["level_price"]),
        "trigger_candle_id": int(confirm["trigger_candle_id"]),
        "confirm_candle_id": int(confirm["confirm_candle_id"]),
        "infer_candle_id": int(confirm["infer_candle_id"]),
        "trigger_candle_ts": int(confirm["trigger_candle_ts"]),
        "confirm_candle_ts": int(confirm["confirm_candle_ts"]),
        "infer_ts": int(confirm["infer_ts"]),
        "lookback_start_id": int(confirm["lookback_start_id"]),
        "lookback_end_id": int(confirm["lookback_end_id"]),
        "lookfwd_start_id": int(confirm["lookfwd_start_id"]),
        "lookfwd_end_id": int(confirm["lookfwd_end_id"]),
        "approach_direction": int(confirm["approach_direction"]),
        "is_standard_approach": bool(confirm["is_standard_approach"]),
        "outcome": confirm["outcome"],
        "outcome_score": float(confirm["outcome_score"]),
        "max_signed_dist": float(confirm["max_signed_dist"]),
        "min_signed_dist": float(confirm["min_signed_dist"]),
        "chop_flag": bool(confirm["chop_flag"]),
        "failed_break_flag": bool(confirm["failed_break_flag"]),
        "both_sides_hit_flag": bool(confirm["both_sides_hit_flag"]),
        "first_hit_side": str(confirm["first_hit_side"]),
        "first_hit_offset_bars": int(confirm["first_hit_offset_bars"]),
        "mae_pts": float(confirm["mae_pts"]),
        "mfe_pts": float(confirm["mfe_pts"]),
        "vector": vector_parts,
        "vector_dim": len(vector_parts),
    }

    for cfg_field in CONFIG_FIELDS:
        if cfg_field in confirm.index:
            val = confirm[cfg_field]
            if isinstance(val, (np.integer, int)):
                result[cfg_field] = int(val)
            elif isinstance(val, (np.floating, float)):
                result[cfg_field] = float(val)
            else:
                result[cfg_field] = str(val)

    return result


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
        signature_cols: Optional[List[str]] = None

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

            if signature_cols is None:
                signature_cols = _detect_signature_cols(df)

            episode_ids = df["episode_id"].unique()
            for episode_id in episode_ids:
                df_episode = df[df["episode_id"] == episode_id].copy()
                result = _build_vector_for_episode(df_episode, signature_cols)

                if result is None:
                    continue

                result["vector_id"] = f"{dt}_{symbol}_{level_type}_{vector_counter}"
                result["dt"] = dt
                result["symbol"] = symbol
                result["front_month_symbol"] = front_month_symbol
                result["dominance_ratio"] = dominance_ratio
                result["roll_contaminated"] = roll_contaminated
                result["is_front_month"] = is_front_month
                result["signature_cols_count"] = len(signature_cols)

                all_vectors.append(result)
                vector_counter += 1

        col_order = [
            "vector_id", "episode_id", "dt", "symbol", "level_type", "level_price",
            "trigger_candle_id", "confirm_candle_id", "infer_candle_id",
            "trigger_candle_ts", "confirm_candle_ts", "infer_ts",
            "lookback_start_id", "lookback_end_id", "lookfwd_start_id", "lookfwd_end_id",
            "approach_direction", "is_standard_approach",
            "outcome", "outcome_score", "max_signed_dist", "min_signed_dist",
            "chop_flag", "failed_break_flag", "both_sides_hit_flag",
            "first_hit_side", "first_hit_offset_bars", "mae_pts", "mfe_pts",
            "front_month_symbol", "dominance_ratio", "roll_contaminated", "is_front_month",
            "signature_cols_count",
        ] + CONFIG_FIELDS + ["vector", "vector_dim"]

        if len(all_vectors) == 0:
            df_out = pd.DataFrame(columns=col_order)
        else:
            df_out = pd.DataFrame(all_vectors)
            for col in col_order:
                if col not in df_out.columns:
                    df_out[col] = None
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
