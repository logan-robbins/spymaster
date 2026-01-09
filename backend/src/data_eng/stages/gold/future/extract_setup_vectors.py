from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

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
from src.common.utils.contract_selector import ContractSelector

EPSILON = 1e-9
LEVEL_TYPES = ["PM_HIGH", "PM_LOW", "OR_HIGH", "OR_LOW"]
TARGET_DIM = 256
PRE_WINDOW_CANDLES = 5
DERIV_WINDOWS = [1, 2, 3, 5]

CANDLE_FEATURES = [
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
    "bar2m_time_in_zone_frac",
    "bar2m_time_far_side_frac",
    "bar2m_late_time_far_side_frac",
    "bar2m_close_in_zone",
    "bar2m_close_side",
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

CANDLE_FEATURE_DIM = len(CANDLE_FEATURES)

META_FIELDS = [
    "vector_id",
    "episode_id",
    "dt",
    "symbol",
    "level_type",
    "level_price",
    "trigger_candle_ts",
    "approach_direction",
    "outcome",
    "outcome_score",
    "velocity_at_trigger",
    "obi0_at_trigger",
    "wall_imbal_at_trigger",
    "front_month_symbol",
    "dominance_ratio",
    "roll_contaminated",
    "runner_up_symbol",
    "runner_up_ratio",
    "is_front_month",
]

FORBIDDEN_COLUMNS = {
    "bar5s_microprice_eob",
    "bar5s_midprice_eob",
    "bar5s_trade_last_px",
    "bar5s_trade_vol_sum",
    "bar5s_trade_signed_vol_sum",
    "bar5s_depth_bid10_qty_eob",
    "bar5s_depth_ask10_qty_eob",
    "level_price",
}


def _pad_sequence(values: np.ndarray, size: int) -> np.ndarray:
    if len(values) >= size:
        return values[-size:]
    pad = np.zeros(size - len(values), dtype=np.float64)
    return np.concatenate([pad, values])


def _window_delta(values: np.ndarray, window: int) -> float:
    if len(values) == 0:
        return 0.0
    if window >= len(values):
        delta = float((values[-1] - values[0]) / float(window))
    else:
        delta = float((values[-1] - values[-(window + 1)]) / float(window))
    return float(np.sign(delta) * np.log1p(abs(delta)))


def _time_bucket_norm(bar_ts: int, dt: str) -> float:
    tz = ZoneInfo("America/New_York")
    ts_local = pd.to_datetime(bar_ts, unit="ns", utc=True).tz_convert(tz)
    start = pd.Timestamp(dt, tz=tz).replace(hour=9, minute=30, second=0, microsecond=0)
    minutes = (ts_local - start).total_seconds() / 60.0
    bucket = int(np.floor(minutes / 2.0))
    bucket = int(np.clip(bucket, 0, 89))
    return float(bucket / 89.0)


def _candle_vector(df_window: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in df_window.iterrows():
        vals = row[CANDLE_FEATURES].to_numpy(dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(vals)
    if not rows:
        return np.zeros(PRE_WINDOW_CANDLES * CANDLE_FEATURE_DIM, dtype=np.float64)

    seq = np.vstack(rows)
    if seq.shape[0] < PRE_WINDOW_CANDLES:
        pad = np.zeros((PRE_WINDOW_CANDLES - seq.shape[0], CANDLE_FEATURE_DIM), dtype=np.float64)
        seq = np.vstack([pad, seq])
    return seq.flatten()


def _derivative_features(df_window: pd.DataFrame) -> List[float]:
    u_mean = _pad_sequence(df_window["bar2m_sig_u_mean"].to_numpy(dtype=np.float64), PRE_WINDOW_CANDLES)
    p_mean = _pad_sequence(df_window["bar2m_sig_pressure_mean"].to_numpy(dtype=np.float64), PRE_WINDOW_CANDLES)

    feats: List[float] = []
    for w in DERIV_WINDOWS:
        feats.append(_window_delta(u_mean, w))
    for w in DERIV_WINDOWS:
        feats.append(_window_delta(p_mean, w))
    return feats


def _context_features(trigger_row: pd.Series, level_type: str, dt: str) -> List[float]:
    level_onehot = [1.0 if level_type == lt else 0.0 for lt in LEVEL_TYPES]
    approach_direction = float(trigger_row.get("approach_direction", 0.0))
    time_bucket = _time_bucket_norm(int(trigger_row.get("bar_ts", 0)), dt)
    time_in_zone = float(trigger_row.get("bar2m_time_in_zone_frac", 0.0) or 0.0)
    time_far_side = float(trigger_row.get("bar2m_time_far_side_frac", 0.0) or 0.0)
    return level_onehot + [approach_direction, time_bucket, time_in_zone, time_far_side]


def _vector_from_window(df_window: pd.DataFrame, trigger_row: pd.Series, level_type: str, dt: str) -> np.ndarray:
    candle_vec = _candle_vector(df_window)
    deriv_vec = np.array(_derivative_features(df_window), dtype=np.float64)
    context_vec = np.array(_context_features(trigger_row, level_type, dt), dtype=np.float64)
    vector = np.concatenate([candle_vec, deriv_vec, context_vec])
    if vector.shape[0] != TARGET_DIM:
        raise ValueError(f"Vector dim {vector.shape[0]} != {TARGET_DIM}")
    return vector


class GoldExtractSetupVectors(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="gold_extract_setup_vectors",
            io=StageIO(
                inputs=[
                    f"silver.future.market_by_price_10_{lt.lower()}_approach2m"
                    for lt in LEVEL_TYPES
                ],
                output="gold.future.setup_vectors",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        output_key = self.io.output
        out_ref = partition_ref(cfg, output_key, symbol, dt)
        if is_partition_complete(out_ref):
            return

        bronze_root = cfg.lake_root / "bronze"
        selector = ContractSelector(bronze_root=str(bronze_root))
        selection = selector.select_front_month_mbp10(date=dt, metric="trade_count")

        if selection.roll_contaminated or symbol != selection.front_month_symbol:
            df_out = self._empty_output(df_has_selection=False)
            self._write_output(cfg, repo_root, dt, symbol, df_out, [])
            return

        all_vectors: List[np.ndarray] = []
        all_meta: List[Dict[str, Any]] = []
        lineage = []

        for level_type in LEVEL_TYPES:
            input_key = f"silver.future.market_by_price_10_{level_type.lower()}_approach2m"
            in_ref = partition_ref(cfg, input_key, symbol, dt)
            if not is_partition_complete(in_ref):
                continue

            in_contract_path = repo_root / cfg.dataset(input_key).contract
            in_contract = load_avro_contract(in_contract_path)
            df_in = read_partition(in_ref)
            if len(df_in) == 0:
                continue

            df_in = enforce_contract(df_in, in_contract)
            vectors, meta_rows = self._extract_vectors(df_in, level_type, dt, symbol)
            all_vectors.extend(vectors)
            all_meta.extend(meta_rows)

            lineage.append({
                "dataset": in_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(in_ref),
            })

        if not all_vectors:
            df_out = self._empty_output(df_has_selection=True)
        else:
            vectors_array = np.vstack(all_vectors)
            df_meta = pd.DataFrame(all_meta)
            df_meta["vector_id"] = range(len(df_meta))
            df_meta["dt"] = dt
            df_meta["front_month_symbol"] = selection.front_month_symbol
            df_meta["dominance_ratio"] = selection.dominance_ratio
            df_meta["roll_contaminated"] = selection.roll_contaminated
            df_meta["runner_up_symbol"] = selection.runner_up_symbol or ""
            df_meta["runner_up_ratio"] = selection.runner_up_ratio or 0.0
            df_meta["is_front_month"] = symbol == selection.front_month_symbol

            vector_df = pd.DataFrame(vectors_array, columns=[f"v_{i}" for i in range(TARGET_DIM)])
            df_out = pd.concat([df_meta, vector_df], axis=1)

        self._write_output(cfg, repo_root, dt, symbol, df_out, lineage)

    def transform(self, df: pd.DataFrame, dt: str) -> pd.DataFrame:
        raise NotImplementedError("Use run() directly")

    def _empty_output(self, df_has_selection: bool) -> pd.DataFrame:
        cols = META_FIELDS + [f"v_{i}" for i in range(TARGET_DIM)]
        df_out = pd.DataFrame(columns=cols)
        if df_has_selection:
            return df_out
        return df_out

    def _write_output(
        self,
        cfg: AppConfig,
        repo_root: Path,
        dt: str,
        symbol: str,
        df_out: pd.DataFrame,
        lineage: List[Dict[str, Any]],
    ) -> None:
        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)
        if len(df_out) > 0:
            df_out = enforce_contract(df_out, out_contract)

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

    def _extract_vectors(
        self,
        df: pd.DataFrame,
        level_type: str,
        dt: str,
        symbol: str,
    ) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        vectors: List[np.ndarray] = []
        metadata: List[Dict[str, Any]] = []

        df = df.sort_values(["touch_id", "bar_ts"]).reset_index(drop=True)

        for touch_id in df["touch_id"].unique():
            touch_df = df[df["touch_id"] == touch_id].copy()
            trigger_mask = touch_df["is_trigger_candle"] == True
            if not trigger_mask.any():
                continue

            trigger_row = touch_df[trigger_mask].iloc[0]

            window_mask = touch_df["bars_to_trigger"].between(-(PRE_WINDOW_CANDLES - 1), 0)
            window_df = touch_df[window_mask].sort_values("bars_to_trigger")

            for col in CANDLE_FEATURES:
                if col in FORBIDDEN_COLUMNS:
                    raise ValueError(f"Forbidden feature in vector: {col}")
                if col not in window_df.columns:
                    raise ValueError(f"Missing vector feature: {col}")

            vector = _vector_from_window(window_df, trigger_row, level_type, dt)
            vectors.append(vector)

            meta_row = {
                "episode_id": str(trigger_row.get("episode_id", "")),
                "symbol": symbol,
                "level_type": level_type,
                "level_price": float(trigger_row.get("level_price", 0.0)),
                "trigger_candle_ts": int(trigger_row.get("trigger_candle_ts", 0)),
                "approach_direction": int(trigger_row.get("approach_direction", 0)),
                "outcome": str(trigger_row.get("outcome", "CHOP")),
                "outcome_score": float(trigger_row.get("outcome_score", 0.0)),
                "velocity_at_trigger": float(trigger_row.get("bar2m_sig_u_slope", 0.0) or 0.0),
                "obi0_at_trigger": float(trigger_row.get("bar2m_comp_obi0_lin_mean", 0.0) or 0.0),
                "wall_imbal_at_trigger": float(trigger_row.get("bar2m_comp_wall_support_mean", 0.0) or 0.0),
            }
            metadata.append(meta_row)

        return vectors, metadata
