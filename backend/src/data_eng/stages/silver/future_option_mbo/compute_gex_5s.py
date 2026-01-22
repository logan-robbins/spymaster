from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.stats import norm

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

PRICE_SCALE = 1e-9
WINDOW_NS = 5_000_000_000
REST_NS = 500_000_000
EPS_QTY = 1.0
EPS_GEX = 1e-6
PULL_ACTIONS = {"C", "M"}
TRADE_ACTION = "T"
CONTRACT_MULTIPLIER = 50
NUM_STRIKES_EACH_SIDE = 5
STRIKE_STEP = 5
MAX_DTE_DAYS = 90
RISK_FREE_RATE = 0.05
DEFAULT_IV = 0.20

BASE_GEX_FEATURES = [
    "gex_call_above_1",
    "gex_call_above_2",
    "gex_call_above_3",
    "gex_call_above_4",
    "gex_call_above_5",
    "gex_put_above_1",
    "gex_put_above_2",
    "gex_put_above_3",
    "gex_put_above_4",
    "gex_put_above_5",
    "gex_call_below_1",
    "gex_call_below_2",
    "gex_call_below_3",
    "gex_call_below_4",
    "gex_call_below_5",
    "gex_put_below_1",
    "gex_put_below_2",
    "gex_put_below_3",
    "gex_put_below_4",
    "gex_put_below_5",
    "gex_net_above",
    "gex_net_below",
    "gex_imbalance_ratio",
    "gex_total",
]

FLOW_FEATURES = [
    "flow_call_add_above",
    "flow_call_pull_above",
    "flow_put_add_above",
    "flow_put_pull_above",
    "flow_call_add_below",
    "flow_call_pull_below",
    "flow_put_add_below",
    "flow_put_pull_below",
    "flow_net_above",
    "flow_net_below",
]

BASE_FEATURES = BASE_GEX_FEATURES + FLOW_FEATURES

DERIV_FEATURES = (
    [f"d1_{name}" for name in BASE_FEATURES]
    + [f"d2_{name}" for name in BASE_FEATURES]
    + [f"d3_{name}" for name in BASE_FEATURES]
)

OUTPUT_COLUMNS = [
    "window_start_ts_ns",
    "window_end_ts_ns",
    "ref_price",
    "strike_above_1",
    "strike_above_2",
    "strike_above_3",
    "strike_above_4",
    "strike_above_5",
    "strike_below_1",
    "strike_below_2",
    "strike_below_3",
    "strike_below_4",
    "strike_below_5",
] + BASE_FEATURES + DERIV_FEATURES


@dataclass
class OptionOrderState:
    side: str
    price_int: int
    qty: int
    strike: int
    right: str
    bucket_enter_ts: int
    expiration: str


class SilverComputeGex5s(Stage):
    def __init__(self) -> None:
        super().__init__(
            name="silver_compute_gex_5s",
            io=StageIO(
                inputs=["bronze.future_option_mbo.mbo"],
                output="silver.future_option_mbo.gex_5s",
            ),
        )

    def run(self, cfg: AppConfig, repo_root: Path, symbol: str, dt: str) -> None:
        out_ref = partition_ref(cfg, self.io.output, symbol, dt)
        out_contract_path = repo_root / cfg.dataset(self.io.output).contract
        out_contract = load_avro_contract(out_contract_path)

        if is_partition_complete(out_ref):
            try:
                existing = read_partition(out_ref)
                enforce_contract(existing, out_contract)
                return
            except Exception:
                pass

        input_key = self.io.inputs[0]
        in_ref = partition_ref(cfg, input_key, symbol, dt)
        if not is_partition_complete(in_ref):
            raise FileNotFoundError(f"Input not ready: {input_key} dt={dt}")

        in_contract_path = repo_root / cfg.dataset(input_key).contract
        in_contract = load_avro_contract(in_contract_path)
        df_in = read_partition(in_ref)
        df_in = enforce_contract(df_in, in_contract)

        df_out = self.transform(df_in, dt)
        df_out = enforce_contract(df_out, out_contract)

        lineage: List[Dict[str, Any]] = [
            {
                "dataset": in_ref.dataset_key,
                "dt": dt,
                "manifest_sha256": read_manifest_hash(in_ref),
            }
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
        if len(df) == 0:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        df = _filter_by_dte(df, dt)
        if len(df) == 0:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        return compute_gex_5s(df, dt)


def _filter_by_dte(df: pd.DataFrame, dt: str) -> pd.DataFrame:
    session_date = datetime.strptime(dt, "%Y-%m-%d").date()

    def calc_dte(exp_str: str) -> int:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            return (exp_date - session_date).days
        except (ValueError, TypeError):
            return 999

    df = df.copy()
    df["_dte"] = df["expiration"].apply(calc_dte)
    df = df.loc[(df["_dte"] > 0) & (df["_dte"] <= MAX_DTE_DAYS)]
    df = df.drop(columns=["_dte"])
    return df


def _compute_dte_map(df: pd.DataFrame, dt: str) -> Dict[str, float]:
    session_date = datetime.strptime(dt, "%Y-%m-%d").date()
    dte_map = {}
    for exp_str in df["expiration"].unique():
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte_days = (exp_date - session_date).days
            dte_map[exp_str] = max(dte_days / 365.0, 1.0 / 365.0)
        except (ValueError, TypeError):
            dte_map[exp_str] = 30.0 / 365.0
    return dte_map


def compute_gex_5s(df: pd.DataFrame, dt: str = "2026-01-06") -> pd.DataFrame:
    required_cols = {
        "ts_event",
        "action",
        "side",
        "price",
        "size",
        "order_id",
        "sequence",
        "strike",
        "right",
        "expiration",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[df["action"] != "N"].copy()
    if len(df) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    dte_map = _compute_dte_map(df, dt)

    df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])
    df["ts_event"] = df["ts_event"].astype("int64")
    df["sequence"] = df["sequence"].astype("int64")
    df["size"] = df["size"].astype("int64")
    df["order_id"] = df["order_id"].astype("int64")
    df["price"] = df["price"].astype("int64")
    df["strike"] = df["strike"].astype("int64")

    orders: dict[int, OptionOrderState] = {}
    strike_depth: dict[tuple[int, str, str], float] = {}
    accum = _new_accumulators()
    rows = []
    current_window_id = None
    window_invalid = False
    window_start_ts_ns = None
    window_end_ts_ns = None

    all_strikes: Set[int] = set(df["strike"].unique())
    sorted_strikes = sorted(all_strikes)

    for row in df.itertuples(index=False):
        ts_event = int(row.ts_event)
        window_id = ts_event // WINDOW_NS

        if current_window_id is None:
            current_window_id = window_id
            window_invalid = False
            window_start_ts_ns = current_window_id * WINDOW_NS
            window_end_ts_ns = window_start_ts_ns + WINDOW_NS
        elif window_id != current_window_id:
            if not window_invalid:
                ref_price = _compute_ref_price(strike_depth, sorted_strikes)
                strikes_above, strikes_below = _get_strike_buckets(
                    ref_price, sorted_strikes
                )
                gex_features = _compute_gex_features(
                    strike_depth, ref_price, strikes_above, strikes_below, dte_map
                )
                flow_features = _compute_flow_features(accum, strikes_above, strikes_below)
                rows.append(
                    {
                        "window_start_ts_ns": int(window_start_ts_ns),
                        "window_end_ts_ns": int(window_end_ts_ns),
                        "ref_price": float(ref_price),
                        "strike_above_1": int(strikes_above[0]) if len(strikes_above) > 0 else 0,
                        "strike_above_2": int(strikes_above[1]) if len(strikes_above) > 1 else 0,
                        "strike_above_3": int(strikes_above[2]) if len(strikes_above) > 2 else 0,
                        "strike_above_4": int(strikes_above[3]) if len(strikes_above) > 3 else 0,
                        "strike_above_5": int(strikes_above[4]) if len(strikes_above) > 4 else 0,
                        "strike_below_1": int(strikes_below[0]) if len(strikes_below) > 0 else 0,
                        "strike_below_2": int(strikes_below[1]) if len(strikes_below) > 1 else 0,
                        "strike_below_3": int(strikes_below[2]) if len(strikes_below) > 2 else 0,
                        "strike_below_4": int(strikes_below[3]) if len(strikes_below) > 3 else 0,
                        "strike_below_5": int(strikes_below[4]) if len(strikes_below) > 4 else 0,
                        **gex_features,
                        **flow_features,
                    }
                )
            current_window_id = window_id
            window_invalid = False
            accum = _new_accumulators()
            window_start_ts_ns = current_window_id * WINDOW_NS
            window_end_ts_ns = window_start_ts_ns + WINDOW_NS

        action = row.action
        if action == "R":
            orders.clear()
            strike_depth.clear()
            accum = _new_accumulators()
            window_invalid = True
            continue

        order_id = int(row.order_id)
        strike = int(row.strike)
        right = str(row.right)
        expiration = str(row.expiration)
        old = orders.get(order_id)

        if old is None and action in {"C", "M", "F", "T"}:
            continue

        if old is not None:
            old_side = old.side
            old_qty = old.qty
            old_strike = old.strike
            old_right = old.right
            old_bucket_enter_ts = old.bucket_enter_ts
            old_expiration = old.expiration
        else:
            old_side = ""
            old_qty = 0
            old_strike = 0
            old_right = ""
            old_bucket_enter_ts = 0
            old_expiration = ""

        new_order = None
        new_side = old_side
        new_qty = old_qty

        if action == "A":
            new_side = row.side
            new_qty = int(row.size)
            new_order = OptionOrderState(
                side=new_side,
                price_int=int(row.price),
                qty=new_qty,
                strike=strike,
                right=right,
                bucket_enter_ts=ts_event,
                expiration=expiration,
            )
        elif action == "C":
            new_order = None
        elif action == "M":
            new_qty = int(row.size)
            new_order = OptionOrderState(
                side=old_side,
                price_int=int(row.price),
                qty=new_qty,
                strike=old_strike,
                right=old_right,
                bucket_enter_ts=old_bucket_enter_ts,
                expiration=old_expiration,
            )
        elif action == "F":
            new_qty = old_qty - int(row.size)
            if new_qty > 0:
                new_order = OptionOrderState(
                    side=old_side,
                    price_int=old.price_int,
                    qty=new_qty,
                    strike=old_strike,
                    right=old_right,
                    bucket_enter_ts=old_bucket_enter_ts,
                    expiration=old_expiration,
                )
        elif action == "T":
            new_order = old

        q_old = old_qty if old_side == "B" else old_qty
        q_new = new_order.qty if new_order is not None else 0

        if old_strike > 0 and old_right and old_expiration:
            key = (old_strike, old_right, old_expiration)
            strike_depth[key] = strike_depth.get(key, 0.0) - float(q_old)

        if new_order is not None:
            key = (new_order.strike, new_order.right, new_order.expiration)
            strike_depth[key] = strike_depth.get(key, 0.0) + float(q_new)

        delta = q_new - q_old
        if action == "A" and delta > 0:
            _accumulate_add(accum, strike, right, float(delta))
        elif action in PULL_ACTIONS and delta < 0:
            age_ns = ts_event - old_bucket_enter_ts if old_bucket_enter_ts > 0 else REST_NS
            if age_ns >= REST_NS:
                _accumulate_pull(accum, old_strike, old_right, float(-delta))

        if action == "A":
            orders[order_id] = new_order
        elif action == "C":
            orders.pop(order_id, None)
        elif action == "M":
            orders[order_id] = new_order
        elif action == "F":
            if new_order is None:
                orders.pop(order_id, None)
            else:
                orders[order_id] = new_order

    if current_window_id is not None and not window_invalid:
        ref_price = _compute_ref_price(strike_depth, sorted_strikes)
        strikes_above, strikes_below = _get_strike_buckets(ref_price, sorted_strikes)
        gex_features = _compute_gex_features(
            strike_depth, ref_price, strikes_above, strikes_below, dte_map
        )
        flow_features = _compute_flow_features(accum, strikes_above, strikes_below)
        rows.append(
            {
                "window_start_ts_ns": int(window_start_ts_ns),
                "window_end_ts_ns": int(window_end_ts_ns),
                "ref_price": float(ref_price),
                "strike_above_1": int(strikes_above[0]) if len(strikes_above) > 0 else 0,
                "strike_above_2": int(strikes_above[1]) if len(strikes_above) > 1 else 0,
                "strike_above_3": int(strikes_above[2]) if len(strikes_above) > 2 else 0,
                "strike_above_4": int(strikes_above[3]) if len(strikes_above) > 3 else 0,
                "strike_above_5": int(strikes_above[4]) if len(strikes_above) > 4 else 0,
                "strike_below_1": int(strikes_below[0]) if len(strikes_below) > 0 else 0,
                "strike_below_2": int(strikes_below[1]) if len(strikes_below) > 1 else 0,
                "strike_below_3": int(strikes_below[2]) if len(strikes_below) > 2 else 0,
                "strike_below_4": int(strikes_below[3]) if len(strikes_below) > 3 else 0,
                "strike_below_5": int(strikes_below[4]) if len(strikes_below) > 4 else 0,
                **gex_features,
                **flow_features,
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    _add_derivatives(rows, BASE_FEATURES)
    df_out = pd.DataFrame(rows)
    return df_out.loc[:, OUTPUT_COLUMNS]


def _compute_ref_price(
    strike_depth: dict[tuple[int, str, str], float], sorted_strikes: list[int]
) -> float:
    if not strike_depth:
        if sorted_strikes:
            return float(sorted_strikes[len(sorted_strikes) // 2])
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0
    for (strike, _, _), depth in strike_depth.items():
        if depth > 0:
            total_weight += depth
            weighted_sum += float(strike) * depth

    if total_weight < EPS_QTY:
        if sorted_strikes:
            return float(sorted_strikes[len(sorted_strikes) // 2])
        return 0.0

    return weighted_sum / total_weight


def _get_strike_buckets(
    ref_price: float, sorted_strikes: list[int]
) -> tuple[list[int], list[int]]:
    strikes_above = []
    strikes_below = []

    for strike in sorted_strikes:
        if strike > ref_price:
            strikes_above.append(strike)
        else:
            strikes_below.append(strike)

    strikes_above = strikes_above[:NUM_STRIKES_EACH_SIDE]
    strikes_below = strikes_below[-NUM_STRIKES_EACH_SIDE:]
    strikes_below = list(reversed(strikes_below))

    return strikes_above, strikes_below


def _gamma_bs(strike: int, ref_price: float, T: float, sigma: float = DEFAULT_IV) -> float:
    if ref_price <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    S = ref_price
    K = float(strike)
    r = RISK_FREE_RATE

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    n_prime_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    gamma = n_prime_d1 / (S * sigma * math.sqrt(T))
    return gamma


def _aggregate_depth_by_strike_right(
    strike_depth: dict[tuple[int, str, str], float],
    dte_map: Dict[str, float],
    ref_price: float,
) -> dict[tuple[int, str], float]:
    aggregated: dict[tuple[int, str], float] = {}
    for (strike, right, expiration), depth in strike_depth.items():
        if depth <= 0:
            continue
        T = dte_map.get(expiration, 30.0 / 365.0)
        gamma = _gamma_bs(strike, ref_price, T)
        gex_contribution = depth * gamma * CONTRACT_MULTIPLIER
        key = (strike, right)
        aggregated[key] = aggregated.get(key, 0.0) + gex_contribution
    return aggregated


def _compute_gex_features(
    strike_depth: dict[tuple[int, str, str], float],
    ref_price: float,
    strikes_above: list[int],
    strikes_below: list[int],
    dte_map: Dict[str, float],
) -> dict:
    features = {}

    gex_by_strike_right = _aggregate_depth_by_strike_right(strike_depth, dte_map, ref_price)

    for i, strike in enumerate(strikes_above[:NUM_STRIKES_EACH_SIDE], 1):
        features[f"gex_call_above_{i}"] = float(gex_by_strike_right.get((strike, "C"), 0.0))
        features[f"gex_put_above_{i}"] = float(gex_by_strike_right.get((strike, "P"), 0.0))

    for i in range(len(strikes_above) + 1, NUM_STRIKES_EACH_SIDE + 1):
        features[f"gex_call_above_{i}"] = 0.0
        features[f"gex_put_above_{i}"] = 0.0

    for i, strike in enumerate(strikes_below[:NUM_STRIKES_EACH_SIDE], 1):
        features[f"gex_call_below_{i}"] = float(gex_by_strike_right.get((strike, "C"), 0.0))
        features[f"gex_put_below_{i}"] = float(gex_by_strike_right.get((strike, "P"), 0.0))

    for i in range(len(strikes_below) + 1, NUM_STRIKES_EACH_SIDE + 1):
        features[f"gex_call_below_{i}"] = 0.0
        features[f"gex_put_below_{i}"] = 0.0

    gex_above = sum(
        features.get(f"gex_call_above_{i}", 0.0) - features.get(f"gex_put_above_{i}", 0.0)
        for i in range(1, NUM_STRIKES_EACH_SIDE + 1)
    )
    gex_below = sum(
        features.get(f"gex_call_below_{i}", 0.0) - features.get(f"gex_put_below_{i}", 0.0)
        for i in range(1, NUM_STRIKES_EACH_SIDE + 1)
    )

    features["gex_net_above"] = float(gex_above)
    features["gex_net_below"] = float(gex_below)
    features["gex_total"] = float(gex_above + gex_below)

    denom = abs(gex_above) + abs(gex_below) + EPS_GEX
    features["gex_imbalance_ratio"] = float((gex_above - gex_below) / denom)

    return features


def _compute_flow_features(
    accum: dict, strikes_above: list[int], strikes_below: list[int]
) -> dict:
    strikes_above_set = set(strikes_above)
    strikes_below_set = set(strikes_below)

    flow_call_add_above = 0.0
    flow_call_pull_above = 0.0
    flow_put_add_above = 0.0
    flow_put_pull_above = 0.0
    flow_call_add_below = 0.0
    flow_call_pull_below = 0.0
    flow_put_add_below = 0.0
    flow_put_pull_below = 0.0

    for (strike, right), add_qty in accum["add"].items():
        if strike in strikes_above_set:
            if right == "C":
                flow_call_add_above += add_qty
            else:
                flow_put_add_above += add_qty
        elif strike in strikes_below_set:
            if right == "C":
                flow_call_add_below += add_qty
            else:
                flow_put_add_below += add_qty

    for (strike, right), pull_qty in accum["pull"].items():
        if strike in strikes_above_set:
            if right == "C":
                flow_call_pull_above += pull_qty
            else:
                flow_put_pull_above += pull_qty
        elif strike in strikes_below_set:
            if right == "C":
                flow_call_pull_below += pull_qty
            else:
                flow_put_pull_below += pull_qty

    flow_net_above = (flow_call_add_above - flow_call_pull_above) - (
        flow_put_add_above - flow_put_pull_above
    )
    flow_net_below = (flow_call_add_below - flow_call_pull_below) - (
        flow_put_add_below - flow_put_pull_below
    )

    return {
        "flow_call_add_above": float(flow_call_add_above),
        "flow_call_pull_above": float(flow_call_pull_above),
        "flow_put_add_above": float(flow_put_add_above),
        "flow_put_pull_above": float(flow_put_pull_above),
        "flow_call_add_below": float(flow_call_add_below),
        "flow_call_pull_below": float(flow_call_pull_below),
        "flow_put_add_below": float(flow_put_add_below),
        "flow_put_pull_below": float(flow_put_pull_below),
        "flow_net_above": float(flow_net_above),
        "flow_net_below": float(flow_net_below),
    }


def _new_accumulators() -> dict:
    return {
        "add": {},
        "pull": {},
    }


def _accumulate_add(accum: dict, strike: int, right: str, qty: float) -> None:
    key = (strike, right)
    accum["add"][key] = accum["add"].get(key, 0.0) + qty


def _accumulate_pull(accum: dict, strike: int, right: str, qty: float) -> None:
    key = (strike, right)
    accum["pull"][key] = accum["pull"].get(key, 0.0) + qty


def _add_derivatives(rows: list[dict], feature_names: list[str]) -> list[dict]:
    prev = {name: None for name in feature_names}
    prev_d1 = {name: 0.0 for name in feature_names}
    prev_d2 = {name: 0.0 for name in feature_names}

    for row in rows:
        for name in feature_names:
            if prev[name] is None:
                d1 = 0.0
                d2 = 0.0
                d3 = 0.0
            else:
                d1 = row[name] - prev[name]
                d2 = d1 - prev_d1[name]
                d3 = d2 - prev_d2[name]
            row[f"d1_{name}"] = float(d1)
            row[f"d2_{name}"] = float(d2)
            row[f"d3_{name}"] = float(d3)
            prev[name] = row[name]
            prev_d1[name] = d1
            prev_d2[name] = d2

    return rows
