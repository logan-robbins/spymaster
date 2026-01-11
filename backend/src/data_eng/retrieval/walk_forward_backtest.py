from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..stages.silver.future_mbo.compute_level_vacuum_5s import (
    WINDOW_NS,
    TICK_INT,
    _compute_approach_dir,
    _compute_px_end_int,
    _extract_trade_stream,
    compute_mbo_level_vacuum_5s,
)
from ..stages.gold.future_mbo.build_trigger_vectors import (
    BAR_NS,
    LOOKBACK_WINDOWS,
    N_BARS,
    THRESH_TICKS,
    _build_v_matrix,
    _build_x_matrix,
    _label_trigger,
    _session_start_ns,
)
from ..stages.gold.future_mbo.build_pressure_stream import (
    ARMED_VACUUM,
    WATCH_VACUUM,
    _pressure_scores,
)
from ..retrieval.normalization import RobustStats, apply_robust_scaling, fit_robust_stats, l2_normalize
from ..retrieval.query import SimilarVector
from ..retrieval.trigger_engine import (
    COOLDOWN_WINDOWS,
    H_FIRE,
    K,
    K_RAW,
    LONG_CLASSES,
    MAX_WHIPSAW_RATE,
    MIN_GAP_WINDOWS,
    MIN_RESOLVE_RATE,
    P_CHOP_MAX,
    SHORT_CLASSES,
    STOP_TICKS,
    W_POW,
    EpisodeGate,
    TriggerMetrics,
    TriggerThresholds,
    apply_fire_rule,
    compute_metrics,
)

try:
    import faiss
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss-cpu or faiss-gpu required") from exc


WARMUP_SESSIONS = 20
P_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MARGIN_GRID = [0.10, 0.15, 0.20, 0.25]
HORIZONS = list(range(N_BARS + 1))
PER_TRIGGER_COLUMNS = [
    "session_date",
    "ts_end_ns",
    "level_id",
    "approach_dir",
    "episode_id",
    "p_break",
    "p_reject",
    "p_chop",
    "margin",
    "c_top1",
    "fire_flag",
    "signal",
    "state",
    "risk_q80_ticks",
    "resolve_rate",
    "whipsaw_rate",
    "true_outcome_h0",
    "true_outcome_h1",
    "true_outcome_h2",
    "true_outcome_h3",
    "true_outcome_h4",
    "true_outcome_h5",
    "true_outcome_h6",
    "first_hit_bar_offset",
    "whipsaw_flag",
    "mae_before_upper_ticks",
    "mae_before_lower_ticks",
    "mfe_up_ticks",
    "mfe_down_ticks",
]


@dataclass(frozen=True)
class LevelSpec:
    level_id: str
    p_ref: float
    p_ref_int: int


@dataclass(frozen=True)
class SessionSpec:
    session_date: str
    mbo_path: Path
    levels: List[LevelSpec]


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str
    sessions: List[SessionSpec]
    output_dir: Path


@dataclass
class SessionLevelData:
    session_date: str
    level_id: str
    p_ref: float
    p_ref_int: int
    df_vacuum: pd.DataFrame
    window_start_ts: np.ndarray
    window_end_ts: np.ndarray
    approach_dir: np.ndarray
    px_end_int: np.ndarray
    reset_mask: np.ndarray
    lookback_ok: np.ndarray
    v_matrix: np.ndarray
    v_start: int
    trade_ts: np.ndarray
    trade_px: np.ndarray


@dataclass
class SessionData:
    session_date: str
    mbo_df: pd.DataFrame
    trade_ts: np.ndarray
    trade_px: np.ndarray
    bars_df: pd.DataFrame
    levels: Dict[str, SessionLevelData]


class InMemoryFaiss:
    def __init__(self, level_ids: Iterable[str], dims: int) -> None:
        faiss.omp_set_num_threads(1)
        self.dims = int(dims)
        self.indices: Dict[str, Dict[str, faiss.Index]] = {}
        self.metadata: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        for level_id in level_ids:
            self.indices[level_id] = {}
            self.metadata[level_id] = {}
            for approach_dir in ("approach_up", "approach_down"):
                index = faiss.IndexHNSWFlat(self.dims, 32, faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 64
                self.indices[level_id][approach_dir] = index
                self.metadata[level_id][approach_dir] = []

    def add(self, level_id: str, approach_dir: str, vectors: np.ndarray, metadata: List[Dict[str, object]]) -> None:
        if vectors.size == 0:
            return
        index = self.indices[level_id][approach_dir]
        if vectors.shape[1] != self.dims:
            raise ValueError("Vector dimensions do not match index")
        index.add(vectors.astype(np.float32))
        self.metadata[level_id][approach_dir].extend(metadata)

    def find_similar(self, level_id: str, approach_dir: str, vector: np.ndarray, k: int) -> List[SimilarVector]:
        index = self.indices[level_id][approach_dir]
        if index.ntotal == 0:
            return []
        query = vector.reshape(1, -1).astype(np.float32)
        if query.shape[1] != self.dims:
            raise ValueError("Query vector dimensions do not match index")
        k_fetch = min(k, index.ntotal)
        distances, indices = index.search(query, k_fetch)
        meta = self.metadata[level_id][approach_dir]
        out: List[SimilarVector] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            out.append(
                SimilarVector(
                    index_id=int(idx),
                    similarity=float(dist),
                    metadata=meta[int(idx)],
                )
            )
        return out

    def assert_no_leakage(self, session_date: str) -> None:
        for level_meta in self.metadata.values():
            for meta_list in level_meta.values():
                for meta in meta_list:
                    if str(meta.get("session_date")) >= session_date:
                        raise ValueError("Index contains same or future session data")


def _load_config(path: Path) -> BacktestConfig:
    payload = json.loads(path.read_text())
    symbol = str(payload.get("symbol", "")).strip()
    if not symbol:
        raise ValueError("Missing symbol in config")
    sessions_raw = payload.get("sessions", [])
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ValueError("Missing sessions in config")
    output_dir = Path(str(payload.get("output_dir", "")).strip())
    if not output_dir:
        raise ValueError("Missing output_dir in config")

    sessions: List[SessionSpec] = []
    for session in sessions_raw:
        session_date = str(session.get("session_date", "")).strip()
        if not session_date:
            raise ValueError("Missing session_date in config")
        mbo_path = Path(str(session.get("mbo_path", "")).strip())
        if not mbo_path:
            raise ValueError("Missing mbo_path in config")
        levels_raw = session.get("levels", [])
        if not isinstance(levels_raw, list) or not levels_raw:
            raise ValueError("Missing levels in config")
        levels: List[LevelSpec] = []
        for level in levels_raw:
            level_id = str(level.get("level_id", "")).strip()
            if not level_id:
                raise ValueError("Missing level_id in config")
            if "p_ref" not in level or "p_ref_int" not in level:
                raise ValueError("Missing p_ref or p_ref_int in config")
            p_ref = float(level["p_ref"])
            p_ref_int = int(level["p_ref_int"])
            levels.append(LevelSpec(level_id=level_id, p_ref=p_ref, p_ref_int=p_ref_int))
        sessions.append(SessionSpec(session_date=session_date, mbo_path=mbo_path, levels=levels))

    session_dates = [s.session_date for s in sessions]
    if session_dates != sorted(session_dates):
        raise ValueError("Sessions must be sorted by session_date ascending")
    if len(session_dates) != len(set(session_dates)):
        raise ValueError("Duplicate session_date in config")

    return BacktestConfig(symbol=symbol, sessions=sessions, output_dir=output_dir)


def _parse_price(value: object) -> int:
    if value is None:
        raise ValueError("Missing price")
    if isinstance(value, int):
        if abs(value) < 1_000_000_000:
            return int(Decimal(value) * Decimal("1000000000"))
        return int(value)
    return int(Decimal(str(value)) * Decimal("1000000000"))


def _parse_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"Missing {name}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc


def _load_mbo_preview(path: Path, symbol: str, session_date: str) -> pd.DataFrame:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("MBO source must be a JSON list")

    rows = []
    for entry in raw:
        if entry.get("symbol") != symbol:
            continue
        hd = entry.get("hd", {})
        ts_event_raw = hd.get("ts_event")
        ts_recv_raw = entry.get("ts_recv")
        if ts_event_raw is None or ts_recv_raw is None:
            raise ValueError("Missing ts_event or ts_recv in MBO source")
        ts_event_ns = pd.to_datetime(ts_event_raw, utc=True).value
        ts_recv_ns = pd.to_datetime(ts_recv_raw, utc=True).value

        action = entry.get("action")
        side = entry.get("side")
        if action is None or side is None:
            raise ValueError("Missing action or side in MBO source")

        price_val = entry.get("price")
        price_int = 0
        if price_val is not None:
            price_int = _parse_price(price_val)

        rows.append(
            {
                "ts_recv": int(ts_recv_ns),
                "size": int(entry.get("size", 0)),
                "ts_event": int(ts_event_ns),
                "channel_id": int(entry.get("channel_id", 0)),
                "rtype": int(hd.get("rtype", 0)),
                "order_id": _parse_int(entry.get("order_id"), "order_id"),
                "publisher_id": int(hd.get("publisher_id", 0)),
                "flags": int(entry.get("flags", 0)),
                "instrument_id": int(hd.get("instrument_id", 0)),
                "ts_in_delta": int(entry.get("ts_in_delta", 0)),
                "action": action,
                "sequence": int(entry.get("sequence", 0)),
                "side": side,
                "symbol": symbol,
                "price": int(price_int),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No rows loaded for symbol")

    event_dates = pd.to_datetime(df["ts_event"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    df = df.loc[event_dates == session_date].copy()
    if len(df) == 0:
        raise ValueError("No rows found for session_date in MBO source")

    df = df.sort_values(["ts_event", "sequence"], ascending=[True, True])
    for col in [
        "ts_recv",
        "size",
        "ts_event",
        "channel_id",
        "rtype",
        "order_id",
        "publisher_id",
        "flags",
        "instrument_id",
        "ts_in_delta",
        "sequence",
        "price",
    ]:
        df[col] = df[col].astype("int64")
    return df.reset_index(drop=True)


def _build_trade_bars(trade_ts: np.ndarray, trade_px: np.ndarray) -> pd.DataFrame:
    if trade_ts.size == 0:
        return pd.DataFrame(columns=["bar_id", "high_int", "low_int", "close_int", "bar_end_ts"])
    bar_id = trade_ts // BAR_NS
    df = pd.DataFrame({"bar_id": bar_id, "price": trade_px})
    grouped = df.groupby("bar_id", sort=True)["price"]
    bars = pd.DataFrame(
        {
            "bar_id": grouped.max().index.astype(np.int64),
            "high_int": grouped.max().to_numpy(dtype=np.int64),
            "low_int": grouped.min().to_numpy(dtype=np.int64),
            "close_int": grouped.last().to_numpy(dtype=np.int64),
        }
    )
    bars["bar_end_ts"] = (bars["bar_id"] + 1) * BAR_NS
    return bars.reset_index(drop=True)


def _prepare_session_data(spec: SessionSpec, symbol: str) -> SessionData:
    mbo_df = _load_mbo_preview(spec.mbo_path, symbol, spec.session_date)
    trade_ts, trade_px = _extract_trade_stream(mbo_df)
    bars_df = _build_trade_bars(trade_ts, trade_px)

    levels: Dict[str, SessionLevelData] = {}
    reset_windows = set(
        (mbo_df.loc[mbo_df["action"] == "R", "ts_event"].to_numpy(dtype=np.int64) // WINDOW_NS).tolist()
    )

    for level in spec.levels:
        df_vacuum = compute_mbo_level_vacuum_5s(mbo_df, level.p_ref, symbol)
        df_vacuum = df_vacuum.sort_values("window_end_ts_ns").reset_index(drop=True)
        if len(df_vacuum) == 0:
            levels[level.level_id] = SessionLevelData(
                session_date=spec.session_date,
                level_id=level.level_id,
                p_ref=level.p_ref,
                p_ref_int=level.p_ref_int,
                df_vacuum=df_vacuum,
                window_start_ts=np.array([], dtype=np.int64),
                window_end_ts=np.array([], dtype=np.int64),
                approach_dir=np.array([], dtype=object),
                px_end_int=np.array([], dtype=np.float64),
                reset_mask=np.array([], dtype=bool),
                lookback_ok=np.array([], dtype=bool),
                v_matrix=np.empty((0, 0), dtype=np.float64),
                v_start=LOOKBACK_WINDOWS - 1,
                trade_ts=trade_ts,
                trade_px=trade_px,
            )
            continue

        window_start_ts = df_vacuum["window_start_ts_ns"].to_numpy(dtype=np.int64)
        window_end_ts = df_vacuum["window_end_ts_ns"].to_numpy(dtype=np.int64)
        px_end_int = _compute_px_end_int(trade_ts, trade_px, window_end_ts)
        approach_dir = _compute_approach_dir(px_end_int, level.p_ref_int)

        window_ids = (window_end_ts // WINDOW_NS) - 1
        reset_mask = np.isin(window_ids, list(reset_windows))

        lookback_ok = np.zeros(len(window_end_ts), dtype=bool)
        start = LOOKBACK_WINDOWS - 1
        if len(window_end_ts) > start:
            expected = start * WINDOW_NS
            ok = (window_end_ts[start:] - window_end_ts[:-start]) == expected
            lookback_ok[start:] = ok

        x_matrix = _build_x_matrix(df_vacuum)
        v_matrix = _build_v_matrix(x_matrix)

        levels[level.level_id] = SessionLevelData(
            session_date=spec.session_date,
            level_id=level.level_id,
            p_ref=level.p_ref,
            p_ref_int=level.p_ref_int,
            df_vacuum=df_vacuum,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            approach_dir=approach_dir,
            px_end_int=px_end_int,
            reset_mask=reset_mask,
            lookback_ok=lookback_ok,
            v_matrix=v_matrix,
            v_start=start,
            trade_ts=trade_ts,
            trade_px=trade_px,
        )

    return SessionData(
        session_date=spec.session_date,
        mbo_df=mbo_df,
        trade_ts=trade_ts,
        trade_px=trade_px,
        bars_df=bars_df,
        levels=levels,
    )


def _collect_warmup_vectors(sessions: List[SessionData]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for session in sessions[:WARMUP_SESSIONS]:
        for level in session.levels.values():
            if level.v_matrix.size == 0:
                continue
            n_rows = len(level.window_end_ts)
            if n_rows <= level.v_start:
                continue
            eligible = (
                level.lookback_ok
                & (level.approach_dir != "approach_none")
                & (~level.reset_mask)
            )
            idx = np.where(eligible & (np.arange(n_rows) >= level.v_start))[0]
            if idx.size == 0:
                continue
            vectors.append(level.v_matrix[idx - level.v_start])
    if not vectors:
        raise ValueError("No warmup vectors available for scaling")
    return np.vstack(vectors)


def _normalize_vectors(v_matrix: np.ndarray, stats: RobustStats) -> Tuple[np.ndarray, np.ndarray]:
    scaled = apply_robust_scaling(v_matrix, stats)
    normalized, valid = l2_normalize(scaled)
    return normalized.astype(np.float32), valid


def _build_metrics(
    index: InMemoryFaiss,
    level_id: str,
    approach_dir: str,
    vector: np.ndarray,
) -> TriggerMetrics:
    neighbors = index.find_similar(level_id, approach_dir, vector, K_RAW)
    if len(neighbors) < K:
        return TriggerMetrics(
            p_break=0.0,
            p_reject=0.0,
            p_chop=0.0,
            p_top1=0.0,
            margin=0.0,
            risk_q80_ticks=0.0,
            resolve_rate=0.0,
            whipsaw_rate=0.0,
            c_top1="CHOP",
            neighbors_ok=False,
        )
    return compute_metrics(neighbors[:K], approach_dir, W_POW)


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _build_pressure_message(
    symbol: str,
    session_date: str,
    level_id: str,
    p_ref: float,
    ts_end: int,
    approach_dir: str,
    pressure: Dict[str, object],
    retrieval: Dict[str, object],
    state: str,
    fire_flag: int,
    signal: str,
    episode_id: int,
) -> Dict[str, object]:
    if state == "IDLE":
        pressure_block = {
            "above": {
                "retreat": None,
                "decay_or_build": None,
                "localization": None,
                "shock": None,
                "score": None,
            },
            "below": {
                "retreat_or_recede": None,
                "decay_or_build": None,
                "localization": None,
                "shock": None,
                "score": None,
            },
            "vacuum_score": None,
        }
        retrieval_block = {
            "h_fire": int(H_FIRE),
            "p_break": None,
            "p_reject": None,
            "p_chop": None,
            "margin": None,
            "risk_q80_ticks": None,
            "resolve_rate": None,
            "whipsaw_rate": None,
        }
    else:
        pressure_block = {
            "above": {
                "retreat": _safe_float(pressure["above_retreat"]),
                "decay_or_build": _safe_float(pressure["above_decay"]),
                "localization": _safe_float(pressure["above_local"]),
                "shock": _safe_float(pressure["above_shock"]),
                "score": _safe_float(pressure["above_score"]),
            },
            "below": {
                "retreat_or_recede": _safe_float(pressure["below_retreat"]),
                "decay_or_build": _safe_float(pressure["below_decay"]),
                "localization": _safe_float(pressure["below_local"]),
                "shock": _safe_float(pressure["below_shock"]),
                "score": _safe_float(pressure["below_score"]),
            },
            "vacuum_score": _safe_float(pressure["vacuum_score"]),
        }
        retrieval_block = {
            "h_fire": int(H_FIRE),
            "p_break": _safe_float(retrieval["p_break"]),
            "p_reject": _safe_float(retrieval["p_reject"]),
            "p_chop": _safe_float(retrieval["p_chop"]),
            "margin": _safe_float(retrieval["margin"]),
            "risk_q80_ticks": _safe_float(retrieval["risk_q80_ticks"]),
            "resolve_rate": _safe_float(retrieval["resolve_rate"]),
            "whipsaw_rate": _safe_float(retrieval["whipsaw_rate"]),
        }

    return {
        "ts_end_ns": int(ts_end),
        "symbol": symbol,
        "session_date": session_date,
        "level_id": level_id,
        "p_ref": float(p_ref),
        "approach_dir": approach_dir,
        "pressure": pressure_block,
        "retrieval": retrieval_block,
        "signal": {
            "state": state,
            "fire_flag": int(fire_flag),
            "signal": signal,
            "episode_id": int(episode_id),
        },
    }


def _select_thresholds(calibration_streams: List[Dict[str, object]]) -> TriggerThresholds:
    if not calibration_streams:
        raise ValueError("No calibration streams available")

    results: List[Dict[str, float]] = []
    total_eligible = sum(len(stream["trigger_by_ts"]) for stream in calibration_streams)
    if total_eligible == 0:
        raise ValueError("No eligible calibration triggers available")
    for p_min in P_GRID:
        for margin_min in MARGIN_GRID:
            thresholds = TriggerThresholds(
                p_min=p_min,
                margin_min=margin_min,
                p_chop_max=P_CHOP_MAX,
                stop_ticks=STOP_TICKS,
                min_resolve_rate=MIN_RESOLVE_RATE,
                max_whipsaw_rate=MAX_WHIPSAW_RATE,
            )
            total_fires = 0
            correct = 0
            chop_false = 0
            whipsaw_hit = 0
            stop_violation = 0
            resolve_bar1 = 0

            for stream in calibration_streams:
                gate = EpisodeGate(
                    cooldown_windows=COOLDOWN_WINDOWS,
                    min_gap_windows=MIN_GAP_WINDOWS,
                )
                window_end_ts = stream["window_end_ts"]
                approach_dirs = stream["approach_dir"]
                trigger_by_ts = stream["trigger_by_ts"]
                for idx in range(len(window_end_ts)):
                    approach_dir = str(approach_dirs[idx])
                    episode_id, blocked = gate.step(idx, approach_dir)
                    ts_end = int(window_end_ts[idx])
                    row = trigger_by_ts.get(ts_end)
                    if row is None:
                        continue
                    metrics = TriggerMetrics(
                        p_break=float(row["p_break"]),
                        p_reject=float(row["p_reject"]),
                        p_chop=float(row["p_chop"]),
                        p_top1=float(row["p_top1"]),
                        margin=float(row["margin"]),
                        risk_q80_ticks=float(row["risk_q80_ticks"]),
                        resolve_rate=float(row["resolve_rate"]),
                        whipsaw_rate=float(row["whipsaw_rate"]),
                        c_top1=str(row["c_top1"]),
                        neighbors_ok=bool(row["neighbors_ok"]),
                    )
                    fire_flag, _ = apply_fire_rule(metrics, thresholds)
                    if blocked:
                        fire_flag = 0
                    if fire_flag != 1:
                        continue
                    gate.register_fire(episode_id)
                    total_fires += 1

                    true_h1 = str(row["true_outcome_h1"])
                    if true_h1 != "WHIPSAW" and metrics.c_top1 == true_h1:
                        correct += 1
                    if true_h1 == "CHOP":
                        chop_false += 1
                    if true_h1 == "WHIPSAW":
                        whipsaw_hit += 1

                    if metrics.c_top1 in LONG_CLASSES:
                        mae = _safe_float(row["mae_before_upper_ticks"])
                    elif metrics.c_top1 in SHORT_CLASSES:
                        mae = _safe_float(row["mae_before_lower_ticks"])
                    else:
                        mae = 0.0
                    if mae > STOP_TICKS:
                        stop_violation += 1

                    offset = row["first_hit_bar_offset"]
                    if offset is not None and int(offset) <= 1:
                        resolve_bar1 += 1

            precision = (correct / total_fires) if total_fires else 0.0
            fire_rate = (total_fires / total_eligible) if total_eligible else 0.0
            chop_false_rate = (chop_false / total_fires) if total_fires else 0.0
            whipsaw_hit_rate = (whipsaw_hit / total_fires) if total_fires else 0.0
            stop_violation_rate = (stop_violation / total_fires) if total_fires else 0.0
            resolve_by_bar1_rate = (resolve_bar1 / total_fires) if total_fires else 0.0

            results.append(
                {
                    "p_min": p_min,
                    "margin_min": margin_min,
                    "precision": precision,
                    "fire_rate": fire_rate,
                    "chop_false_rate": chop_false_rate,
                    "whipsaw_hit_rate": whipsaw_hit_rate,
                    "stop_violation_rate": stop_violation_rate,
                    "resolve_by_bar1_rate": resolve_by_bar1_rate,
                }
            )

    filtered = []
    for row in results:
        if row["precision"] < 0.60:
            continue
        if row["chop_false_rate"] > 0.15:
            continue
        if row["stop_violation_rate"] > 0.20:
            continue
        if row["fire_rate"] < 0.01 or row["fire_rate"] > 0.20:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("No threshold pair meets constraints")

    def score(r: Dict[str, float]) -> Tuple[float, float, float]:
        return (r["fire_rate"] * r["precision"], r["precision"], r["resolve_by_bar1_rate"])

    best = max(filtered, key=score)
    return TriggerThresholds(
        p_min=best["p_min"],
        margin_min=best["margin_min"],
        p_chop_max=P_CHOP_MAX,
        stop_ticks=STOP_TICKS,
        min_resolve_rate=MIN_RESOLVE_RATE,
        max_whipsaw_rate=MAX_WHIPSAW_RATE,
    )


def _apply_thresholds(
    metrics: TriggerMetrics,
    thresholds: TriggerThresholds,
    blocked: bool,
) -> Tuple[int, str]:
    fire_flag, signal = apply_fire_rule(metrics, thresholds)
    if blocked:
        return 0, "NONE"
    return fire_flag, signal


def _trigger_state(fire_flag: int, vacuum_score: float | None, margin: float) -> str:
    if fire_flag == 1:
        return "FIRE"
    if vacuum_score is None:
        return "TRACK"
    if vacuum_score >= ARMED_VACUUM and margin >= 0.20:
        return "ARMED"
    if vacuum_score >= WATCH_VACUUM:
        return "WATCH"
    return "TRACK"


def _process_level(
    session: SessionData,
    level: SessionLevelData,
    symbol: str,
    stats: RobustStats,
    index: InMemoryFaiss,
    thresholds: TriggerThresholds | None,
    query_enabled: bool,
    apply_thresholds_flag: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Tuple[np.ndarray, Dict[str, object]]]]:
    df_vacuum = level.df_vacuum
    if len(df_vacuum) == 0:
        return [], [], []

    normalized, valid = _normalize_vectors(level.v_matrix, stats) if level.v_matrix.size else (np.empty((0, 0)), np.array([], dtype=bool))
    n_rows = len(level.window_end_ts)
    embed_valid = np.zeros(n_rows, dtype=bool)
    if normalized.size:
        embed_valid[level.v_start :] = valid

    eligible = (
        level.lookback_ok
        & (level.approach_dir != "approach_none")
        & (~level.reset_mask)
        & embed_valid
    )

    gate = EpisodeGate(
        cooldown_windows=COOLDOWN_WINDOWS,
        min_gap_windows=MIN_GAP_WINDOWS,
    )
    insert_rows: List[Tuple[np.ndarray, Dict[str, object]]] = []
    trigger_rows: List[Dict[str, object]] = []
    pressure_rows: List[Dict[str, object]] = []

    session_start_ns = _session_start_ns(level.session_date)
    session_start_bar_id = session_start_ns // BAR_NS
    upper_barrier_int = level.p_ref_int + THRESH_TICKS * TICK_INT
    lower_barrier_int = level.p_ref_int - THRESH_TICKS * TICK_INT

    for idx, vac in enumerate(df_vacuum.itertuples(index=False)):
        ts_end = int(level.window_end_ts[idx])
        approach_dir = str(level.approach_dir[idx])
        episode_id, blocked = gate.step(idx, approach_dir)
        pressure = _pressure_scores(vac, approach_dir)

        metrics = TriggerMetrics(
            p_break=0.0,
            p_reject=0.0,
            p_chop=0.0,
            p_top1=0.0,
            margin=0.0,
            risk_q80_ticks=0.0,
            resolve_rate=0.0,
            whipsaw_rate=0.0,
            c_top1="CHOP",
            neighbors_ok=False,
        )
        fire_flag = 0
        signal = "NONE"

        if eligible[idx]:
            vec = normalized[idx - level.v_start]
            label = _label_trigger(
                trade_ts=level.trade_ts,
                trade_px=level.trade_px,
                trigger_ts=ts_end,
                approach_dir=approach_dir,
                p_ref_int=level.p_ref_int,
                upper_barrier_int=upper_barrier_int,
                lower_barrier_int=lower_barrier_int,
                session_start_bar_id=session_start_bar_id,
            )
            if query_enabled:
                metrics = _build_metrics(index, level.level_id, approach_dir, vec)
            if apply_thresholds_flag and thresholds is not None:
                fire_flag, signal = _apply_thresholds(metrics, thresholds, blocked)
                if fire_flag == 1:
                    gate.register_fire(episode_id)

            trigger_rows.append(
                {
                    "session_date": level.session_date,
                    "ts_end_ns": ts_end,
                    "symbol": symbol,
                    "level_id": level.level_id,
                    "approach_dir": approach_dir,
                    "episode_id": int(episode_id),
                    "p_break": _safe_float(metrics.p_break),
                    "p_reject": _safe_float(metrics.p_reject),
                    "p_chop": _safe_float(metrics.p_chop),
                    "p_top1": _safe_float(metrics.p_top1),
                    "margin": _safe_float(metrics.margin),
                    "c_top1": str(metrics.c_top1),
                    "risk_q80_ticks": _safe_float(metrics.risk_q80_ticks),
                    "resolve_rate": _safe_float(metrics.resolve_rate),
                    "whipsaw_rate": _safe_float(metrics.whipsaw_rate),
                    "fire_flag": int(fire_flag),
                    "signal": str(signal),
                    "state": _trigger_state(fire_flag, _safe_float(pressure["vacuum_score"]), _safe_float(metrics.margin)),
                    "first_hit": label.first_hit,
                    "true_outcome_h0": label.true_outcome_h[0],
                    "true_outcome_h1": label.true_outcome_h[1],
                    "true_outcome_h2": label.true_outcome_h[2],
                    "true_outcome_h3": label.true_outcome_h[3],
                    "true_outcome_h4": label.true_outcome_h[4],
                    "true_outcome_h5": label.true_outcome_h[5],
                    "true_outcome_h6": label.true_outcome_h[6],
                    "first_hit_bar_offset": label.first_hit_bar_offset,
                    "whipsaw_flag": int(label.whipsaw_flag),
                    "mae_before_upper_ticks": _safe_float(label.mae_before_upper_ticks),
                    "mae_before_lower_ticks": _safe_float(label.mae_before_lower_ticks),
                    "mfe_up_ticks": _safe_float(label.mfe_up_ticks),
                    "mfe_down_ticks": _safe_float(label.mfe_down_ticks),
                    "neighbors_ok": bool(metrics.neighbors_ok),
                }
            )

            insert_rows.append(
                (
                    vec,
                    {
                        "session_date": level.session_date,
                        "ts_end_ns": ts_end,
                        "level_id": level.level_id,
                        "approach_dir": approach_dir,
                        "true_outcome_h1": label.true_outcome_h[1],
                        "true_outcome_h0": label.true_outcome_h[0],
                        "true_outcome_h2": label.true_outcome_h[2],
                        "true_outcome_h3": label.true_outcome_h[3],
                        "true_outcome_h4": label.true_outcome_h[4],
                        "true_outcome_h5": label.true_outcome_h[5],
                        "true_outcome_h6": label.true_outcome_h[6],
                        "whipsaw_flag": int(label.whipsaw_flag),
                        "first_hit_bar_offset": label.first_hit_bar_offset,
                        "mae_before_upper_ticks": _safe_float(label.mae_before_upper_ticks),
                        "mae_before_lower_ticks": _safe_float(label.mae_before_lower_ticks),
                    },
                )
            )

        retrieval = {
            "p_break": metrics.p_break,
            "p_reject": metrics.p_reject,
            "p_chop": metrics.p_chop,
            "margin": metrics.margin,
            "risk_q80_ticks": metrics.risk_q80_ticks,
            "resolve_rate": metrics.resolve_rate,
            "whipsaw_rate": metrics.whipsaw_rate,
        }
        if approach_dir == "approach_none":
            state = "IDLE"
            fire_out = 0
            signal_out = "NONE"
        else:
            state = _trigger_state(fire_flag, _safe_float(pressure["vacuum_score"]), _safe_float(metrics.margin))
            fire_out = fire_flag
            signal_out = signal

        pressure_rows.append(
            _build_pressure_message(
                symbol=symbol,
                session_date=level.session_date,
                level_id=level.level_id,
                p_ref=level.p_ref,
                ts_end=ts_end,
                approach_dir=approach_dir,
                pressure=pressure,
                retrieval=retrieval,
                state=state,
                fire_flag=fire_out,
                signal=signal_out,
                episode_id=episode_id,
            )
        )

    return trigger_rows, pressure_rows, insert_rows


def _summarize_group(df: pd.DataFrame) -> Dict[str, object]:
    eligible = len(df)
    fires = df[df["fire_flag"] == 1]
    fires_count = len(fires)

    precision_h = {}
    for h in HORIZONS:
        label_col = f"true_outcome_h{h}"
        if fires_count == 0:
            precision_h[h] = 0.0
        else:
            correct = (fires[label_col] == fires["c_top1"]) & (fires[label_col] != "WHIPSAW")
            precision_h[h] = float(correct.sum()) / float(fires_count)

    true_h1 = fires["true_outcome_h1"] if fires_count else pd.Series([], dtype=object)
    chop_false_rate = float((true_h1 == "CHOP").sum()) / float(fires_count) if fires_count else 0.0

    stop_violation = 0
    resolve_bar1 = 0
    for row in fires.itertuples(index=False):
        c_top1 = str(getattr(row, "c_top1"))
        if c_top1 in LONG_CLASSES:
            mae = _safe_float(getattr(row, "mae_before_upper_ticks"))
        elif c_top1 in SHORT_CLASSES:
            mae = _safe_float(getattr(row, "mae_before_lower_ticks"))
        else:
            mae = 0.0
        if mae > STOP_TICKS:
            stop_violation += 1
        offset = getattr(row, "first_hit_bar_offset")
        if offset is not None and int(offset) <= 1:
            resolve_bar1 += 1

    stop_violation_rate = float(stop_violation) / float(fires_count) if fires_count else 0.0
    resolve_by_bar1_rate = float(resolve_bar1) / float(fires_count) if fires_count else 0.0

    summary = {
        "eligible_count": int(eligible),
        "fires_count": int(fires_count),
        "fire_rate": float(fires_count) / float(eligible) if eligible else 0.0,
        "precision_H1": precision_h[1],
        "chop_false_rate_H1": chop_false_rate,
        "stop_violation_rate_H1": stop_violation_rate,
        "resolve_by_bar1_rate": resolve_by_bar1_rate,
    }
    for h in HORIZONS:
        summary[f"precision_H{h}"] = precision_h[h]
    return summary


def _check_monotonicity(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        labels = [row[f"true_outcome_h{h}"] for h in HORIZONS]
        locked = None
        for label in labels:
            if locked is None:
                if label in ("BREAK_UP", "BREAK_DOWN", "REJECT_UP", "REJECT_DOWN", "WHIPSAW"):
                    locked = label
            else:
                if label != locked:
                    raise ValueError("Label monotonicity violated")


def _check_stop_consistency(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        true_h1 = row["true_outcome_h1"]
        first_hit = row.get("first_hit")
        if true_h1 == "BREAK_UP" and first_hit not in ("UPPER",):
            raise ValueError("Stop consistency violated for BREAK_UP")
        if true_h1 == "BREAK_DOWN" and first_hit not in ("LOWER",):
            raise ValueError("Stop consistency violated for BREAK_DOWN")


def _assert_finite(rows: List[Dict[str, object]], fields: List[str]) -> None:
    for row in rows:
        for field in fields:
            val = row.get(field)
            if val is None:
                continue
            try:
                out = float(val)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(out):
                raise ValueError(f"Non-finite value in {field}")


def _assert_pressure_finite(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        pressure = row.get("pressure", {})
        retrieval = row.get("retrieval", {})
        for block in (pressure.get("above", {}), pressure.get("below", {})):
            for val in block.values():
                if val is None:
                    continue
                if not np.isfinite(float(val)):
                    raise ValueError("Non-finite value in pressure block")
        vac = pressure.get("vacuum_score")
        if vac is not None and not np.isfinite(float(vac)):
            raise ValueError("Non-finite value in vacuum_score")
        for val in retrieval.values():
            if val is None:
                continue
            if not np.isfinite(float(val)):
                raise ValueError("Non-finite value in retrieval block")


def _write_outputs(
    output_dir: Path,
    trigger_rows: List[Dict[str, object]],
    pressure_rows: List[Dict[str, object]],
    per_session_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if trigger_rows:
        df_trigger = pd.DataFrame(trigger_rows)
        missing = [col for col in PER_TRIGGER_COLUMNS if col not in df_trigger.columns]
        if missing:
            raise ValueError(f"Missing trigger columns: {missing}")
        df_trigger = df_trigger.loc[:, PER_TRIGGER_COLUMNS]
        df_trigger.to_csv(output_dir / "per_trigger.csv", index=False)
    if pressure_rows:
        with (output_dir / "pressure_stream.jsonl").open("w") as fh:
            for row in pressure_rows:
                fh.write(json.dumps(row) + "\n")
    per_session_summary.to_csv(output_dir / "per_session_summary.csv", index=False)
    global_summary.to_csv(output_dir / "global_summary.csv", index=False)


def run_backtest(config: BacktestConfig) -> None:
    if len(config.sessions) < WARMUP_SESSIONS:
        raise ValueError("Insufficient sessions for warmup")

    sessions_data = [_prepare_session_data(spec, config.symbol) for spec in config.sessions]
    warmup_vectors = _collect_warmup_vectors(sessions_data)
    stats = fit_robust_stats(warmup_vectors)

    level_ids = sorted({lvl.level_id for session in sessions_data for lvl in session.levels.values()})
    dims = warmup_vectors.shape[1]
    index = InMemoryFaiss(level_ids, dims)

    calibration_streams: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []
    pressure_rows: List[Dict[str, object]] = []

    thresholds: TriggerThresholds | None = None

    for i, session in enumerate(sessions_data):
        index.assert_no_leakage(session.session_date)
        query_enabled = i > 0
        apply_thresholds_flag = i >= WARMUP_SESSIONS and thresholds is not None

        session_trigger_rows: List[Dict[str, object]] = []
        session_pressure_rows: List[Dict[str, object]] = []
        session_insert: List[Tuple[np.ndarray, Dict[str, object]]] = []

        for level in session.levels.values():
            triggers, pressure, inserts = _process_level(
                session=session,
                level=level,
                symbol=config.symbol,
                stats=stats,
                index=index,
                thresholds=thresholds,
                query_enabled=query_enabled,
                apply_thresholds_flag=apply_thresholds_flag,
            )
            session_trigger_rows.extend(triggers)
            session_pressure_rows.extend(pressure)
            session_insert.extend(inserts)

        if i < WARMUP_SESSIONS and i >= 1:
            for level in session.levels.values():
                trigger_by_ts = {
                    row["ts_end_ns"]: row
                    for row in session_trigger_rows
                    if row["level_id"] == level.level_id
                }
                calibration_streams.append(
                    {
                        "session_date": session.session_date,
                        "level_id": level.level_id,
                        "window_end_ts": level.window_end_ts,
                        "approach_dir": level.approach_dir,
                        "trigger_by_ts": trigger_by_ts,
                    }
                )
        else:
            eval_rows.extend(session_trigger_rows)

        pressure_rows.extend(session_pressure_rows)

        if session_insert:
            session_insert.sort(key=lambda item: item[1]["ts_end_ns"])
            for level_id in level_ids:
                for approach_dir in ("approach_up", "approach_down"):
                    vecs = []
                    metas = []
                    for vec, meta in session_insert:
                        if meta["level_id"] != level_id or meta["approach_dir"] != approach_dir:
                            continue
                        vecs.append(vec)
                        metas.append(meta)
                    if vecs:
                        index.add(level_id, approach_dir, np.vstack(vecs), metas)

        if i == WARMUP_SESSIONS - 1:
            thresholds = _select_thresholds(calibration_streams)

    if not eval_rows:
        raise ValueError("No evaluation rows produced")

    _check_monotonicity(eval_rows)
    _check_stop_consistency(eval_rows)
    _assert_finite(
        eval_rows,
        [
            "p_break",
            "p_reject",
            "p_chop",
            "p_top1",
            "margin",
            "risk_q80_ticks",
            "resolve_rate",
            "whipsaw_rate",
        ],
    )
    _assert_pressure_finite(pressure_rows)

    df_eval = pd.DataFrame(eval_rows)
    group_keys = ["session_date", "level_id", "approach_dir"]
    summaries = []
    for keys, group in df_eval.groupby(group_keys):
        summary = _summarize_group(group)
        summary["session_date"] = keys[0]
        summary["level_id"] = keys[1]
        summary["approach_dir"] = keys[2]
        summaries.append(summary)
    per_session_summary = pd.DataFrame(summaries)

    global_rows = []
    overall = _summarize_group(df_eval)
    overall["group_type"] = "overall"
    overall["group_value"] = "ALL"
    global_rows.append(overall)
    for level_id, group in df_eval.groupby("level_id"):
        summary = _summarize_group(group)
        summary["group_type"] = "level_id"
        summary["group_value"] = level_id
        global_rows.append(summary)
    for approach_dir, group in df_eval.groupby("approach_dir"):
        summary = _summarize_group(group)
        summary["group_type"] = "approach_dir"
        summary["group_value"] = approach_dir
        global_rows.append(summary)
    global_summary = pd.DataFrame(global_rows)

    _write_outputs(config.output_dir, eval_rows, pressure_rows, per_session_summary, global_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest runner.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    run_backtest(config)


if __name__ == "__main__":
    main()
