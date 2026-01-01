from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd

from .level_universe import LevelUniverse
from .market_state import MarketState
from .viewport_manager import ViewportManager, ViewportState
from .viewport_feature_builder import ViewportFeatureBuilder
from .inference_engine import ViewportInferenceEngine
from src.ml.feature_sets import STAGE_B_ONLY_PREFIXES
from src.common.config import CONFIG
from src.core.feature_historian import FeatureHistorian


@dataclass
class InferenceState:
    last_infer_ns: Optional[int] = None
    last_stage: Optional[str] = None
    last_distance_signed: Optional[float] = None
    last_in_monitor: Optional[bool] = None
    last_barrier_state: Optional[str] = None
    last_tape_imbalance: Optional[float] = None
    last_gamma_exposure: Optional[float] = None
    last_sweep_detected: Optional[bool] = None


class ViewportScoringService:
    """
    Orchestrates viewport updates, feature building, and inference scoring.
    """

    def __init__(
        self,
        market_state: MarketState,
        level_universe: LevelUniverse,
        viewport_manager: ViewportManager,
        feature_builder: ViewportFeatureBuilder,
        stage_a_engine: ViewportInferenceEngine,
        stage_b_engine: ViewportInferenceEngine,
        trading_date: Optional[str] = None,
        feature_historian: Optional[FeatureHistorian] = None
    ):
        self.market_state = market_state
        self.level_universe = level_universe
        self.viewport_manager = viewport_manager
        self.feature_builder = feature_builder
        self.stage_a_engine = stage_a_engine
        self.stage_b_engine = stage_b_engine
        self.trading_date = trading_date
        self.feature_historian = feature_historian
        self._confirmation_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._inference_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._inference_state: Dict[str, InferenceState] = {}

    def score_viewport(self) -> List[Dict[str, Any]]:
        ts_ns = self.market_state.get_current_ts_ns()
        universe = self.level_universe.get_levels(self.market_state, ts_ns=ts_ns)
        viewport_targets = self.viewport_manager.update(universe, self.market_state, ts_ns=ts_ns)

        stage_a_rows: List[Dict[str, Any]] = []
        stage_b_rows: List[Dict[str, Any]] = []
        metadata_by_level: Dict[str, Dict[str, Any]] = {}
        active_confirmation_keys: set[Tuple[str, int]] = set()
        pending_results: List[Dict[str, Any]] = []

        sigma_points = self._compute_sigma_points()

        for target in viewport_targets:
            stage = "stage_b" if target.state == ViewportState.CONFIRMATION else "stage_a"
            cache_key = (target.level.id, target.touch_ts_ns or 0)

            if stage == "stage_b":
                active_confirmation_keys.add(cache_key)
                cached = self._confirmation_cache.get(cache_key)
                if cached is None:
                    row = self.feature_builder.build_feature_row(
                        level=target.level,
                        market_state=self.market_state,
                        universe=universe,
                        ts_ns=target.confirm_ts_ns or ts_ns,
                        trading_date=self.trading_date,
                        historian=self.feature_historian
                    ).data
                    row["confirm_ts_ns"] = target.confirm_ts_ns
                    self._confirmation_cache[cache_key] = row
                else:
                    row = dict(cached)
            else:
                row = self.feature_builder.build_feature_row(
                    level=target.level,
                    market_state=self.market_state,
                    universe=universe,
                    ts_ns=ts_ns,
                    trading_date=self.trading_date,
                    historian=self.feature_historian
                ).data

            row["viewport_state"] = target.state.value
            row["pinned"] = target.pinned
            row["level_id"] = target.level.id
            row["stage"] = stage
            row["confirm_ts_ns"] = target.confirm_ts_ns
            row["relevance"] = target.relevance

            metadata = {
                "viewport_state": row["viewport_state"],
                "pinned": row["pinned"],
                "stage": stage,
                "touch_ts_ns": target.touch_ts_ns,
                "confirm_ts_ns": target.confirm_ts_ns,
                "relevance": target.relevance
            }
            metadata_by_level[row["level_id"]] = metadata

            inference_due = self._should_infer(row, stage, ts_ns, sigma_points)
            if inference_due:
                if stage == "stage_a":
                    self._strip_stage_b_features(row)
                    stage_a_rows.append(row)
                else:
                    stage_b_rows.append(row)
            else:
                cached = self._inference_cache.get((row["level_id"], stage))
                if cached is not None:
                    pending_results.append(self._merge_cached_result(cached, row, metadata))

        if not active_confirmation_keys:
            self._confirmation_cache.clear()
        else:
            for key in list(self._confirmation_cache.keys()):
                if key not in active_confirmation_keys:
                    del self._confirmation_cache[key]

        if not stage_a_rows and not stage_b_rows and not pending_results:
            return []

        results: List[Dict[str, Any]] = []
        if stage_a_rows:
            results.extend(self.stage_a_engine.score_targets(pd.DataFrame(stage_a_rows)))
        if stage_b_rows:
            results.extend(self.stage_b_engine.score_targets(pd.DataFrame(stage_b_rows)))

        results.extend(pending_results)

        for result in results:
            meta = metadata_by_level.get(result.get("level_id"))
            if meta:
                result.update(meta)
            stage = result.get("stage") or (meta.get("stage") if meta else "stage_a")
            level_id = result.get("level_id")
            if level_id:
                self._inference_cache[(level_id, stage)] = result

        active_ids = set(metadata_by_level.keys())
        for key in list(self._inference_cache.keys()):
            if key[0] not in active_ids:
                del self._inference_cache[key]
        for key in list(self._inference_state.keys()):
            if key not in active_ids:
                del self._inference_state[key]

        results.sort(key=lambda r: r.get("utility_score", 0.0), reverse=True)
        return results

    def _compute_sigma_points(self) -> float:
        sigma_per_min = self.market_state.get_recent_return_std(CONFIG.INFERENCE_VOL_WINDOW_SECONDS)
        if sigma_per_min is None or sigma_per_min <= 0:
            atr = self.market_state.get_atr()
            sigma_points = atr if atr is not None else CONFIG.INFERENCE_MIN_SIGMA_POINTS
        else:
            horizon_minutes = CONFIG.INFERENCE_VOL_WINDOW_SECONDS / 60.0
            sigma_points = sigma_per_min * math.sqrt(max(horizon_minutes, 1e-6))
        return max(float(sigma_points), CONFIG.INFERENCE_MIN_SIGMA_POINTS)

    def _interval_for_z(self, z: float) -> float:
        if z <= CONFIG.INFERENCE_Z_ENGAGED:
            return CONFIG.INFERENCE_INTERVAL_ENGAGED_S
        if z <= CONFIG.INFERENCE_Z_APPROACH:
            return CONFIG.INFERENCE_INTERVAL_APPROACH_S
        return CONFIG.INFERENCE_INTERVAL_FAR_S

    def _should_infer(
        self,
        row: Dict[str, Any],
        stage: str,
        ts_ns: int,
        sigma_points: float
    ) -> bool:
        level_id = row.get("level_id")
        if not level_id:
            return True

        state = self._inference_state.get(level_id, InferenceState())

        distance = row.get("distance")
        distance_signed = row.get("distance_signed")
        barrier_state = row.get("barrier_state")
        tape_imbalance = row.get("tape_imbalance")
        gamma_exposure = row.get("gamma_exposure")
        sweep_detected = row.get("sweep_detected")

        z = float(distance) / sigma_points if isinstance(distance, (int, float)) else float("inf")
        interval_s = self._interval_for_z(z)
        interval_ns = int(interval_s * 1e9)

        in_monitor = False
        if isinstance(distance, (int, float)):
            in_monitor = distance <= CONFIG.MONITOR_BAND

        stage_changed = state.last_stage is not None and stage != state.last_stage
        entered_monitor = state.last_in_monitor is not None and (not state.last_in_monitor) and in_monitor
        exited_monitor = state.last_in_monitor is not None and state.last_in_monitor and (not in_monitor)

        crossed_level = False
        if isinstance(distance_signed, (int, float)) and isinstance(state.last_distance_signed, (int, float)):
            crossed_level = (
                (state.last_distance_signed <= 0 < distance_signed)
                or (state.last_distance_signed >= 0 > distance_signed)
            )

        barrier_changed = (
            state.last_barrier_state is not None
            and barrier_state is not None
            and barrier_state != state.last_barrier_state
        )

        tape_jump = False
        if isinstance(tape_imbalance, (int, float)) and isinstance(state.last_tape_imbalance, (int, float)):
            tape_jump = abs(tape_imbalance - state.last_tape_imbalance) >= CONFIG.INFERENCE_TAPE_IMBALANCE_JUMP

        gamma_flip = False
        if isinstance(gamma_exposure, (int, float)) and isinstance(state.last_gamma_exposure, (int, float)):
            gamma_flip = (
                (state.last_gamma_exposure <= 0 < gamma_exposure)
                or (state.last_gamma_exposure >= 0 > gamma_exposure)
            ) and max(abs(state.last_gamma_exposure), abs(gamma_exposure)) >= CONFIG.INFERENCE_GAMMA_FLIP_THRESHOLD

        sweep_trigger = bool(sweep_detected) and not bool(state.last_sweep_detected)

        triggered = (
            stage_changed
            or entered_monitor
            or exited_monitor
            or crossed_level
            or barrier_changed
            or tape_jump
            or gamma_flip
            or sweep_trigger
        )

        due_by_interval = state.last_infer_ns is None or (ts_ns - state.last_infer_ns) >= interval_ns

        infer = triggered or due_by_interval or ((level_id, stage) not in self._inference_cache)

        if infer:
            state.last_infer_ns = ts_ns

        state.last_stage = stage
        state.last_distance_signed = float(distance_signed) if isinstance(distance_signed, (int, float)) else None
        state.last_in_monitor = in_monitor
        state.last_barrier_state = barrier_state
        state.last_tape_imbalance = float(tape_imbalance) if isinstance(tape_imbalance, (int, float)) else None
        state.last_gamma_exposure = float(gamma_exposure) if isinstance(gamma_exposure, (int, float)) else None
        state.last_sweep_detected = bool(sweep_detected) if sweep_detected is not None else None

        self._inference_state[level_id] = state
        return infer

    @staticmethod
    def _merge_cached_result(
        cached: Dict[str, Any],
        row: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        refreshed = dict(cached)
        for key in ("level_price", "direction", "distance", "distance_signed"):
            if key in row:
                refreshed[key] = row.get(key)
        refreshed.update(metadata)
        return refreshed

    @staticmethod
    def _strip_stage_b_features(row: Dict[str, Any]) -> None:
        extras = {"sweep_detected"}
        for key in list(row.keys()):
            if key in extras or any(key.startswith(prefix) for prefix in STAGE_B_ONLY_PREFIXES):
                value = row.get(key)
                if isinstance(value, str):
                    row[key] = "NEUTRAL"
                elif isinstance(value, bool):
                    row[key] = False
                elif isinstance(value, (int, float)):
                    row[key] = 0.0 if isinstance(value, float) else 0
                else:
                    row[key] = None
