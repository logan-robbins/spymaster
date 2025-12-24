from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .level_universe import LevelUniverse
from .market_state import MarketState
from .viewport_manager import ViewportManager, ViewportState
from .viewport_feature_builder import ViewportFeatureBuilder
from .inference_engine import ViewportInferenceEngine
from src.ml.feature_sets import STAGE_B_ONLY_PREFIXES


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
        trading_date: Optional[str] = None
    ):
        self.market_state = market_state
        self.level_universe = level_universe
        self.viewport_manager = viewport_manager
        self.feature_builder = feature_builder
        self.stage_a_engine = stage_a_engine
        self.stage_b_engine = stage_b_engine
        self.trading_date = trading_date
        self._confirmation_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def score_viewport(self) -> List[Dict[str, Any]]:
        ts_ns = self.market_state.get_current_ts_ns()
        universe = self.level_universe.get_levels(self.market_state, ts_ns=ts_ns)
        viewport_targets = self.viewport_manager.update(universe, self.market_state, ts_ns=ts_ns)

        stage_a_rows: List[Dict[str, Any]] = []
        stage_b_rows: List[Dict[str, Any]] = []
        metadata_by_level: Dict[str, Dict[str, Any]] = {}
        active_confirmation_keys: set[Tuple[str, int]] = set()
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
                        trading_date=self.trading_date
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
                    trading_date=self.trading_date
                ).data

            row["viewport_state"] = target.state.value
            row["pinned"] = target.pinned
            row["level_id"] = target.level.id
            row["stage"] = stage
            row["confirm_ts_ns"] = target.confirm_ts_ns
            row["relevance"] = target.relevance

            metadata_by_level[row["level_id"]] = {
                "viewport_state": row["viewport_state"],
                "pinned": row["pinned"],
                "stage": stage,
                "touch_ts_ns": target.touch_ts_ns,
                "confirm_ts_ns": target.confirm_ts_ns,
                "relevance": target.relevance
            }

            if stage == "stage_a":
                self._strip_stage_b_features(row)
                stage_a_rows.append(row)
            else:
                stage_b_rows.append(row)

        if not active_confirmation_keys:
            self._confirmation_cache.clear()
        else:
            for key in list(self._confirmation_cache.keys()):
                if key not in active_confirmation_keys:
                    del self._confirmation_cache[key]

        if not stage_a_rows and not stage_b_rows:
            return []

        results: List[Dict[str, Any]] = []
        if stage_a_rows:
            results.extend(self.stage_a_engine.score_targets(pd.DataFrame(stage_a_rows)))
        if stage_b_rows:
            results.extend(self.stage_b_engine.score_targets(pd.DataFrame(stage_b_rows)))

        for result in results:
            meta = metadata_by_level.get(result.get("level_id"))
            if meta:
                result.update(meta)

        results.sort(key=lambda r: r.get("utility_score", 0.0), reverse=True)
        return results

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
