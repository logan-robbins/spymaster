from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import math
import numpy as np

from .barrier_engine import BarrierEngine, Direction as BarrierDirection
from .tape_engine import TapeEngine
from .fuel_engine import FuelEngine
from .level_universe import Level
from .market_state import MarketState, SmaContext
from src.common.config import CONFIG
from src.ml.episode_vector import compute_dct_coefficients
from src.core.feature_historian import FeatureHistorian


@dataclass
class FeatureRow:
    data: Dict[str, Any]


class ViewportFeatureBuilder:
    """
    Builds target-relative feature rows for viewport inference.
    """

    def __init__(
        self,
        barrier_engine: BarrierEngine,
        tape_engine: TapeEngine,
        fuel_engine: FuelEngine,
        config=None
    ):
        self.config = config or CONFIG
        self.barrier_engine = barrier_engine
        self.tape_engine = tape_engine
        self.fuel_engine = fuel_engine

    def build_feature_row(
        self,
        level: Level,
        market_state: MarketState,
        universe: List[Level],
        ts_ns: Optional[int] = None,
        trading_date: Optional[str] = None,
        historian: Optional[FeatureHistorian] = None
    ) -> FeatureRow:
        ts_ns = ts_ns if ts_ns is not None else market_state.get_current_ts_ns()
        spot = market_state.get_spot()
        if spot is None:
            raise ValueError("Spot price missing for feature construction.")

        distance = abs(spot - level.price)
        direction = "UP" if spot < level.price else "DOWN"
        direction_sign = 1 if direction == "UP" else -1

        atr = market_state.get_atr()
        sma_context = market_state.get_sma_context()

        barrier_direction = BarrierDirection.RESISTANCE if direction == "UP" else BarrierDirection.SUPPORT
        barrier_metrics = self.barrier_engine.compute_barrier_state(
            level_price=level.price,
            direction=barrier_direction,
            market_state=market_state
        )
        tape_metrics = self.tape_engine.compute_tape_state(
            level_price=level.price,
            market_state=market_state
        )
        fuel_metrics = self.fuel_engine.compute_fuel_state(
            level_price=level.price,
            market_state=market_state,
            exp_date_filter=trading_date
        )
        kinematics = self._compute_kinematics(level.price, direction, market_state)

        confluence = self._compute_confluence(level, universe)

        approach_velocity, approach_bars, approach_distance = self._approach_context(
            direction, market_state
        )

        is_first_15m = self._is_first_15m(ts_ns)
        pm_high = market_state.get_premarket_high()
        pm_low = market_state.get_premarket_low()
        sma_200 = sma_context.sma_200
        sma_400 = sma_context.sma_400

        confluence_alignment = self._compute_confluence_alignment(
            level.price,
            direction,
            sma_context
        )
        mean_reversion = self._compute_mean_reversion_features(
            level.price,
            sma_context,
            market_state
        )

        gamma_exposure = fuel_metrics.net_dealer_gamma
        if gamma_exposure is None or not math.isfinite(gamma_exposure):
            gamma_bucket = "UNKNOWN"
        else:
            gamma_bucket = "SHORT_GAMMA" if gamma_exposure < 0 else "LONG_GAMMA"

        row = {
            "ts_ns": ts_ns,
            "spot": spot,
            "level_price": level.price,
            "level_kind_name": level.kind.value,
            "direction": direction,
            "direction_sign": direction_sign,
            "distance": distance,
            "distance_signed": level.price - spot,
            "atr": atr,
            "is_first_15m": is_first_15m,
            "dist_to_pm_high": (level.price - pm_high) if pm_high is not None else None,
            "dist_to_pm_low": (level.price - pm_low) if pm_low is not None else None,
            "sma_200": sma_200,
            "sma_400": sma_400,
            "dist_to_sma_200": (level.price - sma_200) if sma_200 is not None else None,
            "dist_to_sma_400": (level.price - sma_400) if sma_400 is not None else None,
            "sma_200_slope": sma_context.sma_200_slope,
            "sma_400_slope": sma_context.sma_400_slope,
            "sma_200_slope_5bar": sma_context.sma_200_slope_5bar,
            "sma_400_slope_5bar": sma_context.sma_400_slope_5bar,
            "sma_spread": sma_context.sma_spread,
        }
        # Merge kinematics (Section B)
        row.update(kinematics)

        # Continues Section B/C
        row.update({
            "approach_velocity": approach_velocity,
            "approach_bars": approach_bars,
            "approach_distance": approach_distance,
            "confluence_count": confluence["count"],
            "confluence_weighted_score": confluence["weighted_score"],
            "confluence_min_distance": confluence["min_distance"],
            "confluence_pressure": confluence["pressure"],
            "confluence_alignment": confluence_alignment,
            "barrier_state": barrier_metrics.state.value,
            "barrier_delta_liq": barrier_metrics.delta_liq,
            "barrier_replenishment_ratio": barrier_metrics.replenishment_ratio,
            "wall_ratio": barrier_metrics.wall_ratio,
            "tape_imbalance": tape_metrics.imbalance,
            "tape_buy_vol": tape_metrics.buy_vol,
            "tape_sell_vol": tape_metrics.sell_vol,
            "tape_velocity": tape_metrics.velocity,
            "sweep_detected": tape_metrics.sweep.detected,
            "fuel_effect": fuel_metrics.effect.value,
            "gamma_exposure": gamma_exposure,
            "gamma_bucket": gamma_bucket,
            "fuel_call_tide": fuel_metrics.call_tide,
            "fuel_put_tide": fuel_metrics.put_tide
        })

        # ─── SECTION F: Trajectory Basis (DCT) ───
        # Retrieve trajectories from historian and compute DCT
        if historian is not None:
             trajectories = historian.get_trajectories(level.id)
             # Map keys to series names
             # 'distance_signed_atr' -> 'd_atr'
             # 'ofi_60s' -> 'ofi_60s'
             # 'barrier_delta_liq_log' -> 'barrier_delta_liq_log'
             # 'tape_imbalance' -> 'tape_imbalance'
             
             # Name mapping for series in vector
             series_map = {
                 'd_atr': 'distance_signed_atr',
                 'ofi_60s': 'ofi_60s',
                 'barrier_delta_liq_log': 'barrier_delta_liq_log',
                 'tape_imbalance': 'tape_imbalance'
             }
             
             for name, key in series_map.items():
                 series = trajectories.get(key)
                 if series is not None:
                     coeffs = compute_dct_coefficients(series, n_coeffs=8)
                     for k in range(8):
                         row[f'dct_{name}_c{k}'] = float(coeffs[k])
                 else:
                     # Fill zeros if missing
                     for k in range(8):
                         row[f'dct_{name}_c{k}'] = 0.0
        else:
             # Fill zeros if no historian (e.g. cold start / dry run)
             for name in ['d_atr', 'ofi_60s', 'barrier_delta_liq_log', 'tape_imbalance']:
                 for k in range(8):
                     row[f'dct_{name}_c{k}'] = 0.0

        row.update(mean_reversion)
        self._apply_sparse_transforms(row)
        self._apply_normalized_features(row, spot, atr)

        return FeatureRow(data=row)

    def _compute_confluence(self, level: Level, universe: List[Level]) -> Dict[str, float]:
        key_weights = {
            "PM_HIGH": 1.0,
            "PM_LOW": 1.0,
            "OR_HIGH": 0.9,
            "OR_LOW": 0.9,
            "SMA_200": 0.8,
            "SMA_400": 0.8,
            "VWAP": 0.7,
            "SESSION_HIGH": 0.6,
            "SESSION_LOW": 0.6,
            "CALL_WALL": 1.0,
            "PUT_WALL": 1.0
        }
        band = self.config.CONFLUENCE_BAND
        distances = []
        weights = []
        for other in universe:
            if other.id == level.id:
                continue
            weight = key_weights.get(other.kind.value)
            if weight is None:
                continue
            dist = abs(other.price - level.price)
            if dist > band:
                continue
            distances.append(dist)
            weights.append(weight)

        if not distances:
            return {"count": 0, "weighted_score": 0.0, "min_distance": None, "pressure": 0.0}

        distances = list(distances)
        weights = list(weights)
        min_distance = min(distances)
        decay = [max(0.0, 1.0 - (d / band)) for d in distances]
        weighted_score = sum(w * d for w, d in zip(weights, decay))
        pressure = weighted_score / max(sum(key_weights.values()), 1e-6)
        return {
            "count": len(distances),
            "weighted_score": weighted_score,
            "min_distance": min_distance,
            "pressure": pressure
        }

    def _approach_context(self, direction: str, market_state: MarketState) -> tuple[float, int, float]:
        closes_with_ts = market_state.get_recent_minute_closes(self.config.LOOKBACK_MINUTES)
        if len(closes_with_ts) < 2:
            return 0.0, 0, 0.0
            
        # Extract prices
        closes = [c[1] for c in closes_with_ts]
        
        price_change = closes[-1] - closes[0]
        minutes = len(closes)
        if direction == "UP":
            approach_velocity = price_change / minutes
        else:
            approach_velocity = -price_change / minutes

        consecutive = 0
        for j in range(len(closes) - 1, 0, -1):
            bar_move = closes[j] - closes[j - 1]
            if direction == "UP":
                if bar_move > 0:
                    consecutive += 1
                else:
                    break
            else:
                if bar_move < 0:
                    consecutive += 1
                else:
                    break

        approach_distance = abs(closes[-1] - closes[0])
        return approach_velocity, consecutive, approach_distance

    @staticmethod
    def _is_first_15m(ts_ns: int) -> bool:
        dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).astimezone(
            ZoneInfo("America/New_York")
        )
        return (dt.hour == 9 and 30 <= dt.minute < 45)

    @staticmethod
    def _compute_confluence_alignment(
        level_price: float,
        direction: str,
        sma_context: Optional[SmaContext]
    ) -> int:
        if sma_context is None:
            return 0
        sma_200 = sma_context.sma_200
        sma_400 = sma_context.sma_400
        sma_200_slope = sma_context.sma_200_slope
        sma_400_slope = sma_context.sma_400_slope

        if (
            sma_200 is None
            or sma_400 is None
            or sma_200_slope is None
            or sma_400_slope is None
        ):
            return 0

        below_both = level_price < sma_200 and level_price < sma_400
        above_both = level_price > sma_200 and level_price > sma_400
        slopes_negative = sma_200_slope < 0 and sma_400_slope < 0
        slopes_positive = sma_200_slope > 0 and sma_400_slope > 0
        spread_slope = sma_200_slope - sma_400_slope

        if direction == "UP":
            if below_both and slopes_negative and spread_slope > 0:
                return 1
            if above_both and slopes_positive and spread_slope < 0:
                return -1
        else:
            if above_both and slopes_positive and spread_slope > 0:
                return 1
            if below_both and slopes_negative and spread_slope < 0:
                return -1
        return 0

    def _compute_mean_reversion_features(
        self,
        level_price: float,
        sma_context: Optional[SmaContext],
        market_state: MarketState
    ) -> Dict[str, Optional[float]]:
        if sma_context is None:
            return {
                "mean_reversion_pressure_200": None,
                "mean_reversion_pressure_400": None,
                "mean_reversion_velocity_200": None,
                "mean_reversion_velocity_400": None
            }

        dist_to_sma_200 = None
        dist_to_sma_400 = None
        if sma_context.sma_200 is not None:
            dist_to_sma_200 = level_price - sma_context.sma_200
        if sma_context.sma_400 is not None:
            dist_to_sma_400 = level_price - sma_context.sma_400

        volatility = self._compute_recent_volatility(market_state)
        eps = 1e-6
        mean_reversion_pressure_200 = None
        mean_reversion_pressure_400 = None
        if volatility is not None and volatility > 0:
            if dist_to_sma_200 is not None:
                mean_reversion_pressure_200 = dist_to_sma_200 / (volatility + eps)
            if dist_to_sma_400 is not None:
                mean_reversion_pressure_400 = dist_to_sma_400 / (volatility + eps)

        vel_minutes = max(1, self.config.MEAN_REVERSION_VELOCITY_WINDOW_MINUTES)
        vel_bars = max(1, int(round(vel_minutes / 2)))

        mean_reversion_velocity_200 = None
        mean_reversion_velocity_400 = None
        prev_sma_200 = market_state.get_sma_at_offset(200, vel_bars)
        prev_sma_400 = market_state.get_sma_at_offset(400, vel_bars)

        if dist_to_sma_200 is not None and prev_sma_200 is not None:
            prev_dist_200 = level_price - prev_sma_200
            mean_reversion_velocity_200 = (dist_to_sma_200 - prev_dist_200) / vel_minutes
        if dist_to_sma_400 is not None and prev_sma_400 is not None:
            prev_dist_400 = level_price - prev_sma_400
            mean_reversion_velocity_400 = (dist_to_sma_400 - prev_dist_400) / vel_minutes

        return {
            "mean_reversion_pressure_200": mean_reversion_pressure_200,
            "mean_reversion_pressure_400": mean_reversion_pressure_400,
            "mean_reversion_velocity_200": mean_reversion_velocity_200,
            "mean_reversion_velocity_400": mean_reversion_velocity_400
        }

    def _compute_recent_volatility(self, market_state: MarketState) -> Optional[float]:
        window = max(1, self.config.MEAN_REVERSION_VOL_WINDOW_MINUTES)
        closes = market_state.get_recent_minute_closes(window + 1)
        if len(closes) < 2:
            return None
        
        # Extract prices
        prices = [c[1] for c in closes]

        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if not changes:
            return None
        if len(changes) > window:
            changes = changes[-window:]
        mean = sum(changes) / len(changes)
        variance = sum((val - mean) ** 2 for val in changes) / len(changes)
        return math.sqrt(variance)

    @staticmethod
    def _apply_sparse_transforms(row: Dict[str, Any]) -> None:
        for col in ("wall_ratio", "barrier_delta_liq", "fuel_call_tide", "fuel_put_tide"):
            value = row.get(col)
            if value is None or not math.isfinite(value):
                row[f"{col}_nonzero"] = None
                row[f"{col}_log"] = None
                continue
            row[f"{col}_nonzero"] = 1 if value != 0 else 0
            row[f"{col}_log"] = math.copysign(math.log1p(abs(value)), value) if value != 0 else 0.0

    @staticmethod
    def _apply_normalized_features(row: Dict[str, Any], spot: float, atr: Optional[float]) -> None:
        eps = 1e-6
        distance_cols = [
            "distance",
            "distance_signed",
            "dist_to_pm_high",
            "dist_to_pm_low",
            "dist_to_sma_200",
            "dist_to_sma_400",
            "confluence_min_distance",
            "approach_distance"
        ]
        for col in distance_cols:
            if col not in row:
                continue
            value = row.get(col)
            if value is None:
                row[f"{col}_atr"] = None
                row[f"{col}_pct"] = None
                continue
            if atr is None:
                row[f"{col}_atr"] = None
            else:
                row[f"{col}_atr"] = value / (atr + eps)
            row[f"{col}_pct"] = value / (spot + eps)


        row["level_price_pct"] = (row["level_price"] - spot) / (spot + eps)

    def _compute_kinematics(self, level_price: float, direction: str, market_state: MarketState) -> Dict[str, float]:
        """
        Compute multi-window kinematics (Velocity, Accel, Jerk) for Section B.
        Windows: 1, 2, 3, 5, 10, 20 min.
        """
        # Get history for max window (20m) -> 21 points for 20m diffs
        history = market_state.get_recent_minute_closes(lookback_minutes=21)
        
        if not history:
             return self._empty_kinematics()

        ts_array = np.array([h[0] for h in history], dtype=np.float64) / 1e9
        price_array = np.array([h[1] for h in history], dtype=np.float64)
        
        # Direction sign for level-frame coordinates
        dir_sign = 1 if direction == "UP" else -1
        
        p_series = dir_sign * (price_array - level_price)
        
        results = {}
        windows = [1, 2, 3, 5, 10, 20]
        
        current_ts = ts_array[-1]
        
        for w in windows:
            # Filter history to window
            window_seconds = w * 60
            start_ts = current_ts - window_seconds
            
            mask = ts_array >= (start_ts - 0.1) # Tolerance
            
            t_win = ts_array[mask]
            p_win = p_series[mask]
            
            if len(t_win) < 2:
                results[f'velocity_{w}min'] = 0.0
                results[f'acceleration_{w}min'] = 0.0
                results[f'jerk_{w}min'] = 0.0
                continue
                
            # Normalize t to start at 0
            t_norm = t_win - t_win[0]
            
            # Velocity: Linear Slope
            if len(t_norm) >= 2:
                 coeffs_v = np.polyfit(t_norm, p_win, 1)
                 results[f'velocity_{w}min'] = float(coeffs_v[0])
            else:
                 results[f'velocity_{w}min'] = 0.0
                 
            # Acceleration: Quadratic (d=2)
            if len(t_norm) >= 3:
                 coeffs_a = np.polyfit(t_norm, p_win, 2)
                 results[f'acceleration_{w}min'] = float(2.0 * coeffs_a[0])
            else:
                 results[f'acceleration_{w}min'] = 0.0
                 
            # Jerk: Cubic (d=3)
            if len(t_norm) >= 4:
                 coeffs_j = np.polyfit(t_norm, p_win, 3)
                 results[f'jerk_{w}min'] = float(6.0 * coeffs_j[0])
            else:
                 results[f'jerk_{w}min'] = 0.0
                 
        for w in [3, 5, 10, 20]:
            results[f'momentum_trend_{w}min'] = results.get(f'acceleration_{w}min', 0.0)
            
        return results

    def _empty_kinematics(self) -> Dict[str, float]:
        res = {}
        for w in [1, 2, 3, 5, 10, 20]:
            res[f'velocity_{w}min'] = 0.0
            res[f'acceleration_{w}min'] = 0.0
            res[f'jerk_{w}min'] = 0.0
        for w in [3, 5, 10, 20]:
             res[f'momentum_trend_{w}min'] = 0.0
        return res
