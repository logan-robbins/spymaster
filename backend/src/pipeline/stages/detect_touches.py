"""Detect level touches stage."""
import logging
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from src.pipeline.core.stage import BaseStage, StageContext
from src.pipeline.stages.generate_levels import LevelInfo
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


def detect_touches_numpy(
    timestamps: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    levels: np.ndarray,
    tolerance: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    NumPy-based touch detection using broadcasting.

    Returns arrays of (bar_idx, level_idx, direction, distance) for all touches.
    """
    n_bars = len(timestamps)
    n_levels = len(levels)

    # Broadcasting: (n_bars, 1) vs (n_levels,) -> (n_bars, n_levels)
    lows_2d = lows[:, np.newaxis]
    highs_2d = highs[:, np.newaxis]
    closes_2d = closes[:, np.newaxis]
    levels_2d = levels[np.newaxis, :]

    # Touch mask: level within [low - tol, high + tol]
    touch_mask = (lows_2d - tolerance <= levels_2d) & (levels_2d <= highs_2d + tolerance)

    # Get indices of touches
    bar_indices, level_indices = np.where(touch_mask)

    if len(bar_indices) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int8),
            np.array([], dtype=np.float64)
        )

    # Compute direction and distance
    touch_closes = closes[bar_indices]
    touch_levels = levels[level_indices]
    directions = np.where(touch_closes < touch_levels, 1, -1).astype(np.int8)
    distances = np.abs(touch_closes - touch_levels)

    return bar_indices.astype(np.int64), level_indices.astype(np.int64), directions, distances


def detect_touches(
    ohlcv_df: pd.DataFrame,
    level_info: LevelInfo,
    touch_tolerance: float = 0.10
) -> pd.DataFrame:
    """
    Detect all level touches using numpy broadcasting.

    Args:
        ohlcv_df: OHLCV DataFrame with 1-min bars
        level_info: Level universe
        touch_tolerance: How close counts as a touch

    Returns:
        DataFrame with columns: ts_ns, bar_idx, level_price, level_kind, direction, distance, spot
    """
    if ohlcv_df.empty or len(level_info.prices) == 0:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    # Extract numpy arrays
    timestamps = ohlcv_df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    lows = ohlcv_df['low'].values.astype(np.float64)
    highs = ohlcv_df['high'].values.astype(np.float64)
    closes = ohlcv_df['close'].values.astype(np.float64)
    levels = level_info.prices

    # Use numpy broadcasting
    bar_idx, level_idx, directions, distances = detect_touches_numpy(
        timestamps, lows, highs, closes, levels, touch_tolerance
    )

    if len(bar_idx) == 0:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    # Build result DataFrame using the computed indices
    result = pd.DataFrame({
        'ts_ns': timestamps[bar_idx],
        'bar_idx': bar_idx.astype(np.int64),
        'level_price': levels[level_idx],
        'level_kind': level_info.kinds[level_idx],
        'level_kind_name': [level_info.kind_names[int(i)] for i in level_idx],
        'direction': np.where(directions == 1, 'UP', 'DOWN'),
        'distance': distances,
        'spot': closes[bar_idx]
    })

    # Deduplicate: one touch per level per minute
    result = result.drop_duplicates(subset=['ts_ns', 'level_price'])

    # Filter to only keep touches where close is near the level
    result = result[result['distance'] <= CONFIG.MONITOR_BAND]

    return result


def detect_dynamic_level_touches(
    ohlcv_df: pd.DataFrame,
    dynamic_levels: Dict[str, pd.Series],
    touch_tolerance: float = 0.10
) -> pd.DataFrame:
    """Detect touches against dynamic level series (causal)."""
    if ohlcv_df.empty or not dynamic_levels:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    df = ohlcv_df.copy()
    timestamps = df['timestamp'].values.astype('datetime64[ns]').astype(np.int64)
    lows = df['low'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)

    kind_map = {
        'PM_HIGH': 0,
        'PM_LOW': 1,
        'OR_HIGH': 2,
        'OR_LOW': 3,
        'SESSION_HIGH': 4,
        'SESSION_LOW': 5,
        'SMA_200': 6,
        'VWAP': 7,
        'CALL_WALL': 10,
        'PUT_WALL': 11,
        'SMA_400': 12
    }

    rows = []
    for kind_name, series in dynamic_levels.items():
        if kind_name not in kind_map:
            continue
        values = series.to_numpy(dtype=np.float64)
        mask = np.isfinite(values) & (lows - touch_tolerance <= values) & (values <= highs + touch_tolerance)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        level_prices = values[idx]
        direction = np.where(closes[idx] < level_prices, 'UP', 'DOWN')
        distance = np.abs(closes[idx] - level_prices)
        rows.append(pd.DataFrame({
            'ts_ns': timestamps[idx],
            'bar_idx': idx.astype(np.int64),
            'level_price': level_prices,
            'level_kind': kind_map[kind_name],
            'level_kind_name': kind_name,
            'direction': direction,
            'distance': distance,
            'spot': closes[idx]
        }))

    if not rows:
        return pd.DataFrame(columns=['ts_ns', 'bar_idx', 'level_price', 'level_kind',
                                     'level_kind_name', 'direction', 'distance', 'spot'])

    result = pd.concat(rows, ignore_index=True)
    result = result.drop_duplicates(subset=['ts_ns', 'level_kind_name', 'level_price'])
    result = result[result['distance'] <= CONFIG.MONITOR_BAND]
    return result


class DetectTouchesStage(BaseStage):
    """Detect all level touches using numpy broadcasting.

    Detects touches for both static and dynamic levels,
    then merges and deduplicates.

    Args:
        touch_tolerance: How close counts as a touch (default: 0.10)
        max_touches: Maximum touches to process (default: 5000)

    Outputs:
        touches_df: DataFrame with touch information
    """

    def __init__(
        self,
        touch_tolerance: float = 0.10,
        max_touches: int = 5000
    ):
        self.touch_tolerance = touch_tolerance
        self.max_touches = max_touches

    @property
    def name(self) -> str:
        return "detect_touches"

    @property
    def required_inputs(self) -> List[str]:
        return ['ohlcv_1min', 'static_level_info', 'dynamic_levels']

    def execute(self, ctx: StageContext) -> Dict[str, Any]:
        ohlcv_df = ctx.data['ohlcv_1min']
        static_level_info = ctx.data['static_level_info']
        dynamic_levels = ctx.data['dynamic_levels']

        logger.info(f"  Detecting touches (tolerance={self.touch_tolerance})...")
        logger.debug(f"    Static levels: {len(static_level_info.prices)}")
        logger.debug(f"    Dynamic level types: {list(dynamic_levels.keys())}")

        # Detect static level touches
        touches_df = detect_touches(
            ohlcv_df, static_level_info,
            touch_tolerance=self.touch_tolerance
        )
        static_count = len(touches_df)
        logger.info(f"    Static level touches: {static_count:,}")

        # Detect dynamic level touches
        dynamic_touches = detect_dynamic_level_touches(
            ohlcv_df, dynamic_levels,
            touch_tolerance=self.touch_tolerance
        )
        dynamic_count = len(dynamic_touches)
        logger.info(f"    Dynamic level touches: {dynamic_count:,}")

        # Merge and deduplicate
        if not dynamic_touches.empty:
            touches_df = pd.concat([touches_df, dynamic_touches], ignore_index=True)
            before_dedup = len(touches_df)
            touches_df = touches_df.drop_duplicates(
                subset=['ts_ns', 'level_kind_name', 'level_price']
            )
            logger.debug(f"    After dedup: {len(touches_df):,} (removed {before_dedup - len(touches_df):,} dups)")

        # Limit touches if needed
        if len(touches_df) > self.max_touches:
            logger.warning(f"    Limiting touches from {len(touches_df):,} to {self.max_touches:,}")
            touches_df = touches_df.head(self.max_touches)

        if touches_df.empty:
            raise ValueError("No touches detected")

        # Log level type distribution
        if not touches_df.empty:
            level_dist = touches_df['level_kind_name'].value_counts().head(5).to_dict()
            logger.info(f"    Top level types: {level_dist}")

        return {'touches_df': touches_df}
