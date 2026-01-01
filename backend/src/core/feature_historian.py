"""
Feature Historian: Real-time maintenance of level feature history for DCT Vector construction.

Fills the gap between real-time scalar states and the 20-minute trajectory history 
required by the Retrieval Engine (Phase 4/5).

Maintains:
- 40-step history (20 minutes @ 30s cadence) for every active level.
- Global metrics (OFI, ATR, Spot).
- Per-level metrics (Barrier Delta Liq, Tape Imbalance, Distance).

Usage:
    historian = FeatureHistorian(market_state)
    historian.update_loop(ts_ns, active_levels, barrier_engine, tape_engine)
    
    # Retrieve vector-ready trajectory arrays
    trajectories = historian.get_trajectories(level_id)
"""

import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Deque, Tuple
import numpy as np

from .market_state import MarketState
from .level_universe import Level
from src.common.event_types import MBP10
from src.common.config import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class LevelHistorySnapshot:
    """Snapshot of level-specific metrics at a single timestamp."""
    ts_ns: int
    distance_signed: float
    barrier_delta_liq: float
    tape_imbalance: float
    # We store the global metrics here too for easy alignment/retrieval per level
    ofi_60s: float
    spot: float
    atr: float


class FeatureHistorian:
    """
    Maintains a rolling buffer of computed features for all active levels.
    Used to construct the 'Section F: Trajectory Basis' of the Episode Vector.
    """

    def __init__(self, market_state: MarketState, history_size: int = 40):
        self.market_state = market_state
        self.history_size = history_size  # Default 40 samples (20 mins @ 30s)
        
        # Key: level_id -> Deque[LevelHistorySnapshot]
        self.histories: Dict[str, Deque[LevelHistorySnapshot]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        
        self.last_update_ts_ns = 0

    def update(
        self,
        ts_ns: int,
        active_levels: List[Level],
        barrier_engine,
        tape_engine,
        force: bool = False
    ):
        """
        Update history for all active levels.
        Should be called loosely every 30 seconds.
        """
        # Enforce 30s cadence (approx)
        if not force and (ts_ns - self.last_update_ts_ns) < 29 * 1e9:
            return

        self.last_update_ts_ns = ts_ns
        
        # Compute global metrics once
        spot = self.market_state.get_spot() or 0.0
        atr = self.market_state.get_atr() or 1.0
        ofi_60s = self._compute_global_ofi_60s(ts_ns)

        for level in active_levels:
            # 1. Barrier
            # BarrierEngine needs 'direction' but delta_liq is roughly direction-agnostic
            # explicitly? No, compute_barrier_state requires direction.
            # We can infer direction from spot relative to level.
            if spot < level.price:
                direction = "RESISTANCE" 
            else:
                direction = "SUPPORT"
                
            barrier = barrier_engine.compute_barrier_state(
                level_price=level.price,
                direction=direction,
                market_state=self.market_state
            ) # Returns BarrierMetrics

            # 2. Tape (Imbalance)
            tape = tape_engine.compute_tape_state(
                level_price=level.price,
                market_state=self.market_state
            ) # Returns TapeMetrics
            
            snapshot = LevelHistorySnapshot(
                ts_ns=ts_ns,
                distance_signed=spot - level.price,
                barrier_delta_liq=barrier.delta_liq,
                tape_imbalance=tape.imbalance,
                ofi_60s=ofi_60s,
                spot=spot,
                atr=atr
            )
            
            self.histories[level.id].append(snapshot)
            
        # Cleanup inactive levels? maybe once in a while. 
        # For now, memory usage is low (128GB RAM machine).
        # We can implement cleanup if thousands of levels accumulate.

    def get_trajectories(self, level_id: str) -> Dict[str, np.ndarray]:
        """
        Get the 4 trajectory series required for DCT computation (Section F).
        Returns numpy arrays of length 40 (padded with zeros if insufficient history).
        
        Keys: 'distance_signed_atr', 'ofi_60s', 'barrier_delta_liq_log', 'tape_imbalance'
        """
        history = list(self.histories.get(level_id, []))
        
        length = self.history_size
        
        # Prepare arrays
        d_atr_series = np.zeros(length, dtype=np.float32)
        ofi_series = np.zeros(length, dtype=np.float32)
        barrier_series = np.zeros(length, dtype=np.float32)
        tape_series = np.zeros(length, dtype=np.float32)
        
        if not history:
            return {
                'distance_signed_atr': d_atr_series,
                'ofi_60s': ofi_series,
                'barrier_delta_liq_log': barrier_series,
                'tape_imbalance': tape_series
            }
            
        # Fill from end (latest data at right)
        # Actually episode_vector extracts it as a list and computes DCT.
        # Order matters! DCT expects time series [t-N, ... t].
        # History is deque [oldest ... newest].
        
        # We align to the end of the array
        valid_len = len(history)
        start_idx = length - valid_len
        
        for i, snap in enumerate(history):
            idx = start_idx + i
            
            # 1. d_atr
            # Avoid division by zero
            safe_atr = snap.atr if snap.atr is not None and snap.atr > 0 else 1.0
            d_atr_series[idx] = snap.distance_signed / safe_atr
            
            # 2. ofi_60s
            ofi_series[idx] = snap.ofi_60s
            
            # 3. barrier_delta_liq_log (Signed Log)
            val = snap.barrier_delta_liq
            barrier_series[idx] = np.log1p(abs(val)) * np.sign(val)
            
            # 4. tape_imbalance
            tape_series[idx] = snap.tape_imbalance
            
        return {
            'distance_signed_atr': d_atr_series,
            'ofi_60s': ofi_series,
            'barrier_delta_liq_log': barrier_series,
            'tape_imbalance': tape_series
        }

    def _compute_global_ofi_60s(self, ts_now_ns: int) -> float:
        """
        Compute Order Flow Imbalance (OFI) over the last 60 seconds.
        Uses raw MBP-10 snapshots from MarketState buffer.
        """
        snapshots = self.market_state.get_es_mbp10_in_window(ts_now_ns, 60.0)
        if len(snapshots) < 2:
            return 0.0
            
        ofi = 0.0
        
        # Iterate consecutive snapshots
        for i in range(1, len(snapshots)):
            prev = snapshots[i-1]
            curr = snapshots[i]
            
            # Top-1 OFI (most significant)
            if not prev.levels or not curr.levels:
                continue
                
            prev_bid = prev.levels[0]
            curr_bid = curr.levels[0]
            prev_ask = prev.levels[0]
            curr_ask = curr.levels[0]
            
            # Bid Contribution
            e_bid = 0.0
            if curr_bid.bid_px > prev_bid.bid_px:
                e_bid = curr_bid.bid_sz
            elif curr_bid.bid_px < prev_bid.bid_px:
                e_bid = -prev_bid.bid_sz
            else:
                e_bid = curr_bid.bid_sz - prev_bid.bid_sz
                
            # Ask Contribution
            e_ask = 0.0
            if curr_ask.ask_px > prev_ask.ask_px:
                e_ask = -prev_ask.ask_sz
            elif curr_ask.ask_px < prev_ask.ask_px:
                e_ask = curr_ask.ask_sz
            else:
                e_ask = curr_ask.ask_sz - prev_ask.ask_sz
                
            # Net OFI step (Bid - Ask) -> No, standard formula is e_bid - e_ask
            ofi += (e_bid - e_ask)
            
        return ofi
