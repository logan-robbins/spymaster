from __future__ import annotations

import numpy as np

from .constants import BANDS, BAR_DURATION_NS, EPSILON


class BarAccumulator:

    def __init__(self, bar_start_ns: int) -> None:
        self.bar_start_ns = bar_start_ns
        self.bar_end_ns = bar_start_ns + BAR_DURATION_NS
        
        self.t_last = bar_start_ns
        
        self.meta_msg_cnt = 0.0
        self.meta_clear_cnt = 0.0
        self.meta_add_cnt = 0.0
        self.meta_cancel_cnt = 0.0
        self.meta_modify_cnt = 0.0
        self.meta_trade_cnt = 0.0
        
        self.twa_spread_pts = 0.0
        self.twa_obi0 = 0.0
        self.twa_obi10 = 0.0
        self.twa_cdi = {band: 0.0 for band in BANDS}
        
        self.eob_spread_pts = 0.0
        self.eob_obi0 = 0.0
        self.eob_obi10 = 0.0
        self.eob_cdi = {band: 0.0 for band in BANDS}
        
        self.twa_bid10_qty = 0.0
        self.twa_ask10_qty = 0.0
        self.eob_bid10_qty = 0.0
        self.eob_ask10_qty = 0.0
        
        self.twa_below_qty = {band: 0.0 for band in BANDS}
        self.eob_below_qty = {band: 0.0 for band in BANDS}
        self.twa_above_qty = {band: 0.0 for band in BANDS}
        self.eob_above_qty = {band: 0.0 for band in BANDS}
        
        self.twa_below_frac = {band: 0.0 for band in BANDS}
        self.eob_below_frac = {band: 0.0 for band in BANDS}
        self.twa_above_frac = {band: 0.0 for band in BANDS}
        self.eob_above_frac = {band: 0.0 for band in BANDS}
        
        self.eob_bid_sz = np.zeros(10, dtype=np.float64)
        self.eob_ask_sz = np.zeros(10, dtype=np.float64)
        self.eob_bid_ct = np.zeros(10, dtype=np.float64)
        self.eob_ask_ct = np.zeros(10, dtype=np.float64)
        
        self.eob_bid_px = np.zeros(10, dtype=np.float64)
        self.eob_ask_px = np.zeros(10, dtype=np.float64)
        
        self.flow_add_vol = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_add_vol.update({f"ask_{band}": 0.0 for band in BANDS})
        self.flow_rem_vol = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_rem_vol.update({f"ask_{band}": 0.0 for band in BANDS})
        self.flow_net_vol = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_net_vol.update({f"ask_{band}": 0.0 for band in BANDS})
        
        self.flow_cnt_add = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_cnt_add.update({f"ask_{band}": 0.0 for band in BANDS})
        self.flow_cnt_cancel = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_cnt_cancel.update({f"ask_{band}": 0.0 for band in BANDS})
        self.flow_cnt_modify = {f"bid_{band}": 0.0 for band in BANDS}
        self.flow_cnt_modify.update({f"ask_{band}": 0.0 for band in BANDS})
        
        self.trade_cnt = 0.0
        self.trade_vol = 0.0
        self.trade_aggbuy_vol = 0.0
        self.trade_aggsell_vol = 0.0
        
    def finalize_twa(self, t_final: int) -> None:
        dt_end = t_final - self.t_last
        if dt_end > 0:
            pass
        
        duration = float(BAR_DURATION_NS)
        self.twa_spread_pts /= duration
        self.twa_obi0 /= duration
        self.twa_obi10 /= duration
        for band in BANDS:
            self.twa_cdi[band] /= duration
        
        self.twa_bid10_qty /= duration
        self.twa_ask10_qty /= duration
        for band in BANDS:
            self.twa_below_qty[band] /= duration
            self.twa_above_qty[band] /= duration
            self.twa_below_frac[band] /= duration
            self.twa_above_frac[band] /= duration
    
    def accumulate_twa_state(self, dt_ns: int, spread_pts: float, obi0: float, obi10: float, 
                            cdi: dict[str, float], bid10_qty: float, ask10_qty: float,
                            below_qty: dict[str, float], above_qty: dict[str, float],
                            below_frac: dict[str, float], above_frac: dict[str, float]) -> None:
        dt = float(dt_ns)
        self.twa_spread_pts += spread_pts * dt
        self.twa_obi0 += obi0 * dt
        self.twa_obi10 += obi10 * dt
        for band in BANDS:
            self.twa_cdi[band] += cdi[band] * dt
        
        self.twa_bid10_qty += bid10_qty * dt
        self.twa_ask10_qty += ask10_qty * dt
        for band in BANDS:
            self.twa_below_qty[band] += below_qty[band] * dt
            self.twa_above_qty[band] += above_qty[band] * dt
            self.twa_below_frac[band] += below_frac[band] * dt
            self.twa_above_frac[band] += above_frac[band] * dt

