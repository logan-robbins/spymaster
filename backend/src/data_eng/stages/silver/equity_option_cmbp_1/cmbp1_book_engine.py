from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

PRICE_SCALE = 1e-9
WINDOW_NS = 1_000_000_000

SIDE_ASK = "A"
SIDE_BID = "B"


@dataclass
class Cmbp1InstrumentState:
    bid_price_int: int = 0
    bid_size: int = 0
    ask_price_int: int = 0
    ask_size: int = 0


class Cmbp1BookEngine:
    def __init__(self, window_ns: int = WINDOW_NS) -> None:
        self.window_ns = window_ns

        self.curr_window_id: int | None = None
        self.window_start_ts = 0
        self.window_end_ts = 0

        self.state: Dict[int, Cmbp1InstrumentState] = {}
        self.acc_add: Dict[Tuple[int, str, int], float] = {}
        self.acc_pull: Dict[Tuple[int, str, int], float] = {}

        self.flow_rows: List[Dict[str, object]] = []
        self.bbo_rows: List[Dict[str, object]] = []

    def process_batch(self, df_cmbp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if df_cmbp.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = df_cmbp.copy()
        if "ts_recv" not in df.columns:
            df["ts_recv"] = df["ts_event"]

        df = df.sort_values(["ts_event", "ts_recv"])

        required = {"ts_event", "instrument_id", "bid_px_00", "bid_sz_00", "ask_px_00", "ask_sz_00"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required CMBP-1 columns: {sorted(missing)}")

        for row in df.itertuples(index=False):
            ts = int(row.ts_event)
            window_id = ts // self.window_ns
            if self.curr_window_id is None:
                self._start_window(window_id)
            elif window_id > self.curr_window_id:
                self._flush_until(window_id)

            iid = int(row.instrument_id)
            bid_px = int(row.bid_px_00)
            bid_sz = int(row.bid_sz_00)
            ask_px = int(row.ask_px_00)
            ask_sz = int(row.ask_sz_00)

            self._apply_update(iid, bid_px, bid_sz, ask_px, ask_sz)

        self.flush_final()

        df_flow = pd.DataFrame(self.flow_rows) if self.flow_rows else pd.DataFrame()
        df_bbo = pd.DataFrame(self.bbo_rows) if self.bbo_rows else pd.DataFrame()
        return df_flow, df_bbo

    def flush_final(self) -> None:
        if self.curr_window_id is None:
            return
        self._emit_window()

    def _start_window(self, window_id: int) -> None:
        self.curr_window_id = window_id
        self.window_start_ts = window_id * self.window_ns
        self.window_end_ts = self.window_start_ts + self.window_ns
        self._reset_accumulators()

    def _flush_until(self, target_window_id: int) -> None:
        while self.curr_window_id is not None and self.curr_window_id < target_window_id:
            self._emit_window()
            next_window = self.curr_window_id + 1
            self._start_window(next_window)

    def _emit_window(self) -> None:
        active_keys = set(self.acc_add.keys()) | set(self.acc_pull.keys())

        for iid, state in self.state.items():
            if state.bid_price_int > 0 and state.bid_size > 0:
                active_keys.add((iid, SIDE_BID, state.bid_price_int))
            if state.ask_price_int > 0 and state.ask_size > 0:
                active_keys.add((iid, SIDE_ASK, state.ask_price_int))

        for iid, side, price in sorted(active_keys):
            depth_total = 0.0
            state = self.state.get(iid)
            if state is not None:
                if side == SIDE_BID and price == state.bid_price_int:
                    depth_total = float(state.bid_size)
                elif side == SIDE_ASK and price == state.ask_price_int:
                    depth_total = float(state.ask_size)

            add_qty = float(self.acc_add.get((iid, side, price), 0.0))
            pull_qty = float(self.acc_pull.get((iid, side, price), 0.0))

            if depth_total <= 0 and add_qty == 0.0 and pull_qty == 0.0:
                continue

            self.flow_rows.append(
                {
                    "window_end_ts_ns": self.window_end_ts,
                    "instrument_id": iid,
                    "side": side,
                    "price_int": price,
                    "depth_total": depth_total,
                    "add_qty": add_qty,
                    "pull_qty": pull_qty,
                    "pull_rest_qty": 0.0,
                    "fill_qty": 0.0,
                }
            )

        for iid, state in self.state.items():
            if state.bid_price_int > 0 and state.ask_price_int > 0 and state.ask_price_int > state.bid_price_int:
                mid = (state.bid_price_int + state.ask_price_int) * 0.5
                self.bbo_rows.append(
                    {
                        "window_end_ts_ns": self.window_end_ts,
                        "instrument_id": iid,
                        "bid_price_int": state.bid_price_int,
                        "ask_price_int": state.ask_price_int,
                        "mid_price_int": mid,
                    }
                )

        self._reset_accumulators()

    def _apply_update(self, iid: int, bid_px: int, bid_sz: int, ask_px: int, ask_sz: int) -> None:
        state = self.state.get(iid)
        if state is None:
            state = Cmbp1InstrumentState()
            self.state[iid] = state

        self._apply_side_update(
            iid,
            SIDE_BID,
            state.bid_price_int,
            state.bid_size,
            bid_px,
            bid_sz,
        )
        self._apply_side_update(
            iid,
            SIDE_ASK,
            state.ask_price_int,
            state.ask_size,
            ask_px,
            ask_sz,
        )

        state.bid_price_int = bid_px
        state.bid_size = max(bid_sz, 0)
        state.ask_price_int = ask_px
        state.ask_size = max(ask_sz, 0)

    def _apply_side_update(
        self,
        iid: int,
        side: str,
        old_price: int,
        old_size: int,
        new_price: int,
        new_size: int,
    ) -> None:
        old_size = max(old_size, 0)
        new_size = max(new_size, 0)

        if old_price <= 0 or old_size <= 0:
            if new_price > 0 and new_size > 0:
                self._accumulate(self.acc_add, (iid, side, new_price), float(new_size))
            return

        if new_price <= 0 or new_size <= 0:
            self._accumulate(self.acc_pull, (iid, side, old_price), float(old_size))
            return

        if new_price == old_price:
            delta = new_size - old_size
            if delta > 0:
                self._accumulate(self.acc_add, (iid, side, new_price), float(delta))
            elif delta < 0:
                self._accumulate(self.acc_pull, (iid, side, old_price), float(-delta))
            return

        self._accumulate(self.acc_pull, (iid, side, old_price), float(old_size))
        self._accumulate(self.acc_add, (iid, side, new_price), float(new_size))

    @staticmethod
    def _accumulate(target: Dict[Tuple[int, str, int], float], key: Tuple[int, str, int], value: float) -> None:
        target[key] = target.get(key, 0.0) + value
