from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON, POINT

if TYPE_CHECKING:
    from .compute import TickArrays


class BookState:

    __slots__ = ("bid_px", "ask_px", "bid_sz", "ask_sz", "bid_ct", "ask_ct")

    def __init__(self) -> None:
        self.bid_px: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        self.ask_px: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        self.bid_sz: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        self.ask_sz: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        self.bid_ct: NDArray[np.float64] = np.zeros(10, dtype=np.float64)
        self.ask_ct: NDArray[np.float64] = np.zeros(10, dtype=np.float64)

    def copy_from(self, other: BookState) -> None:
        self.bid_px[:] = other.bid_px
        self.ask_px[:] = other.ask_px
        self.bid_sz[:] = other.bid_sz
        self.ask_sz[:] = other.ask_sz
        self.bid_ct[:] = other.bid_ct
        self.ask_ct[:] = other.ask_ct

    def load_from_arrays(self, ticks: TickArrays, idx: int) -> None:
        self.bid_px[:] = ticks.bid_px[idx]
        self.ask_px[:] = ticks.ask_px[idx]
        self.bid_sz[:] = ticks.bid_sz[idx]
        self.ask_sz[:] = ticks.ask_sz[idx]
        self.bid_ct[:] = ticks.bid_ct[idx]
        self.ask_ct[:] = ticks.ask_ct[idx]

    def compute_microprice(self) -> float:
        b0_px = self.bid_px[0]
        a0_px = self.ask_px[0]
        b0_sz = self.bid_sz[0]
        a0_sz = self.ask_sz[0]

        total_sz = b0_sz + a0_sz
        if total_sz < EPSILON:
            return (a0_px + b0_px) / 2.0
        return (a0_px * b0_sz + b0_px * a0_sz) / total_sz

    def compute_spread_pts(self) -> float:
        return (self.ask_px[0] - self.bid_px[0]) / POINT

    def compute_obi0(self) -> float:
        b0_sz = self.bid_sz[0]
        a0_sz = self.ask_sz[0]
        denom = b0_sz + a0_sz + EPSILON
        return (b0_sz - a0_sz) / denom

    def compute_obi10(self) -> float:
        bid_depth = self.bid_sz.sum()
        ask_depth = self.ask_sz.sum()
        denom = bid_depth + ask_depth + EPSILON
        return (bid_depth - ask_depth) / denom

    def compute_total_depth(self) -> tuple[float, float]:
        return float(self.bid_sz.sum()), float(self.ask_sz.sum())
