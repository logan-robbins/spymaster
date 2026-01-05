from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import EPSILON, POINT


class BookState:

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

    def load_from_row(self, row: dict) -> None:
        for i in range(10):
            idx = f"{i:02d}"
            self.bid_px[i] = float(row[f"bid_px_{idx}"]) / 1e9
            self.ask_px[i] = float(row[f"ask_px_{idx}"]) / 1e9
            self.bid_sz[i] = max(0.0, float(row[f"bid_sz_{idx}"]))
            self.ask_sz[i] = max(0.0, float(row[f"ask_sz_{idx}"]))
            self.bid_ct[i] = max(0.0, float(row[f"bid_ct_{idx}"]))
            self.ask_ct[i] = max(0.0, float(row[f"ask_ct_{idx}"]))

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
        return self.bid_sz.sum(), self.ask_sz.sum()

