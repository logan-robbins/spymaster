#!/usr/bin/env python3
"""
Institution-Grade Silver Layer Audit for FUTURE_MBO

A thorough semantic and statistical audit of the silver layer output.
Covers all fields in book_snapshot_1s and depth_and_flow_1s tables.

Last Grunted: 02/01/2026 11:00:00 PM PT
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import numpy as np

# ==============================================================================
# Constants (from book_engine.py)
# ==============================================================================
PRICE_SCALE = 1e-9
TICK_SIZE = 0.25
TICK_INT = int(round(TICK_SIZE / PRICE_SCALE))  # 250,000,000
WINDOW_NS = 1_000_000_000  # 1 second in nanoseconds
REST_NS = 500_000_000  # 500ms resting threshold
GRID_MAX_TICKS = 200  # +/- 200 ticks = +/- $50 range for ES

LAKE_ROOT = Path(__file__).parent.parent / "lake"


# ==============================================================================
# Data Classes for Audit Results
# ==============================================================================
@dataclass
class AuditResult:
    """Container for audit findings."""
    table: str
    field: str
    check: str
    status: str  # PASS, WARN, FAIL, INFO
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """Aggregate audit report."""
    symbol: str
    dt: str
    results: list = field(default_factory=list)
    grade: str = ""
    
    def add(self, result: AuditResult):
        self.results.append(result)
    
    def summary(self) -> dict:
        counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts


# ==============================================================================
# Utility Functions
# ==============================================================================
def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_float(f: float, decimals: int = 4) -> str:
    return f"{f:,.{decimals}f}"


def fmt_pct(pct: float) -> str:
    return f"{pct:.2f}%"


# ==============================================================================
# TASK 1: Load and Inspect Data
# ==============================================================================
def load_parquet(symbol: str, dt: str, table: str) -> pd.DataFrame | None:
    """Load silver layer parquet file."""
    path = LAKE_ROOT / f"silver/product_type=future_mbo/symbol={symbol}/table={table}/dt={dt}"
    parquet_files = list(path.glob("*.parquet"))
    if not parquet_files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    return df


def describe_schema(df: pd.DataFrame, table: str) -> list[AuditResult]:
    """Describe schema, row counts, and basic stats."""
    results = []
    
    # Row count
    results.append(AuditResult(
        table=table,
        field="*",
        check="row_count",
        status="INFO",
        message=f"Total rows: {fmt_int(len(df))}",
        details={"row_count": len(df)}
    ))
    
    # Column count
    results.append(AuditResult(
        table=table,
        field="*",
        check="column_count",
        status="INFO",
        message=f"Total columns: {len(df.columns)}",
        details={"columns": list(df.columns)}
    ))
    
    # Data types
    dtype_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    results.append(AuditResult(
        table=table,
        field="*",
        check="dtypes",
        status="INFO",
        message=f"Schema: {dtype_info}",
        details={"dtypes": dtype_info}
    ))
    
    # Memory usage
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    results.append(AuditResult(
        table=table,
        field="*",
        check="memory",
        status="INFO",
        message=f"Memory usage: {mem_mb:.2f} MB",
        details={"memory_mb": mem_mb}
    ))
    
    return results


# ==============================================================================
# TASK 2: Semantic Analysis (Every Field)
# ==============================================================================
def semantic_book_snapshot(df: pd.DataFrame) -> list[AuditResult]:
    """Semantic analysis of book_snapshot_1s fields."""
    results = []
    
    # -------------------------------------------------------------------------
    # window_start_ts_ns, window_end_ts_ns: Time boundaries (nanoseconds since epoch)
    # -------------------------------------------------------------------------
    # Check window duration is exactly 1 second
    window_duration = df["window_end_ts_ns"] - df["window_start_ts_ns"]
    duration_ok = (window_duration == WINDOW_NS).all()
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="window_start_ts_ns, window_end_ts_ns",
        check="window_duration",
        status="PASS" if duration_ok else "FAIL",
        message=f"Window duration = 1 second: {duration_ok}",
        details={
            "expected_ns": WINDOW_NS,
            "violations": int((window_duration != WINDOW_NS).sum())
        }
    ))
    
    # Check monotonicity
    monotonic = df["window_end_ts_ns"].is_monotonic_increasing
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="window_end_ts_ns",
        check="monotonic",
        status="PASS" if monotonic else "WARN",
        message=f"Timestamps monotonically increasing: {monotonic}",
        details={}
    ))
    
    # -------------------------------------------------------------------------
    # best_bid_price_int, best_ask_price_int: BBO prices (scaled by 1e-9)
    # -------------------------------------------------------------------------
    valid_book = (df["best_bid_price_int"] > 0) & (df["best_ask_price_int"] > 0)
    valid_count = valid_book.sum()
    total_count = len(df)
    
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="best_bid_price_int, best_ask_price_int",
        check="valid_book_coverage",
        status="PASS" if valid_count > total_count * 0.95 else "WARN",
        message=f"Valid BBO rows: {fmt_int(valid_count)}/{fmt_int(total_count)} ({fmt_pct(100*valid_count/total_count)})",
        details={"valid_count": valid_count, "total_count": total_count}
    ))
    
    # Check for crossed book (bid >= ask)
    if valid_count > 0:
        crossed = (df.loc[valid_book, "best_bid_price_int"] >= df.loc[valid_book, "best_ask_price_int"]).sum()
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="best_bid_price_int, best_ask_price_int",
            check="no_crossed_book",
            status="PASS" if crossed == 0 else "FAIL",
            message=f"Crossed book rows (bid >= ask): {fmt_int(crossed)}",
            details={"crossed_count": crossed}
        ))
    
    # Price range sanity (ES should be ~5000-7000 in 2026)
    if valid_count > 0:
        bid_prices_scaled = df.loc[valid_book, "best_bid_price_int"] * PRICE_SCALE
        ask_prices_scaled = df.loc[valid_book, "best_ask_price_int"] * PRICE_SCALE
        min_price = float(min(bid_prices_scaled.min(), ask_prices_scaled.min()))
        max_price = float(max(bid_prices_scaled.max(), ask_prices_scaled.max()))
        
        plausible = 1000 < min_price < 20000 and 1000 < max_price < 20000
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="best_bid_price_int, best_ask_price_int",
            check="price_range_plausible",
            status="PASS" if plausible else "WARN",
            message=f"Price range: [{fmt_float(min_price, 2)}, {fmt_float(max_price, 2)}]",
            details={"min_price": min_price, "max_price": max_price}
        ))
    
    # -------------------------------------------------------------------------
    # best_bid_qty, best_ask_qty: BBO quantities (contracts)
    # -------------------------------------------------------------------------
    for col in ["best_bid_qty", "best_ask_qty"]:
        neg = (df[col] < 0).sum()
        results.append(AuditResult(
            table="book_snapshot_1s",
            field=col,
            check="non_negative",
            status="PASS" if neg == 0 else "FAIL",
            message=f"Negative values: {fmt_int(neg)}",
            details={"negative_count": neg}
        ))
        
        # Stats
        results.append(AuditResult(
            table="book_snapshot_1s",
            field=col,
            check="stats",
            status="INFO",
            message=f"min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.1f}",
            details={"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean())}
        ))
    
    # -------------------------------------------------------------------------
    # mid_price, mid_price_int: (best_bid + best_ask)/2 - verify calculation
    # -------------------------------------------------------------------------
    if valid_count > 0:
        # mid_price = (bid + ask) / 2 * PRICE_SCALE
        expected_mid = (df.loc[valid_book, "best_bid_price_int"] + df.loc[valid_book, "best_ask_price_int"]) * 0.5 * PRICE_SCALE
        actual_mid = df.loc[valid_book, "mid_price"]
        diff = (expected_mid - actual_mid).abs()
        tolerance = 1e-12
        formula_ok = (diff <= tolerance).all()
        violations = int((diff > tolerance).sum())
        
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="mid_price",
            check="formula_verification",
            status="PASS" if formula_ok else "FAIL",
            message=f"mid_price = (bid + ask)/2 * 1e-9: {formula_ok} (violations: {violations})",
            details={"violations": violations, "tolerance": tolerance}
        ))
        
        # mid_price_int = round((bid + ask) / 2)
        expected_mid_int = ((df.loc[valid_book, "best_bid_price_int"] + df.loc[valid_book, "best_ask_price_int"]) * 0.5).round().astype(np.int64)
        actual_mid_int = df.loc[valid_book, "mid_price_int"]
        formula_int_ok = (expected_mid_int == actual_mid_int).all()
        violations_int = int((expected_mid_int != actual_mid_int).sum())
        
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="mid_price_int",
            check="formula_verification",
            status="PASS" if formula_int_ok else "FAIL",
            message=f"mid_price_int = round((bid + ask)/2): {formula_int_ok} (violations: {violations_int})",
            details={"violations": violations_int}
        ))
    
    # -------------------------------------------------------------------------
    # last_trade_price_int: Most recent trade price
    # -------------------------------------------------------------------------
    has_trade = (df["last_trade_price_int"] > 0).sum()
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="last_trade_price_int",
        check="coverage",
        status="PASS" if has_trade > 0 else "WARN",
        message=f"Rows with last trade: {fmt_int(has_trade)}/{fmt_int(total_count)} ({fmt_pct(100*has_trade/total_count)})",
        details={"has_trade_count": has_trade}
    ))
    
    # -------------------------------------------------------------------------
    # spot_ref_price_int: Reference price for grid alignment
    # -------------------------------------------------------------------------
    has_spot_ref = (df["spot_ref_price_int"] > 0).sum()
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="spot_ref_price_int",
        check="coverage",
        status="PASS" if has_spot_ref > 0 else "WARN",
        message=f"Rows with spot_ref: {fmt_int(has_spot_ref)}/{fmt_int(total_count)} ({fmt_pct(100*has_spot_ref/total_count)})",
        details={"has_spot_ref_count": has_spot_ref}
    ))
    
    # -------------------------------------------------------------------------
    # book_valid: Data quality flag
    # -------------------------------------------------------------------------
    valid_true = df["book_valid"].sum()
    valid_false = (~df["book_valid"]).sum()
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="book_valid",
        check="distribution",
        status="INFO",
        message=f"book_valid=True: {fmt_int(valid_true)}, book_valid=False: {fmt_int(valid_false)}",
        details={"true_count": valid_true, "false_count": valid_false}
    ))
    
    return results


def semantic_depth_and_flow(df: pd.DataFrame) -> list[AuditResult]:
    """Semantic analysis of depth_and_flow_1s fields."""
    results = []
    total_count = len(df)
    
    # -------------------------------------------------------------------------
    # price_int: Absolute price level (scaled by 1e-9)
    # -------------------------------------------------------------------------
    neg_price = (df["price_int"] <= 0).sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="price_int",
        check="positive",
        status="PASS" if neg_price == 0 else "FAIL",
        message=f"Non-positive price_int: {fmt_int(neg_price)}",
        details={"non_positive_count": neg_price}
    ))
    
    # -------------------------------------------------------------------------
    # side: "A" (ask) or "B" (bid)
    # -------------------------------------------------------------------------
    valid_sides = {"A", "B"}
    actual_sides = set(df["side"].unique())
    sides_ok = actual_sides.issubset(valid_sides)
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="side",
        check="valid_values",
        status="PASS" if sides_ok else "FAIL",
        message=f"Side values: {sorted(actual_sides)}",
        details={"values": list(actual_sides), "invalid": list(actual_sides - valid_sides)}
    ))
    
    bid_count = (df["side"] == "B").sum()
    ask_count = (df["side"] == "A").sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="side",
        check="distribution",
        status="INFO",
        message=f"Bid rows: {fmt_int(bid_count)}, Ask rows: {fmt_int(ask_count)}",
        details={"bid_count": bid_count, "ask_count": ask_count}
    ))
    
    # -------------------------------------------------------------------------
    # spot_ref_price_int: Grid anchor
    # -------------------------------------------------------------------------
    has_spot = (df["spot_ref_price_int"] > 0).sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="spot_ref_price_int",
        check="coverage",
        status="PASS" if has_spot == total_count else "WARN",
        message=f"Rows with spot_ref: {fmt_int(has_spot)}/{fmt_int(total_count)}",
        details={"has_spot_count": has_spot}
    ))
    
    # -------------------------------------------------------------------------
    # rel_ticks: (price_int - spot_ref_price_int) / tick_int where tick = $0.25
    # -------------------------------------------------------------------------
    # Verify formula
    expected_rel_ticks = ((df["price_int"] - df["spot_ref_price_int"]) / TICK_INT).round().astype(int)
    actual_rel_ticks = df["rel_ticks"]
    rel_ticks_ok = (expected_rel_ticks == actual_rel_ticks).all()
    violations = int((expected_rel_ticks != actual_rel_ticks).sum())
    
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="rel_ticks",
        check="formula_verification",
        status="PASS" if rel_ticks_ok else "FAIL",
        message=f"rel_ticks = (price - spot_ref) / TICK_INT: {rel_ticks_ok} (violations: {violations})",
        details={"violations": violations}
    ))
    
    # Check range (+/- GRID_MAX_TICKS)
    min_rel = df["rel_ticks"].min()
    max_rel = df["rel_ticks"].max()
    range_ok = (min_rel >= -GRID_MAX_TICKS) and (max_rel <= GRID_MAX_TICKS)
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="rel_ticks",
        check="range",
        status="PASS" if range_ok else "WARN",
        message=f"rel_ticks range: [{min_rel}, {max_rel}] (expected +/- {GRID_MAX_TICKS})",
        details={"min": min_rel, "max": max_rel, "limit": GRID_MAX_TICKS}
    ))
    
    # Distribution symmetry around 0
    rel_ticks_mean = df["rel_ticks"].mean()
    symmetric = abs(rel_ticks_mean) < 5  # Within 5 ticks of symmetric
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="rel_ticks",
        check="distribution_symmetry",
        status="PASS" if symmetric else "WARN",
        message=f"Mean rel_ticks: {rel_ticks_mean:.2f} (symmetric if near 0)",
        details={"mean": rel_ticks_mean}
    ))
    
    # -------------------------------------------------------------------------
    # rel_ticks_side: Relative to best bid/ask
    # -------------------------------------------------------------------------
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="rel_ticks_side",
        check="stats",
        status="INFO",
        message=f"min={df['rel_ticks_side'].min()}, max={df['rel_ticks_side'].max()}, mean={df['rel_ticks_side'].mean():.2f}",
        details={"min": int(df["rel_ticks_side"].min()), "max": int(df["rel_ticks_side"].max()), "mean": float(df["rel_ticks_side"].mean())}
    ))
    
    # -------------------------------------------------------------------------
    # Quantity fields: depth_qty_start, depth_qty_end, add_qty, pull_qty, depth_qty_rest, pull_qty_rest, fill_qty
    # -------------------------------------------------------------------------
    qty_fields = ["depth_qty_start", "depth_qty_end", "add_qty", "pull_qty", "depth_qty_rest", "pull_qty_rest", "fill_qty"]
    
    for col in qty_fields:
        # Non-negative constraint
        neg = (df[col] < 0).sum()
        results.append(AuditResult(
            table="depth_and_flow_1s",
            field=col,
            check="non_negative",
            status="PASS" if neg == 0 else "FAIL",
            message=f"Negative values: {fmt_int(neg)}",
            details={"negative_count": neg, "min_value": float(df[col].min())}
        ))
        
        # Stats
        results.append(AuditResult(
            table="depth_and_flow_1s",
            field=col,
            check="stats",
            status="INFO",
            message=f"min={df[col].min():.1f}, max={df[col].max():.1f}, mean={df[col].mean():.2f}, std={df[col].std():.2f}",
            details={"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean()), "std": float(df[col].std())}
        ))
    
    # -------------------------------------------------------------------------
    # window_valid: Data quality flag
    # -------------------------------------------------------------------------
    valid_true = df["window_valid"].sum()
    valid_false = (~df["window_valid"]).sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="window_valid",
        check="distribution",
        status="INFO",
        message=f"window_valid=True: {fmt_int(valid_true)}, window_valid=False: {fmt_int(valid_false)}",
        details={"true_count": valid_true, "false_count": valid_false}
    ))
    
    return results


# ==============================================================================
# TASK 3: Statistical Analysis (Accounting Identities)
# ==============================================================================
def statistical_verification(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> list[AuditResult]:
    """Run statistical verifications."""
    results = []
    
    # -------------------------------------------------------------------------
    # ACCOUNTING IDENTITY: depth_qty_start + add_qty - pull_qty - fill_qty = depth_qty_end
    # -------------------------------------------------------------------------
    # Rearranged: depth_qty_end = depth_qty_start + add_qty - pull_qty - fill_qty
    # But the engine computes: depth_qty_start = depth_qty_end - add_qty + pull_qty + fill_qty
    # So we verify: depth_qty_end - depth_qty_start - add_qty + pull_qty + fill_qty ≈ 0
    balance = df_flow["depth_qty_end"] - df_flow["depth_qty_start"] - df_flow["add_qty"] + df_flow["pull_qty"] + df_flow["fill_qty"]
    tolerance = 0.01
    balance_ok = (balance.abs() <= tolerance).all()
    violations = int((balance.abs() > tolerance).sum())
    max_imbalance = balance.abs().max()
    
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="accounting_identity",
        check="depth_qty_end = depth_qty_start + add_qty - pull_qty - fill_qty",
        status="PASS" if balance_ok else "FAIL",
        message=f"Accounting identity verified: {balance_ok} (violations: {violations}, max_imbalance: {max_imbalance:.4f})",
        details={"violations": violations, "max_imbalance": float(max_imbalance), "tolerance": tolerance}
    ))
    
    # -------------------------------------------------------------------------
    # Spread = (best_ask_price_int - best_bid_price_int) should be positive when book_valid
    # -------------------------------------------------------------------------
    valid_book = (df_snap["best_bid_price_int"] > 0) & (df_snap["best_ask_price_int"] > 0) & df_snap["book_valid"]
    if valid_book.any():
        spread = df_snap.loc[valid_book, "best_ask_price_int"] - df_snap.loc[valid_book, "best_bid_price_int"]
        neg_spread = (spread <= 0).sum()
        spread_ticks = spread / TICK_INT
        
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="spread",
            check="positive",
            status="PASS" if neg_spread == 0 else "FAIL",
            message=f"Non-positive spreads when book_valid: {fmt_int(neg_spread)}",
            details={"non_positive_count": neg_spread}
        ))
        
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="spread",
            check="stats",
            status="INFO",
            message=f"Spread (ticks): min={spread_ticks.min():.1f}, max={spread_ticks.max():.1f}, mean={spread_ticks.mean():.2f}",
            details={"min_ticks": float(spread_ticks.min()), "max_ticks": float(spread_ticks.max()), "mean_ticks": float(spread_ticks.mean())}
        ))
    
    # -------------------------------------------------------------------------
    # Check depth_qty_rest <= depth_qty_end (resting can't exceed total)
    # -------------------------------------------------------------------------
    rest_exceeds = (df_flow["depth_qty_rest"] > df_flow["depth_qty_end"] + 0.01).sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="depth_qty_rest",
        check="depth_qty_rest <= depth_qty_end",
        status="PASS" if rest_exceeds == 0 else "FAIL",
        message=f"Resting exceeds total depth: {fmt_int(rest_exceeds)} rows",
        details={"violations": rest_exceeds}
    ))
    
    # -------------------------------------------------------------------------
    # Check pull_qty_rest <= pull_qty (can only pull resting from total pulls)
    # -------------------------------------------------------------------------
    pull_rest_exceeds = (df_flow["pull_qty_rest"] > df_flow["pull_qty"] + 0.01).sum()
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="pull_qty_rest",
        check="pull_qty_rest <= pull_qty",
        status="PASS" if pull_rest_exceeds == 0 else "FAIL",
        message=f"Resting pull exceeds total pull: {fmt_int(pull_rest_exceeds)} rows",
        details={"violations": pull_rest_exceeds}
    ))
    
    # -------------------------------------------------------------------------
    # Check for negative depth (impossible in real markets)
    # -------------------------------------------------------------------------
    for col in ["depth_qty_start", "depth_qty_end"]:
        neg_depth = (df_flow[col] < 0).sum()
        results.append(AuditResult(
            table="depth_and_flow_1s",
            field=col,
            check="no_negative_depth",
            status="PASS" if neg_depth == 0 else "FAIL",
            message=f"Negative depth values: {fmt_int(neg_depth)}",
            details={"negative_count": neg_depth}
        ))
    
    return results


# ==============================================================================
# TASK 4: Data Quality Report
# ==============================================================================
def data_quality_report(df_snap: pd.DataFrame, df_flow: pd.DataFrame) -> list[AuditResult]:
    """Data quality analysis."""
    results = []
    
    # -------------------------------------------------------------------------
    # book_valid distribution
    # -------------------------------------------------------------------------
    snap_valid = df_snap["book_valid"].sum()
    snap_invalid = (~df_snap["book_valid"]).sum()
    snap_pct = 100 * snap_valid / len(df_snap) if len(df_snap) > 0 else 0
    
    status = "PASS" if snap_pct > 95 else ("WARN" if snap_pct > 80 else "FAIL")
    results.append(AuditResult(
        table="book_snapshot_1s",
        field="book_valid",
        check="quality",
        status=status,
        message=f"Valid: {fmt_int(snap_valid)} ({fmt_pct(snap_pct)}), Invalid: {fmt_int(snap_invalid)}",
        details={"valid_count": snap_valid, "invalid_count": snap_invalid, "valid_pct": snap_pct}
    ))
    
    # -------------------------------------------------------------------------
    # window_valid distribution
    # -------------------------------------------------------------------------
    flow_valid = df_flow["window_valid"].sum()
    flow_invalid = (~df_flow["window_valid"]).sum()
    flow_pct = 100 * flow_valid / len(df_flow) if len(df_flow) > 0 else 0
    
    status = "PASS" if flow_pct > 95 else ("WARN" if flow_pct > 80 else "FAIL")
    results.append(AuditResult(
        table="depth_and_flow_1s",
        field="window_valid",
        check="quality",
        status=status,
        message=f"Valid: {fmt_int(flow_valid)} ({fmt_pct(flow_pct)}), Invalid: {fmt_int(flow_invalid)}",
        details={"valid_count": flow_valid, "invalid_count": flow_invalid, "valid_pct": flow_pct}
    ))
    
    # -------------------------------------------------------------------------
    # NaN analysis
    # -------------------------------------------------------------------------
    for table, df in [("book_snapshot_1s", df_snap), ("depth_and_flow_1s", df_flow)]:
        total_nans = df.isna().sum().sum()
        results.append(AuditResult(
            table=table,
            field="*",
            check="nan_count",
            status="PASS" if total_nans == 0 else "WARN",
            message=f"Total NaN values: {fmt_int(total_nans)}",
            details={"nan_count": total_nans}
        ))
    
    # -------------------------------------------------------------------------
    # Anomaly detection: impossible states
    # -------------------------------------------------------------------------
    # Crossed book when valid
    valid_book = (df_snap["best_bid_price_int"] > 0) & (df_snap["best_ask_price_int"] > 0) & df_snap["book_valid"]
    if valid_book.any():
        crossed = (df_snap.loc[valid_book, "best_bid_price_int"] >= df_snap.loc[valid_book, "best_ask_price_int"]).sum()
        results.append(AuditResult(
            table="book_snapshot_1s",
            field="anomaly",
            check="crossed_book_when_valid",
            status="PASS" if crossed == 0 else "FAIL",
            message=f"Crossed book when book_valid=True: {fmt_int(crossed)}",
            details={"count": crossed}
        ))
    
    return results


# ==============================================================================
# TASK 5: Final Grade
# ==============================================================================
def compute_grade(report: AuditReport) -> str:
    """Compute final grade based on audit results."""
    summary = report.summary()
    
    fail_count = summary["FAIL"]
    warn_count = summary["WARN"]
    
    if fail_count == 0 and warn_count == 0:
        return "A"
    elif fail_count == 0 and warn_count <= 3:
        return "A-"
    elif fail_count == 0 and warn_count <= 6:
        return "B+"
    elif fail_count <= 2 and warn_count <= 5:
        return "B"
    elif fail_count <= 3:
        return "B-"
    elif fail_count <= 5:
        return "C"
    elif fail_count <= 8:
        return "D"
    else:
        return "F"


# ==============================================================================
# Main Audit Function
# ==============================================================================
def run_audit(symbol: str, dt: str) -> AuditReport:
    """Run full audit for a single day."""
    report = AuditReport(symbol=symbol, dt=dt)
    
    print(f"\n{'='*80}")
    print(f"  INSTITUTION-GRADE SILVER LAYER AUDIT")
    print(f"  Product Type: future_mbo")
    print(f"  Symbol: {symbol}")
    print(f"  Date: {dt}")
    print(f"{'='*80}")
    
    # Load data
    print("\n[1/5] Loading data...")
    df_snap = load_parquet(symbol, dt, "book_snapshot_1s")
    df_flow = load_parquet(symbol, dt, "depth_and_flow_1s")
    
    if df_snap is None:
        print("  FATAL: book_snapshot_1s data not found")
        return report
    if df_flow is None:
        print("  FATAL: depth_and_flow_1s data not found")
        return report
    
    print(f"  book_snapshot_1s: {fmt_int(len(df_snap))} rows")
    print(f"  depth_and_flow_1s: {fmt_int(len(df_flow))} rows")
    
    # Task 1: Schema inspection
    print("\n[2/5] Schema inspection...")
    for r in describe_schema(df_snap, "book_snapshot_1s"):
        report.add(r)
    for r in describe_schema(df_flow, "depth_and_flow_1s"):
        report.add(r)
    
    # Task 2: Semantic analysis
    print("\n[3/5] Semantic analysis...")
    for r in semantic_book_snapshot(df_snap):
        report.add(r)
    for r in semantic_depth_and_flow(df_flow):
        report.add(r)
    
    # Task 3: Statistical verification
    print("\n[4/5] Statistical verification...")
    for r in statistical_verification(df_snap, df_flow):
        report.add(r)
    
    # Task 4: Data quality report
    print("\n[5/5] Data quality report...")
    for r in data_quality_report(df_snap, df_flow):
        report.add(r)
    
    # Compute grade
    report.grade = compute_grade(report)
    
    return report


def print_report(report: AuditReport):
    """Print formatted audit report."""
    print(f"\n{'='*80}")
    print(f"  AUDIT REPORT")
    print(f"{'='*80}")
    
    summary = report.summary()
    print(f"\n  Summary: PASS={summary['PASS']}, WARN={summary['WARN']}, FAIL={summary['FAIL']}, INFO={summary['INFO']}")
    print(f"  Grade: {report.grade}")
    
    # Print failures
    failures = [r for r in report.results if r.status == "FAIL"]
    if failures:
        print(f"\n  === FAILURES ({len(failures)}) ===")
        for r in failures:
            print(f"    [{r.table}] {r.field}: {r.check}")
            print(f"      {r.message}")
    
    # Print warnings
    warnings = [r for r in report.results if r.status == "WARN"]
    if warnings:
        print(f"\n  === WARNINGS ({len(warnings)}) ===")
        for r in warnings:
            print(f"    [{r.table}] {r.field}: {r.check}")
            print(f"      {r.message}")
    
    # Print detailed results by table
    for table in ["book_snapshot_1s", "depth_and_flow_1s"]:
        print(f"\n  === {table} DETAILS ===")
        table_results = [r for r in report.results if r.table == table]
        for r in table_results:
            status_icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "INFO": "ℹ"}.get(r.status, "?")
            print(f"    {status_icon} [{r.status}] {r.field} | {r.check}: {r.message}")


def main():
    symbol = "ESH6"
    dt = "2026-01-06"
    
    report = run_audit(symbol, dt)
    print_report(report)
    
    # Exit code based on failures
    if report.summary()["FAIL"] > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
