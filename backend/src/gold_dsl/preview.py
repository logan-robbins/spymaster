"""Gold DSL preview executor: run DSL specs against sample data.

Executes a validated GoldDslSpec against parquet data files to produce
summary statistics for each output node. Designed for rapid iteration
during feature engineering -- not for production serving.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    ArithmeticExpr,
    DslNode,
    GoldDslSpec,
    NormExpr,
    OutputNode,
    SilverRef,
    SpatialNeighborhood,
    TemporalWindow,
)
from .validate import _get_source_refs, validate_dsl

logger: logging.Logger = logging.getLogger(__name__)

_EPS = 1e-12


def _topological_order(nodes: dict[str, DslNode]) -> list[str]:
    """Compute topological execution order using Kahn's algorithm.

    Args:
        nodes: Mapping of node name to DslNode.

    Returns:
        List of node names in valid execution order.

    Raises:
        ValueError: If the graph contains a cycle (should not happen
            after validation, but guards against misuse).
    """
    node_names = set(nodes.keys())
    in_degree: dict[str, int] = {name: 0 for name in node_names}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for name, node in nodes.items():
        for ref in _get_source_refs(node):
            if ref in node_names:
                adjacency[ref].append(name)
                in_degree[name] += 1

    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for downstream in adjacency[current]:
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)

    if len(order) != len(node_names):
        raise ValueError("Cycle detected in DSL graph during execution ordering")

    return order


def _compute_silver_ref(
    node: SilverRef,
    grid_df: pd.DataFrame,
) -> pd.Series:
    """Extract a silver column from the grid dataframe.

    Args:
        node: SilverRef node specifying the column.
        grid_df: Grid dataframe with silver columns.

    Returns:
        Pandas Series for the referenced column.

    Raises:
        KeyError: If the field does not exist in the dataframe.
    """
    if node.field not in grid_df.columns:
        raise KeyError(
            f"Silver field '{node.field}' not found in grid dataframe. "
            f"Available: {sorted(grid_df.columns)}"
        )
    return grid_df[node.field].astype(np.float64)


def _compute_temporal_window(
    node: TemporalWindow,
    source_series: pd.Series,
    grid_df: pd.DataFrame,
) -> pd.Series:
    """Apply temporal (per-k rolling) aggregation to a source series.

    Groups by relative tick ``k``, then applies a rolling window
    aggregation across the bin dimension.

    Args:
        node: TemporalWindow node with aggregation parameters.
        source_series: Input series aligned with grid_df index.
        grid_df: Grid dataframe providing ``k`` column for grouping.

    Returns:
        Aggregated series with same index as source_series.
    """
    tmp = grid_df[["k"]].copy()
    tmp["__val"] = source_series.values

    if node.agg == "ewm":
        result = tmp.groupby("k")["__val"].transform(
            lambda s: s.ewm(alpha=node.alpha, adjust=False).mean()
        )
    else:
        agg_func = node.agg
        result = tmp.groupby("k")["__val"].transform(
            lambda s: getattr(s.rolling(node.window_bins, min_periods=1), agg_func)()
        )

    return result.astype(np.float64)


def _compute_spatial_neighborhood(
    node: SpatialNeighborhood,
    source_series: pd.Series,
    grid_df: pd.DataFrame,
) -> pd.Series:
    """Apply spatial (across-k) neighborhood aggregation.

    Pivots the source data to a (bin_seq x k) matrix, applies a rolling
    window along the k axis, then melts back to the original shape.

    Args:
        node: SpatialNeighborhood node with radius and aggregation params.
        source_series: Input series aligned with grid_df index.
        grid_df: Grid dataframe providing ``bin_seq`` and ``k`` columns.

    Returns:
        Spatially aggregated series with same index as source_series.
    """
    tmp = grid_df[["bin_seq", "k"]].copy()
    tmp["__val"] = source_series.values

    pivot = tmp.pivot_table(index="bin_seq", columns="k", values="__val")
    window_size = 2 * node.radius_ticks + 1
    agg_func = node.agg

    if agg_func == "mean":
        smoothed = pivot.T.rolling(window_size, center=True, min_periods=1).mean().T
    elif agg_func == "std":
        smoothed = pivot.T.rolling(window_size, center=True, min_periods=1).std().T
    elif agg_func == "sum":
        smoothed = pivot.T.rolling(window_size, center=True, min_periods=1).sum().T
    elif agg_func == "max":
        smoothed = pivot.T.rolling(window_size, center=True, min_periods=1).max().T
    else:
        raise ValueError(f"Unsupported spatial agg: {agg_func}")

    melted = smoothed.stack().reset_index()
    melted.columns = ["bin_seq", "k", "__val"]

    merged = grid_df[["bin_seq", "k"]].merge(
        melted, on=["bin_seq", "k"], how="left"
    )
    return merged["__val"].astype(np.float64)


def _compute_arithmetic(
    node: ArithmeticExpr,
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Apply a binary arithmetic operation.

    Args:
        node: ArithmeticExpr node specifying the operation.
        left: Left operand series.
        right: Right operand series.

    Returns:
        Result series.
    """
    if node.op == "add":
        return (left + right).astype(np.float64)
    if node.op == "sub":
        return (left - right).astype(np.float64)
    if node.op == "mul":
        return (left * right).astype(np.float64)
    if node.op == "div":
        return (left / (right + _EPS)).astype(np.float64)
    if node.op == "abs_diff":
        return (left - right).abs().astype(np.float64)
    raise ValueError(f"Unknown arithmetic op: {node.op}")


def _compute_norm(
    node: NormExpr,
    source_series: pd.Series,
    grid_df: pd.DataFrame,
) -> pd.Series:
    """Apply a normalization transformation.

    Args:
        node: NormExpr node specifying the method and parameters.
        source_series: Input series to normalize.
        grid_df: Grid dataframe for grouping context.

    Returns:
        Normalized series.
    """
    if node.method == "log1p":
        return np.log1p(source_series.abs()).astype(np.float64)
    if node.method == "tanh":
        return np.tanh(source_series).astype(np.float64)
    if node.method == "minmax":
        mn = source_series.min()
        mx = source_series.max()
        rng = mx - mn
        if abs(rng) < _EPS:
            return pd.Series(np.zeros(len(source_series)), index=source_series.index, dtype=np.float64)
        return ((source_series - mn) / rng).astype(np.float64)
    if node.method == "zscore":
        tmp = grid_df[["k"]].copy()
        tmp["__val"] = source_series.values
        result = tmp.groupby("k")["__val"].transform(
            lambda s: (
                (s - s.rolling(node.window_bins, min_periods=1).mean())
                / (s.rolling(node.window_bins, min_periods=1).std() + _EPS)
            )
        )
        return result.astype(np.float64)
    raise ValueError(f"Unknown norm method: {node.method}")


def execute_dsl_preview(
    spec: GoldDslSpec,
    bins_parquet_path: Path,
    grid_parquet_path: Path,
    sample_bins: int = 100,
) -> dict[str, Any]:
    """Execute a DSL spec against sample parquet data for preview statistics.

    Validates the spec, loads parquet data, computes all nodes in
    topological order, and returns summary statistics for each output.

    Args:
        spec: A GoldDslSpec defining the feature computation graph.
        bins_parquet_path: Path to bins.parquet with bin_seq metadata.
        grid_parquet_path: Path to grid_clean.parquet with silver columns.
        sample_bins: Maximum number of bins to sample. Uses the first
            N bins for deterministic results.

    Returns:
        Dictionary with keys:
            - ``output_stats``: Per-output statistics (mean, std, pct25,
              pct50, pct75, n_valid, n_nan).
            - ``n_bins_sampled``: Actual number of bins processed.
            - ``execution_time_ms``: Wall-clock execution time.
            - ``spec_hash``: Content hash of the spec.

    Raises:
        ValueError: If the spec fails validation.
        FileNotFoundError: If parquet paths do not exist.
    """
    validation_errors = validate_dsl(spec)
    if validation_errors:
        raise ValueError(
            f"DSL spec failed validation with {len(validation_errors)} errors: "
            + "; ".join(validation_errors)
        )

    if not bins_parquet_path.exists():
        raise FileNotFoundError(f"bins parquet not found: {bins_parquet_path}")
    if not grid_parquet_path.exists():
        raise FileNotFoundError(f"grid parquet not found: {grid_parquet_path}")

    t0 = time.monotonic()

    bins_df = pd.read_parquet(bins_parquet_path, columns=["bin_seq"])
    all_bin_seqs = sorted(bins_df["bin_seq"].unique())
    sampled_seqs = all_bin_seqs[:sample_bins]
    n_bins_sampled = len(sampled_seqs)

    grid_df = pd.read_parquet(grid_parquet_path)
    grid_df = grid_df[grid_df["bin_seq"].isin(sampled_seqs)].copy()
    grid_df = grid_df.sort_values(["k", "bin_seq"]).reset_index(drop=True)

    logger.info(
        "DSL preview: %d bins sampled, %d grid rows",
        n_bins_sampled,
        len(grid_df),
    )

    exec_order = _topological_order(spec.nodes)

    computed: dict[str, pd.Series] = {}
    outputs: dict[str, pd.Series] = {}

    for node_name in exec_order:
        node = spec.nodes[node_name]

        if isinstance(node, SilverRef):
            computed[node_name] = _compute_silver_ref(node, grid_df)

        elif isinstance(node, TemporalWindow):
            computed[node_name] = _compute_temporal_window(
                node, computed[node.source], grid_df
            )

        elif isinstance(node, SpatialNeighborhood):
            computed[node_name] = _compute_spatial_neighborhood(
                node, computed[node.source], grid_df
            )

        elif isinstance(node, ArithmeticExpr):
            computed[node_name] = _compute_arithmetic(
                node, computed[node.left], computed[node.right]
            )

        elif isinstance(node, NormExpr):
            computed[node_name] = _compute_norm(
                node, computed[node.source], grid_df
            )

        elif isinstance(node, OutputNode):
            output_series = computed[node.source]
            outputs[node.name] = output_series
            computed[node_name] = output_series

    elapsed_ms = (time.monotonic() - t0) * 1000.0

    output_stats: dict[str, dict[str, Any]] = {}
    for output_name, series in outputs.items():
        n_nan = int(series.isna().sum())
        n_valid = int(series.notna().sum())
        valid_vals = series.dropna()

        if len(valid_vals) > 0:
            output_stats[output_name] = {
                "mean": float(valid_vals.mean()),
                "std": float(valid_vals.std()),
                "pct25": float(valid_vals.quantile(0.25)),
                "pct50": float(valid_vals.quantile(0.50)),
                "pct75": float(valid_vals.quantile(0.75)),
                "n_valid": n_valid,
                "n_nan": n_nan,
            }
        else:
            output_stats[output_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "pct25": float("nan"),
                "pct50": float("nan"),
                "pct75": float("nan"),
                "n_valid": 0,
                "n_nan": n_nan,
            }

    return {
        "output_stats": output_stats,
        "n_bins_sampled": n_bins_sampled,
        "execution_time_ms": round(elapsed_ms, 2),
        "spec_hash": spec.spec_hash(),
    }
