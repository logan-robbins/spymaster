"""Vacuum & Pressure Detection Formulas.

Mathematical framework for detecting directional micro-regimes from
order flow imbalance and liquidity dynamics in the MBO order book.

Supports both equity and futures product types via runtime configuration.
All spatial parameters (proximity tau, near-spot range, depth range) are
defined in **dollar space** and converted to rel-tick counts using the
runtime ``bucket_size_dollars`` to prevent semantic drift across instruments.

Definitions:
    Vacuum: Region where liquidity is thinning (orders pulled > orders added).
        A vacuum above spot (ask-side) is **bullish** -- less resistance upward.
        A vacuum below spot (bid-side) is **bearish** -- less support below.

    Pressure: Region where liquidity is building and migrating toward spot.
        Pressure from below (bids chasing) is **bullish**.
        Pressure from above (asks pressing) is **bearish**.

Metric taxonomy:
    ADDITIVE: volume, add_qty, pull_qty, fill_qty -- sum across rollups.
    NON-ADDITIVE: vacuum_score, flow_imbalance, depth_imbalance -- recompute.
    SEMI-ADDITIVE: depth_qty_end -- snapshot at boundary, sum across price levels.

Proximity weighting:
    w(k) = exp(-|k| / tau),  tau = proximity_tau_dollars / bucket_size_dollars
    For QQQ equity with $0.50 buckets (tau_dollars=$2.50):
        k=1  -> $0.50,  w = 0.819
        k=5  -> $2.50,  w = 0.368
        k=10 -> $5.00,  w = 0.135
        k=20 -> $10.00, w = 0.018

References:
    Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events"
    Cartea, Jaimungal & Penalva (2015), "Algorithmic and High-Frequency Trading"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import VPRuntimeConfig

# ──────────────────────────────────────────────────────────────────────
# Dollar-space constants (instrument-independent)
# ──────────────────────────────────────────────────────────────────────

PROXIMITY_TAU_DOLLARS: float = 2.50
r"""Exponential decay characteristic length in dollar space.

For QQQ ($0.50 buckets): tau_ticks = $2.50 / $0.50 = 5.0 ticks.
For MNQ ($0.25 buckets): tau_ticks = $2.50 / $0.25 = 10.0 ticks.
"""

NEAR_SPOT_DOLLARS: float = 2.50
"""Near-spot flow imbalance radius in dollars.

For QQQ ($0.50 buckets): 5 ticks.
For MNQ ($0.25 buckets): 10 ticks.
"""

DEPTH_RANGE_DOLLARS: float = 5.00
"""Depth imbalance computation radius in dollars.

For QQQ ($0.50 buckets): 10 ticks.
For MNQ ($0.25 buckets): 20 ticks.
"""

EMA_SPAN_D1: int = 5
"""EMA span (seconds) for 1st derivative (velocity) smoothing."""

EMA_SPAN_D2: int = 10
"""EMA span (seconds) for 2nd derivative (acceleration) smoothing."""

EMA_SPAN_D3: int = 20
"""EMA span (seconds) for 3rd derivative (jerk) smoothing."""

COMPOSITE_WEIGHTS: dict[str, float] = {
    "vacuum": 1.0,
    "flow": 0.5,
    "fill": 0.3,
    "institutional_drain": 2.0,
    "rest_depth": 0.2,
}
r"""Weights for composite signal construction.

    S(t) = a1*V_ask_above - a1*V_bid_below
         + a2*F
         + a3*Phi
         + a4*(D_ask - D_bid)
         + a5*Delta_rest

vacuum (1.0):              Direct measure of liquidity thinning / building.
flow (0.5):                Near-spot order addition imbalance.
fill (0.3):                Trade-initiated flow (direct aggression evidence).
institutional_drain (2.0): Resting (>500 ms) cancel activity (high S/N).
rest_depth (0.2):          Structural depth imbalance (slow-moving, context).
"""

ZSCORE_WINDOW: int = 60
"""Rolling window (seconds) for z-score normalization of composite signal."""

EPS: float = 1.0
"""Small constant to prevent division by zero in ratio computations."""


# ──────────────────────────────────────────────────────────────────────
# Config-derived tick parameters
# ──────────────────────────────────────────────────────────────────────


def _proximity_tau_ticks(bucket_size_dollars: float) -> float:
    """Convert dollar-space tau to rel-tick-space tau.

    For QQQ ($0.50 buckets): 2.50 / 0.50 = 5.0 (matches original constant).
    """
    return PROXIMITY_TAU_DOLLARS / bucket_size_dollars


def _near_spot_ticks(bucket_size_dollars: float) -> int:
    """Convert dollar-space near-spot radius to tick count.

    For QQQ ($0.50 buckets): 2.50 / 0.50 = 5 (matches original constant).
    """
    return int(round(NEAR_SPOT_DOLLARS / bucket_size_dollars))


def _depth_range_ticks(bucket_size_dollars: float) -> int:
    """Convert dollar-space depth range to tick count.

    For QQQ ($0.50 buckets): 5.00 / 0.50 = 10 (matches original constant).
    """
    return int(round(DEPTH_RANGE_DOLLARS / bucket_size_dollars))


# ──────────────────────────────────────────────────────────────────────
# Core Functions
# ──────────────────────────────────────────────────────────────────────


def proximity_weight(
    rel_ticks: np.ndarray,
    tau: float,
) -> np.ndarray:
    r"""Compute exponential proximity weight for each price level.

    .. math::
        w(k) = \exp\!\left(-\frac{|k|}{\tau}\right)

    Args:
        rel_ticks: Array of relative tick positions (int). k = 0 is spot.
        tau: Decay constant in tick units.

    Returns:
        Array of weights in [0, 1], same shape as *rel_ticks*.
    """
    return np.exp(-np.abs(rel_ticks.astype(np.float64)) / tau)


def compute_per_bucket_scores(
    df_flow: pd.DataFrame,
    bucket_size_dollars: float,
) -> pd.DataFrame:
    """Add per-bucket vacuum / pressure scores to depth_and_flow data.

    For each row (price-bucket x side x window) computes:
        net_flow            = add_qty - pull_qty
        vacuum_intensity    = max(0, pull_qty - add_qty) * w(k)
        pressure_intensity  = max(0, add_qty - pull_qty) * w(k)
        rest_fraction       = depth_qty_rest / (depth_qty_end + eps)

    Args:
        df_flow: Silver ``depth_and_flow_1s`` DataFrame.
        bucket_size_dollars: Bucket size in dollars for proximity weighting.

    Returns:
        Copy of *df_flow* with additional derived columns.
    """
    df = df_flow.copy()

    tau = _proximity_tau_ticks(bucket_size_dollars)
    rt = df["rel_ticks"].values
    w = proximity_weight(rt, tau)
    df["proximity_weight"] = w

    add = df["add_qty"].values
    pull = df["pull_qty"].values

    net = add - pull
    df["net_flow"] = net

    drain = pull - add
    df["vacuum_intensity"] = np.maximum(drain, 0.0) * w

    build = add - pull
    df["pressure_intensity"] = np.maximum(build, 0.0) * w

    depth_end = df["depth_qty_end"].values
    depth_rest = df["depth_qty_rest"].values
    safe_denom = np.where(depth_end > EPS, depth_end, 1.0)
    df["rest_fraction"] = np.where(depth_end > EPS, depth_rest / safe_denom, 0.0)

    return df


def aggregate_window_metrics(
    df_flow: pd.DataFrame,
    bucket_size_dollars: float,
) -> pd.DataFrame:
    r"""Aggregate per-bucket flow into per-window vacuum / pressure metrics.

    For each 1-second window computes:

    **Vacuum scores** (positive => liquidity draining):

    .. math::
        V_{\text{ask,above}}(t) = \sum_{\substack{k > 0 \\ s = A}}
            (\text{pull}_{k,s} - \text{add}_{k,s}) \, w(k)

    .. math::
        V_{\text{bid,below}}(t) = \sum_{\substack{k < 0 \\ s = B}}
            (\text{pull}_{k,s} - \text{add}_{k,s}) \, w(k)

    **Institutional drain** (resting cancel intensity):

    .. math::
        D_{\text{ask}} = \sum_{k>0,\, s=A} \text{pull\_qty\_rest}_{k,s} \, w(k)

    **Flow imbalance** (positive => bid-dominated):

    .. math::
        F = \sum_{|k| \le N,\, s=B} \text{add}_B \, w(k)
          - \sum_{|k| \le N,\, s=A} \text{add}_A \, w(k)

    **Fill imbalance** (positive => buy aggression):

    .. math::
        \Phi = \sum_{s=A} \text{fill}_A - \sum_{s=B} \text{fill}_B

    (fill on ask = buyer lifted offer; fill on bid = seller hit bid)

    **Depth imbalance** (positive => bid-heavy):

    .. math::
        \Delta = \frac{\text{bid\_depth} - \text{ask\_depth}}
                      {\text{bid\_depth} + \text{ask\_depth} + \varepsilon}

    **Migration centroids** (center-of-mass of new order placement):

    .. math::
        \text{CoM}_B = \frac{\sum_B k \cdot \text{add}_B}
                            {\sum_B \text{add}_B + \varepsilon}

    Args:
        df_flow: Silver ``depth_and_flow_1s`` DataFrame.
        bucket_size_dollars: Bucket size in dollars for range conversion.

    Returns:
        DataFrame with one row per window, sorted by ``window_end_ts_ns``.
    """
    df = df_flow.copy()
    df = df[df["window_valid"]].copy()
    if df.empty:
        return pd.DataFrame()

    tau = _proximity_tau_ticks(bucket_size_dollars)
    near_ticks = _near_spot_ticks(bucket_size_dollars)
    depth_ticks = _depth_range_ticks(bucket_size_dollars)

    # Vectorised masks
    rt = df["rel_ticks"].values
    side = df["side"].values
    w = proximity_weight(rt, tau)
    add = df["add_qty"].values
    pull = df["pull_qty"].values
    fill = df["fill_qty"].values
    pull_rest = df["pull_qty_rest"].values
    depth_end = df["depth_qty_end"].values
    depth_rest = df["depth_qty_rest"].values

    is_ask = side == "A"
    is_bid = side == "B"
    above = rt > 0
    below = rt < 0
    near = np.abs(rt) <= near_ticks
    depth_rng = np.abs(rt) <= depth_ticks

    drain = pull - add

    # Pre-compute conditional weighted columns
    df["_vac_ask_above"] = np.where(is_ask & above, drain * w, 0.0)
    df["_vac_bid_below"] = np.where(is_bid & below, drain * w, 0.0)
    df["_drain_ask"] = np.where(is_ask & above, pull_rest * w, 0.0)
    df["_drain_bid"] = np.where(is_bid & below, pull_rest * w, 0.0)
    df["_near_bid_adds"] = np.where(is_bid & near, add * w, 0.0)
    df["_near_ask_adds"] = np.where(is_ask & near, add * w, 0.0)
    df["_ask_fills"] = np.where(is_ask, fill, 0.0)
    df["_bid_fills"] = np.where(is_bid, fill, 0.0)
    df["_near_bid_depth"] = np.where(is_bid & depth_rng, depth_end, 0.0)
    df["_near_ask_depth"] = np.where(is_ask & depth_rng, depth_end, 0.0)
    df["_near_bid_rest"] = np.where(is_bid & depth_rng, depth_rest, 0.0)
    df["_near_ask_rest"] = np.where(is_ask & depth_rng, depth_rest, 0.0)

    rt_f = rt.astype(np.float64)
    df["_bid_add_mom"] = np.where(is_bid, rt_f * add, 0.0)
    df["_bid_add_tot"] = np.where(is_bid, add, 0.0)
    df["_ask_add_mom"] = np.where(is_ask, rt_f * add, 0.0)
    df["_ask_add_tot"] = np.where(is_ask, add, 0.0)
    df["_w_depth"] = depth_end * w

    # Group-by window
    agg_map = {
        "_vac_ask_above": "sum",
        "_vac_bid_below": "sum",
        "_drain_ask": "sum",
        "_drain_bid": "sum",
        "_near_bid_adds": "sum",
        "_near_ask_adds": "sum",
        "_ask_fills": "sum",
        "_bid_fills": "sum",
        "_near_bid_depth": "sum",
        "_near_ask_depth": "sum",
        "_near_bid_rest": "sum",
        "_near_ask_rest": "sum",
        "_bid_add_mom": "sum",
        "_bid_add_tot": "sum",
        "_ask_add_mom": "sum",
        "_ask_add_tot": "sum",
        "_w_depth": "sum",
        "spot_ref_price_int": "first",
    }
    m = df.groupby("window_end_ts_ns").agg(agg_map).reset_index()

    # Rename vacuum / drain columns
    m = m.rename(columns={
        "_vac_ask_above": "vacuum_above",
        "_vac_bid_below": "vacuum_below",
        "_drain_ask": "resting_drain_ask",
        "_drain_bid": "resting_drain_bid",
    })

    # Derived metrics
    m["flow_imbalance"] = m["_near_bid_adds"] - m["_near_ask_adds"]
    m["fill_imbalance"] = m["_ask_fills"] - m["_bid_fills"]

    bd = m["_near_bid_depth"].values
    ad = m["_near_ask_depth"].values
    m["depth_imbalance"] = (bd - ad) / (bd + ad + EPS)

    br = m["_near_bid_rest"].values
    ar = m["_near_ask_rest"].values
    m["rest_depth_imbalance"] = (br - ar) / (br + ar + EPS)

    m["bid_migration_com"] = m["_bid_add_mom"] / (m["_bid_add_tot"] + EPS)
    m["ask_migration_com"] = m["_ask_add_mom"] / (m["_ask_add_tot"] + EPS)
    m["total_weighted_depth"] = m["_w_depth"]

    # Drop intermediates
    m = m.drop(columns=[c for c in m.columns if c.startswith("_")])

    return m.sort_values("window_end_ts_ns").reset_index(drop=True)


def compute_composite_signal(metrics: pd.DataFrame) -> pd.DataFrame:
    r"""Compute composite directional signal from individual metrics.

    .. math::
        S(t) = \alpha_1 \, V_{\text{ask,above}}
             - \alpha_1 \, V_{\text{bid,below}}
             + \alpha_2 \, F
             + \alpha_3 \, \Phi
             + \alpha_4 \, (D_{\text{ask}} - D_{\text{bid}})
             + \alpha_5 \, \Delta_{\text{rest}}

    **Sign convention**: positive = bullish, negative = bearish.

    Confidence is the fraction of component signs agreeing with the
    composite direction (0-1 scale).

    Args:
        metrics: DataFrame from :func:`aggregate_window_metrics`.

    Returns:
        Copy with additional columns ``composite`` and ``confidence``.
    """
    df = metrics.copy()
    w = COMPOSITE_WEIGHTS

    df["composite"] = (
        w["vacuum"] * df["vacuum_above"]
        - w["vacuum"] * df["vacuum_below"]
        + w["flow"] * df["flow_imbalance"]
        + w["fill"] * df["fill_imbalance"]
        + w["institutional_drain"] * (
            df["resting_drain_ask"] - df["resting_drain_bid"]
        )
        + w["rest_depth"] * df["rest_depth_imbalance"]
    )

    # Confidence: fraction of major components agreeing in sign
    comp_sign = np.sign(df["composite"].values)
    components = np.column_stack([
        df["vacuum_above"].values,
        -df["vacuum_below"].values,
        df["flow_imbalance"].values,
        df["fill_imbalance"].values,
        (df["resting_drain_ask"].values - df["resting_drain_bid"].values),
    ])
    c_signs = np.sign(components)
    # Avoid divide-by-zero when composite == 0
    safe_sign = np.where(comp_sign == 0, 1.0, comp_sign)
    agreement = (c_signs == safe_sign[:, np.newaxis]).astype(np.float64)
    df["confidence"] = agreement.mean(axis=1)
    df.loc[np.abs(df["composite"]) < 1.0, "confidence"] = 0.0

    return df


def compute_derivatives(
    df: pd.DataFrame,
    signal_col: str = "composite",
    d1_span: int = EMA_SPAN_D1,
    d2_span: int = EMA_SPAN_D2,
    d3_span: int = EMA_SPAN_D3,
) -> pd.DataFrame:
    r"""Compute 1st, 2nd, 3rd derivatives of a signal using EMA smoothing.

    .. math::
        \dot{S}     &= \text{EMA}_{\alpha_1}(S_t - S_{t-1})   \\
        \ddot{S}    &= \text{EMA}_{\alpha_2}(\dot{S}_t - \dot{S}_{t-1})  \\
        \dddot{S}   &= \text{EMA}_{\alpha_3}(\ddot{S}_t - \ddot{S}_{t-1})

    where :math:`\alpha_i = 2 / (\text{span}_i + 1)`.

    Progressively longer spans for higher derivatives combat noise
    amplification inherent in numerical differentiation.

    Args:
        df: DataFrame sorted by ``window_end_ts_ns``.
        signal_col: Column to differentiate.
        d1_span: EMA span for 1st derivative.
        d2_span: EMA span for 2nd derivative.
        d3_span: EMA span for 3rd derivative.

    Returns:
        Copy with columns ``d1_{col}``, ``d2_{col}``, ``d3_{col}``.
    """
    df = df.copy()
    sig = df[signal_col]

    raw_d1 = sig.diff()
    df[f"d1_{signal_col}"] = raw_d1.ewm(span=d1_span, adjust=False).mean()

    raw_d2 = df[f"d1_{signal_col}"].diff()
    df[f"d2_{signal_col}"] = raw_d2.ewm(span=d2_span, adjust=False).mean()

    raw_d3 = df[f"d2_{signal_col}"].diff()
    df[f"d3_{signal_col}"] = raw_d3.ewm(span=d3_span, adjust=False).mean()

    return df


def compute_signal_strength(
    df: pd.DataFrame,
    signal_col: str = "composite",
    window: int = ZSCORE_WINDOW,
) -> pd.DataFrame:
    r"""Compute z-score normalised signal strength.

    .. math::
        z(t) = \frac{S(t) - \bar{S}_w}{\sigma_{S,w} + \varepsilon}

    .. math::
        \text{strength}(t) = \min\!\bigl(1,\; |z(t)| / 3\bigr)

    Args:
        df: DataFrame with composite signal, sorted by time.
        signal_col: Column to normalise.
        window: Rolling window size in rows (1 row = 1 s).

    Returns:
        Copy with columns ``z_{col}`` and ``strength``.
    """
    df = df.copy()
    sig = df[signal_col]

    rmean = sig.rolling(window=window, min_periods=1).mean()
    rstd = sig.rolling(window=window, min_periods=1).std().fillna(0)

    z = (sig - rmean) / (rstd + EPS)
    df[f"z_{signal_col}"] = z
    df["strength"] = np.minimum(1.0, np.abs(z.values) / 3.0)

    return df


# ──────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────


def run_full_pipeline(
    df_flow: pd.DataFrame,
    df_snap: pd.DataFrame,
    config: VPRuntimeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete vacuum / pressure computation pipeline.

    Pipeline stages:
        1. Enrich per-bucket data with vacuum / pressure scores.
        2. Aggregate to per-window metrics.
        3. Compute composite directional signal.
        4. Compute 1st / 2nd / 3rd derivatives.
        5. Compute signal strength (z-score normalised).
        6. Join spot reference from book snapshots.

    All spatial parameters are derived from ``config.bucket_size_dollars``
    so the same formulas work correctly across equity and futures instruments.

    Args:
        df_flow: Silver ``depth_and_flow_1s`` DataFrame.
        df_snap: Silver ``book_snapshot_1s`` DataFrame.
        config: Resolved runtime config.

    Returns:
        ``(df_signals, df_flow_enriched)``

        * *df_signals*: One row per window with all computed signals.
        * *df_flow_enriched*: Per-bucket data with vacuum / pressure scores.
    """
    bucket = config.bucket_size_dollars

    # Step 1: per-bucket enrichment
    df_flow_enriched = compute_per_bucket_scores(df_flow, bucket)

    # Step 2: aggregate to per-window
    df_signals = aggregate_window_metrics(df_flow, bucket)
    if df_signals.empty:
        return df_signals, df_flow_enriched

    # Step 3: composite
    df_signals = compute_composite_signal(df_signals)

    # Step 4: derivatives on composite and key sub-signals
    df_signals = compute_derivatives(df_signals, "composite")
    for col in ("vacuum_above", "vacuum_below", "flow_imbalance"):
        if col in df_signals.columns:
            df_signals = compute_derivatives(df_signals, col)

    # Step 5: signal strength
    df_signals = compute_signal_strength(df_signals)

    # Step 6: join spot reference
    if not df_snap.empty:
        snap_cols = [
            "window_end_ts_ns", "mid_price",
            "best_bid_price_int", "best_ask_price_int", "book_valid",
        ]
        avail = [c for c in snap_cols if c in df_snap.columns]
        df_signals = df_signals.merge(
            df_snap[avail], on="window_end_ts_ns", how="left",
        )

    # Fill NaN warmup period
    deriv_cols = [
        c for c in df_signals.columns
        if c.startswith(("d1_", "d2_", "d3_", "z_"))
    ]
    df_signals[deriv_cols] = df_signals[deriv_cols].fillna(0.0)
    df_signals["strength"] = df_signals["strength"].fillna(0.0)
    df_signals["confidence"] = df_signals["confidence"].fillna(0.0)

    return df_signals, df_flow_enriched
