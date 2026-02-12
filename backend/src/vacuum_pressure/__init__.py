"""Vacuum & Pressure Detection for MBO Order Flow.

Detects directional micro-regimes by analyzing liquidity dynamics:
    Vacuum: Regions where liquidity is thinning (orders pulled > added).
    Pressure: Regions where liquidity is building and migrating toward spot.

Supports both equity_mbo and future_mbo product types via runtime
configuration resolved from products.yaml and equity defaults.

Modules:
    config: Runtime instrument configuration resolver.
    formulas: Core metric computations (pure functions on DataFrames).
    engine: Pipeline orchestration (read silver -> compute -> output).
    server: WebSocket streaming endpoint (standalone FastAPI app).
"""
