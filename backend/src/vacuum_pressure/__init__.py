"""Vacuum & Pressure Detection for Equity MBO Order Flow.

Detects directional micro-regimes by analyzing liquidity dynamics:
    Vacuum: Regions where liquidity is thinning (orders pulled > added).
    Pressure: Regions where liquidity is building and migrating toward spot.

Modules:
    formulas: Core metric computations (pure functions on DataFrames).
    engine: Pipeline orchestration (read silver → compute → output).
    server: WebSocket streaming endpoint (standalone FastAPI app).
"""
