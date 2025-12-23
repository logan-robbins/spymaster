# Spymaster: Real-Time Market Physics Engine

Spymaster models how dealer hedging, order-book liquidity, and trade flow interact at SPY 0DTE option levels to estimate BREAK vs BOUNCE outcomes and strength.

**Scope**: SPY 0DTE options only. Every $1 level is actionable, with special focus on PM high/low, 15‑min opening range high/low, SMA‑200/400, VWAP, session high/low, and option walls.

**Primary outputs**: BREAK/BOUNCE probabilities plus a signed strength target, with per‑event context about confluence and dealer mechanics.

---

## Current Backend Capabilities (Summary)

- Barrier physics from ES MBP‑10 depth (liquidity states, delta liquidity, wall ratio).
- Tape physics from ES time & sales (imbalance, velocity, sweep detection).
- Fuel physics from SPY options flow (dealer gamma exposure + effect).
- Dealer mechanics velocity and acceleration (gamma flow velocity, impulses, pressure accel).
- SMA mean‑reversion features for SMA‑200/400, with warmup from prior sessions.
- Confluence metrics across key levels (count, weighted score, min distance, pressure).
- Fluid pressure indicators (liquidity/tape/gamma/reversion/confluence/net).
- Strength targets and time‑to‑threshold metrics; UNDEFINED when lookforward window is incomplete.

---

## Data and Pipeline Snapshot

- Raw sources: Databento ES futures (trades + MBP‑10) and Polygon SPY options.
- Vectorized pipeline builds labeled signals; the authoritative schema and output path are in `backend/features.json`.
- Touch detection uses `CONFIG.TOUCH_BAND`; signals are filtered by `CONFIG.MONITOR_BAND`.
- SMA warmup appends up to `CONFIG.SMA_WARMUP_DAYS` prior weekday sessions for 2‑min SMA calculations.

---

## Feature Contract (Authoritative)

The full schema, dataset statistics, and methodology live in `backend/features.json`. It is the single source of truth for:
- Feature groups (identity, level, context, mean_reversion, confluence, physics, velocity, pressure, approach, outcome).
- Labeling rules (BREAK/BOUNCE/CHOP/UNDEFINED, strength, time‑to‑threshold).
- Dataset statistics and distributions.

---

## ML Workflow (Summary)

- Sequence dataset builder and PatchTST training live under `backend/src/ml/`.
- PatchTST is multi‑task: BREAK/BOUNCE classification + strength regression.
- Experiment tracking is integrated with MLflow and W&B.

See `backend/src/ml/README.md` for the operational details.

---

## Where to Read Next (Authoritative Module Docs)

- `backend/src/core/README.md` — physics engines, scoring, and signal generation.
- `backend/src/common/README.md` — schemas, config contracts, validation.
- `backend/src/ingestor/README.md` — Databento and Polygon ingestion.
- `backend/src/lake/README.md` — bronze/gold storage and parquet outputs.
- `backend/src/gateway/README.md` — streaming service and payload contracts.
- `backend/src/ml/README.md` — ML datasets, PatchTST training, tracking.
