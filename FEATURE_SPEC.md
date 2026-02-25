# qMachina Feature Spec (Rebased to Platform Reality)
Last updated: 2026-02-25

## 1) Product Thesis
qMachina is an event-level market microstructure research and serving platform:
- ingest Databento MBO data
- reconstruct L3 state
- emit fixed-bin silver grids
- compute gold features
- run experiment sweeps
- promote immutable serving specs
- stream to a low-latency visualization client

The business objective is to compress model iteration time while preserving train-serve parity and promotion discipline.

## 2) Current Platform State (Validated)

### 2.1 Bronze Ingestion
Status: Partial
- Implemented:
  - Raw Databento `.dbn` replay from lake storage.
  - Session and symbol filtering.
  - Single-instrument runtime lock (`QMACHINA_INSTRUMENT_CONFIG_PATH`).
  - Two runtime speeds by workflow:
    - stream server path: wall-clock paced bins
    - offline generate/experiment path: fast-forward replay
- Not implemented:
  - direct live Databento streaming ingestion path for production serving.

### 2.2 L3 Reconstruction and Engine
Status: Implemented
- Full event-driven L3 reconstruction with order-level state.
- Absolute tick engine with configurable depth (`n_absolute_ticks`).
- Book cache checkpointing and re-anchor for startup speed.
- No snapshot-based approximations in the core event loop.

### 2.3 Silver Layer (Wire Contract)
Status: Implemented with scope constraints
- Canonical silver schema emitted over WebSocket as Arrow IPC.
- Per-cell mechanics and derivatives:
  - mass/depth fields
  - velocity/acceleration/jerk chains
  - BBO permutation labels
- Hard stage boundary: silver on wire, gold off wire.
- Gap versus original draft:
  - no generic "silver presets" catalog productized in UI.
  - no VWAP/order-count/net-delta canonical silver fields today.

### 2.4 Gold Layer
Status: Partial
- Implemented:
  - deterministic VP force block + derivatives + flow scoring.
  - frontend in-memory gold runtime for stream path.
  - offline `generate-gold` path for dataset parity.
  - config values sourced from runtime snapshot for parity.
- Not implemented:
  - general declarative gold operator DSL (arbitrary neighborhoods, custom feature graphs).
  - full "flat feature vector schema authoring" workflow in product UI.

### 2.5 Experimentation and Tracking
Status: Implemented and broader than prior draft
- Implemented:
  - YAML config chain: `PipelineSpec -> ServingSpec -> ExperimentSpec`.
  - sweep expansion over scoring/signal params, TP/SL, cooldown.
  - parallel execution, persisted results DB, MLflow logging.
  - experiment browser REST API + frontend table/detail UI.
  - optional Feast offline feature retrieval.
  - online simulation mode for latency budget profiling.
- Important reality:
  - platform supports both statistical signals and ML train/predict signals.
  - this is more advanced than the prior "no model training" framing.

### 2.6 Promotion and Serving
Status: Implemented
- Immutable published serving specs with runtime snapshots.
- Alias registry with full promotion audit and rollback by alias repoint.
- Stream contract enforces `?serving=<alias_or_id>` only.
- Model registry supports multiple serving models (`vacuum_pressure`, `ema_ensemble`).
- Gap versus prior wording:
  - serving can also be registered directly from a serving YAML (not only via promote from experiment run).

### 2.7 Frontend Serving and Visualization
Status: Implemented
- Two display modes:
  - heatmap
  - candle
- Overlay system with ordered layers.
- Historical projection overlay is already implemented:
  - forward projection zone
  - faded historical projection envelopes
  - works across render modes via overlay pipeline.
- Runtime-config-driven rendering with strict stream contract checks.

### 2.8 "Create a Model" Product Surface
Status: Not implemented
- No standalone guided model creation UI.
- No in-product stepper funnel for:
  - silver selection
  - gold authoring
  - parameter-space design
  - execution monitoring
  - promotion
- No AI-guided conversational workflow in product.

## 3) Critical Execution Gaps (What to Build Next)

### Priority 0 (Must Build for Institutional Readiness)
1. Live ingestion service
- Build a true live Databento ingestion path with failover, reconnect, and contract-roll handling.
- Why: replay-only serving is not enough for production alpha operations.

2. Model creation control plane
- Build first-class "Create a Model" workflow UI with explicit gates.
- Must include manual YAML mode and guided mode scaffolding.
- Why: current CLI-first workflow does not scale to product users.

3. Experiment job orchestration layer
- Introduce queued jobs, run states, cancellation, resume, and cost controls.
- Stream progress/events to UI in real time.
- Why: long-running sweeps cannot remain terminal-bound.

4. Multi-tenant workspace and governance
- Add authn/authz, workspace isolation, quotas, and audit logging.
- Why: required for collaboration and enterprise deployment.

5. Config contract hardening and CI gate
- Add strict CI validation for every pipeline/serving/experiment YAML in repo.
- Immediate issue to fix: `derivative_baseline.yaml` currently fails ServingSpec validation under the current enum contract.
- Why: broken reference configs undermine trust and block repeatable workflows.

### Priority 1 (Product Differentiation and Retention)
6. Gold authoring DSL and preview engine
- Introduce declarative operator graph for gold features, with preview/distribution diagnostics.
- Why: unlocks strategy expressiveness beyond fixed VP formulas.

7. Guided assistant for modeling workflow
- Add AI advisor at each gate with explainable suggestions, user confirmation required.
- Why: reduces expertise barrier and onboarding time.

8. Parameter sensitivity and diagnostics UX
- Add first-class sensitivity plots, feature redundancy flags, and follow-up sweep suggestions.
- Why: improves decision quality at promotion time.

9. Serving spec lifecycle UX
- Build registry UI for compare/activate/rollback across serving versions.
- Why: makes promotion history operationally usable beyond CLI.

### Priority 2 (Scale and Monetization)
10. Collaboration and sharing primitives
- Shared assets, permissions, and reproducible handoff bundles across users/workspaces.
- Why: foundation for team workflows and monetizable collaboration.

11. Automated model search layer
- AutoML-like sweep templates over signal families and hyperparameter regimes.
- Why: compounds user productivity and platform lock-in.

12. Post-promotion production analytics
- Drift monitoring, live-vs-backtest divergence, latency SLO tracking, incident alerts.
- Why: required for serious production trading operations.

## 4) Canonical Product Decisions Going Forward
1. Keep hard silver/gold separation.
2. Keep immutable runtime snapshots for live serving.
3. Keep promotion and alias indirection as the only activation mechanism in production.
4. Preserve train-serve parity as a non-negotiable invariant.
5. Treat "Create a Model" as a standalone product surface, not an add-on panel in stream visualization.

## 5) Definition of Success for Next Release Train
- A user can create, execute, compare, and promote a model entirely from product UI without CLI.
- Live ingestion can serve promoted specs on real-time market data.
- Every promoted serving spec is reproducible from lineage (dataset + config + params + code version).
- Promotion safety checks prevent invalid or stale configs from reaching runtime.
