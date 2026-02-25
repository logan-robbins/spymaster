# qMachina Implementation Program
Last Updated: 2026-02-25

## 1. Purpose
This document is the implementation blueprint for the roadmap items defined in `FEATURE_SPEC.md`.

It is optimized for execution by AI coding agents working in parallel. Every workstream is broken into:
1. Target outcome.
2. Design decisions that should not be revisited by implementers.
3. Required interfaces and storage changes.
4. Explicit task packets for AI coding agents.
5. Test and acceptance criteria.

The goal is decision-complete implementation planning: an implementer should not need to invent architecture choices.

## 2. Program Execution Model
### 2.1 Milestones
1. `M0` Platform hardening.
2. `M1` Institutional MVP.
3. `M2` Product differentiation.
4. `M3` Scale and monetization.

### 2.2 Recommended Timeline
1. `M0` 3-4 weeks.
2. `M1` 6-8 weeks.
3. `M2` 6-8 weeks.
4. `M3` 8-10 weeks.

### 2.3 Engineering Governance
1. All new APIs must be OpenAPI-declared and contract tested.
2. All new config/spec changes must support strict validation and deterministic hashing.
3. No hidden mutable runtime state for promoted serving versions.
4. Every user-visible mutation is audited.
5. Every long-running operation must be jobified and cancellable.

### 2.4 Canonical Technology Defaults
1. Backend: FastAPI.
2. Worker orchestration: Redis queue with dedicated worker service.
3. Auth: OIDC JWT middleware with workspace-scoped RBAC.
4. Job progress transport: SSE first, WS optional later.
5. Storage: Postgres for control-plane metadata, parquet/lake for heavy experiment artifacts.
6. Metrics: Prometheus-style counters/histograms + structured logs.

## 3. Shared Foundations Before Workstreams
These are cross-cutting prerequisites and should be implemented first.

### 3.1 Control Plane Database
Create a Postgres-backed control-plane schema:
1. `workspace`.
2. `workspace_member`.
3. `experiment_job`.
4. `job_event`.
5. `modeling_session`.
6. `modeling_step_state`.
7. `ingestion_live_session`.
8. `serving_activation`.
9. `audit_event`.

### 3.2 API Versioning
1. Keep existing routes stable.
2. Add new control-plane routes under `/v1/...`.
3. If breaking changes are required later, isolate under `/v2`.

### 3.3 Observability Baseline
All services must emit:
1. Request latency and status metrics.
2. Queue depth and worker throughput.
3. Job success/fail/cancel counts.
4. Stream lag and dropped frame counters.
5. Audit event write failures.

### 3.4 AI-Agent Task Execution Contract
All AI coding agents must follow this packet format:
1. `Inputs`.
2. `Outputs`.
3. `Touched paths`.
4. `Invariants to preserve`.
5. `Verification command`.

## 4. Workstream 1: Live Ingestion Service
### 4.1 Outcome
Support real-time Databento ingestion with reconnect, checkpoint resume, and contract-roll handling.

### 4.2 Non-Negotiable Decisions
1. Live ingestion is session-based and explicitly start/stop controlled.
2. Resume checkpoints are persisted and idempotent.
3. Contract-roll is policy-driven and auditable.
4. Replay and live source adapters expose the same event envelope to downstream pipelines.

### 4.3 Required Interfaces
API:
1. `POST /v1/ingestion/live/sessions`.
2. `GET /v1/ingestion/live/sessions/{session_id}`.
3. `POST /v1/ingestion/live/sessions/{session_id}/stop`.

Schema:
1. `ingestion_live_session(session_id, workspace_id, symbol, status, started_at, stopped_at, checkpoint, config_json)`.
2. `ingestion_checkpoint(session_id, sequence_id, ts_event_ns, updated_at)`.
3. `contract_roll_event(session_id, from_contract, to_contract, reason, created_at)`.

### 4.4 AI-Agent Task Packets
1. Agent `ING-1` transport client.
Task: implement resilient Databento live client wrapper with heartbeat watchdog and reconnect backoff.
Output: `backend/src/ingestion/live/client.py`.
Verify: simulated disconnect test.

2. Agent `ING-2` session manager.
Task: session lifecycle service with start/stop/status and persisted checkpoints.
Output: `backend/src/ingestion/live/session_manager.py`, DB access layer.
Verify: restart process and ensure checkpoint resume.

3. Agent `ING-3` adapter bridge.
Task: map live feed into existing stream event generator interface used by models.
Output: source adapter module + integration in model source registry.
Verify: same downstream schema as replay path.

4. Agent `ING-4` contract roll policy.
Task: roll trigger policy and roll event emission.
Output: `backend/src/ingestion/live/contract_roll.py`.
Verify: deterministic roll simulation over synthetic schedule.

5. Agent `ING-5` API layer.
Task: expose session control routes and wire to RBAC.
Output: `backend/src/qmachina/api_ingestion.py` and `app.py` registration.
Verify: API integration tests.

### 4.5 Acceptance Tests
1. Live session recovers from dropped connection without duplicate event replay.
2. Stop call moves session to terminal state and closes resources.
3. Contract roll logs event and transitions source cleanly.
4. Stream consumers receive continuous `runtime_config` + `grid_update` contract.

## 5. Workstream 2: Create Model Control Plane UI
### 5.1 Outcome
Deliver a standalone “Create a Model” experience with hard step gates and auditable decisions.

### 5.2 Non-Negotiable Decisions
1. This is a separate workflow from stream visualization pages.
2. Steps are lock-gated; no bypass via direct API calls.
3. Every step commit is persisted.
4. Promotion can only occur from completed modeling session state.

### 5.3 Required Interfaces
API:
1. `POST /v1/modeling/sessions`.
2. `GET /v1/modeling/sessions/{session_id}`.
3. `POST /v1/modeling/sessions/{session_id}/steps/{step_name}/commit`.
4. `POST /v1/modeling/sessions/{session_id}/promote`.

Schema:
1. `modeling_session(session_id, workspace_id, status, selected_silver_id, created_at)`.
2. `modeling_step_state(session_id, step_name, status, payload_json, committed_at)`.
3. `modeling_decision_log(session_id, step_name, actor_type, decision_json, created_at)`.

### 5.4 AI-Agent Task Packets
1. Agent `MC-1` backend state machine.
Task: implement strict workflow state transitions.
Output: backend service `modeling_session_service.py`.
Verify: forbidden transition tests.

2. Agent `MC-2` frontend shell.
Task: add route, page shell, and stepper with lock-state rendering.
Output: new frontend page + router wiring.
Verify: unit tests for gate transitions.

3. Agent `MC-3` manual mode.
Task: YAML authoring editor with schema validation and inline errors.
Output: editor component + validator service hook.
Verify: invalid YAML error path tests.

4. Agent `MC-4` replay sample preview.
Task: fetch sample bins/grid and render summary distributions.
Output: preview panel and backend endpoint.
Verify: deterministic snapshot tests with fixture dataset.

5. Agent `MC-5` promotion bridge.
Task: map completed session into existing promotion mechanics.
Output: service adapter to promote API path.
Verify: end-to-end modeling session -> serving alias created.

### 5.5 Acceptance Tests
1. User can complete all six modeling stages without CLI.
2. Out-of-order step commit returns validation error.
3. Promotion fails if any required step is uncommitted.
4. Decision logs are persisted and retrievable.

## 6. Workstream 3: Experiment Job Orchestration
### 6.1 Outcome
Replace terminal-bound experiment execution with managed queue jobs and real-time progress.

### 6.2 Non-Negotiable Decisions
1. Existing runner stays as compute core; orchestration wraps it.
2. Jobs are persisted with terminal outcomes.
3. Cancellation is cooperative and reliable.
4. Logs and artifacts are first-class entities.

### 6.3 Required Interfaces
API:
1. `POST /v1/jobs/experiments`.
2. `GET /v1/jobs/experiments/{job_id}`.
3. `POST /v1/jobs/experiments/{job_id}/cancel`.
4. `GET /v1/jobs/experiments/{job_id}/events`.

Schema:
1. `experiment_job(job_id, workspace_id, spec_ref, status, progress_json, started_at, completed_at)`.
2. `job_event(job_id, sequence, event_type, payload_json, created_at)`.
3. `job_artifact(job_id, artifact_type, uri, checksum, created_at)`.

### 6.4 AI-Agent Task Packets
1. Agent `JOB-1` queue adapter.
Task: Redis queue producer/consumer wrappers.
Output: `backend/src/jobs/queue.py`.
Verify: enqueue/dequeue integration tests.

2. Agent `JOB-2` job runner wrapper.
Task: wrap experiment runner with periodic progress emission and cancel checks.
Output: `backend/src/jobs/experiment_job_runner.py`.
Verify: progress monotonicity and cancel integration tests.

3. Agent `JOB-3` API endpoints.
Task: job submission/status/cancel/events routes.
Output: `api_jobs.py` plus app registration.
Verify: API contract tests.

4. Agent `JOB-4` frontend live monitor.
Task: consume SSE and render progress states and metrics stream.
Output: job monitor components.
Verify: mocked SSE behavior tests.

5. Agent `JOB-5` artifact persistence.
Task: attach run logs/results references to job record.
Output: artifact metadata store and retrieval API.
Verify: artifact integrity checks.

### 6.5 Acceptance Tests
1. Multiple jobs execute concurrently within configured worker limits.
2. Cancel transitions to `canceled` and no new run specs execute.
3. Client receives ordered progress events.
4. Failed jobs preserve error trace and partial artifacts.

## 7. Workstream 4: Multi-Tenant Workspace and Governance
### 7.1 Outcome
Add secure workspace isolation with role-based controls and quotas.

### 7.2 Non-Negotiable Decisions
1. Every request is workspace-scoped.
2. Authorization is deny-by-default.
3. Every privileged mutation writes an audit event.
4. Cross-workspace reads are impossible via query guards and service checks.

### 7.3 Required Interfaces
API:
1. `POST /v1/workspaces`.
2. `GET /v1/workspaces/{workspace_id}`.
3. `POST /v1/workspaces/{workspace_id}/members`.
4. `POST /v1/workspaces/{workspace_id}/quotas`.

Schema:
1. `workspace`.
2. `workspace_member(user_id, workspace_id, role)`.
3. `workspace_quota(workspace_id, max_jobs, max_streams, max_storage_gb)`.
4. `audit_event(event_id, workspace_id, actor_id, action, target, payload_json, created_at)`.

### 7.4 AI-Agent Task Packets
1. Agent `TEN-1` auth middleware.
Task: JWT verification and user identity extraction.
Output: shared auth middleware.
Verify: auth bypass rejection tests.

2. Agent `TEN-2` RBAC policy layer.
Task: role-policy checks for each action type.
Output: policy module + decorators.
Verify: role matrix tests.

3. Agent `TEN-3` data scoping.
Task: retrofit workspace filters into all service/repository queries.
Output: workspace-aware repos.
Verify: cross-tenant access tests.

4. Agent `TEN-4` quota enforcement.
Task: enforce quotas on job submission and stream starts.
Output: quota service.
Verify: over-quota rejection tests.

5. Agent `TEN-5` audit logger.
Task: centralized mutation audit writer with fallback handling.
Output: audit service and route integration.
Verify: audit persistence tests.

### 7.5 Acceptance Tests
1. Viewer cannot mutate resources.
2. Editor cannot change workspace membership.
3. Admin/owner actions are audited.
4. Quota boundaries are strictly enforced.

## 8. Workstream 5: Config Contract Hardening and CI Gate
### 8.1 Outcome
Prevent invalid pipeline/serving/experiment specs from entering runtime.

### 8.2 Known Immediate Defect
`derivative_baseline.yaml` uses legacy stream roles that fail current enum validation.

### 8.3 Non-Negotiable Decisions
1. Config validation runs in CI on every change.
2. Promotion path executes strict preflight validation.
3. Backward-compat support must be explicit and tested, not implicit.

### 8.4 AI-Agent Task Packets
1. Agent `CFG-1` validation CLI.
Task: implement `uv run python -m src.qmachina.validate_configs`.
Output: config validator module.
Verify: fail/pass fixture suite.

2. Agent `CFG-2` enum migration shim.
Task: either migrate all legacy configs or introduce controlled alias mapping with deprecation warning.
Output: migration script or parser shim.
Verify: derivative baseline parses successfully.

3. Agent `CFG-3` CI integration.
Task: add pipeline step validating all YAML assets.
Output: CI config update.
Verify: intentionally broken fixture fails CI.

4. Agent `CFG-4` promotion preflight.
Task: enforce validator call before writing serving version.
Output: promote/register guard logic.
Verify: invalid spec promotion blocked.

5. Agent `CFG-5` golden compatibility tests.
Task: snapshot expected parser behavior for current canonical specs.
Output: tests under `backend/tests`.
Verify: compatibility regression test suite.

### 8.5 Acceptance Tests
1. All config directories validate cleanly in CI.
2. Legacy role mismatch is resolved.
3. Promotion cannot bypass validation.

## 9. Workstream 6: Gold DSL and Preview Engine
### 9.1 Outcome
Provide declarative feature authoring beyond fixed VP formulas.

### 9.2 DSL v1 Scope
1. Source references to silver fields.
2. Temporal windows.
3. Spatial neighborhoods.
4. Arithmetic expressions.
5. Normalization operators.
6. Named outputs with deterministic ordering.

### 9.3 Non-Negotiable Decisions
1. No arbitrary user code execution in DSL.
2. Graph must be acyclic and type-safe.
3. DSL versioning is immutable and hash-addressed.

### 9.4 Required Interfaces
API:
1. `POST /v1/gold/validate`.
2. `POST /v1/gold/preview`.
3. `POST /v1/gold/lineage/compare`.

Schema:
1. `gold_spec_version(spec_id, version, hash, dsl_json, created_at)`.
2. `gold_preview_stat(spec_id, dataset_id, stat_json, created_at)`.

### 9.5 AI-Agent Task Packets
1. Agent `GDSL-1` schema and parser.
Task: define Pydantic models for DSL nodes and parser.
Output: `backend/src/gold_dsl/schema.py`.
Verify: valid/invalid fixture tests.

2. Agent `GDSL-2` validator.
Task: cycle detection, field existence checks, type checks.
Output: `backend/src/gold_dsl/validate.py`.
Verify: graph-validation tests.

3. Agent `GDSL-3` preview executor.
Task: run DSL against sample dataset windows and return statistics.
Output: preview service.
Verify: deterministic numeric snapshot tests.

4. Agent `GDSL-4` lineage bridge.
Task: include DSL hash/version in serving runtime snapshot.
Output: promotion integration changes.
Verify: runtime config payload includes DSL lineage.

5. Agent `GDSL-5` compatibility mapper.
Task: map legacy VP gold config into DSL equivalent.
Output: adapter module and tests.
Verify: parity tests between legacy and DSL outputs.

### 9.6 Acceptance Tests
1. DSL validation catches cycles and unknown fields.
2. Preview outputs stable stats for fixed fixture data.
3. Legacy and DSL feature parity confirmed for baseline configs.

## 10. Workstream 7: Guided Assistant Workflow
### 10.1 Outcome
Deliver AI advisor functionality inside model creation without automation risk.

### 10.2 Non-Negotiable Decisions
1. Advisor is recommendation-only.
2. Every execution/promotion requires explicit user action.
3. Suggestions must include rationale and confidence.

### 10.3 Required Interfaces
API:
1. `POST /v1/modeling/sessions/{session_id}/advisor/suggest`.
2. `POST /v1/modeling/sessions/{session_id}/advisor/explain`.

Schema:
1. `advisor_message(session_id, step_name, prompt, response_json, created_at)`.
2. `advisor_suggestion(session_id, suggestion_type, payload_json, rationale, confidence, created_at)`.

### 10.4 AI-Agent Task Packets
1. Agent `ADV-1` context assembler.
Task: aggregate silver schema, historical runs, current step state.
Output: context builder service.
Verify: deterministic context snapshot tests.

2. Agent `ADV-2` suggestion contract.
Task: define strict JSON schema for suggestion payloads.
Output: schema module.
Verify: schema validation tests.

3. Agent `ADV-3` backend endpoint integration.
Task: advisor endpoints with guardrails against mutation actions.
Output: API routes and permission checks.
Verify: no side-effect guarantee tests.

4. Agent `ADV-4` frontend guided mode.
Task: conversation UI integrated with step commits.
Output: guided panel and suggestion acceptance controls.
Verify: UX flow tests.

### 10.5 Acceptance Tests
1. Advisor can recommend but cannot execute jobs.
2. Suggestions are persisted and referenceable.
3. User can override any recommendation.

## 11. Workstream 8: Parameter Sensitivity and Diagnostics UX
### 11.1 Outcome
Improve promotion quality with systematic diagnostics.

### 11.2 Non-Negotiable Decisions
1. Sensitivity analysis is computed from persisted run results, not ad hoc UI calculations.
2. Diagnostic flags are explicit and documented.

### 11.3 AI-Agent Task Packets
1. Agent `SENS-1` analytics backend.
Task: compute per-parameter effect summaries and interactions.
Output: analytics service + API.
Verify: fixture-based analytics tests.

2. Agent `SENS-2` diagnostic rules engine.
Task: detect collinearity, low variance, sparse firing, unstable top-rank shifts.
Output: diagnostics module.
Verify: rule test suite.

3. Agent `SENS-3` frontend views.
Task: sensitivity plots, run comparison overlays, diagnostics badges.
Output: experiment analysis UI module.
Verify: component tests.

### 11.4 Acceptance Tests
1. User can compare 2-3 runs on same window and inspect diagnostics.
2. Promotion page displays sensitivity summary and warnings.

## 12. Workstream 9: Serving Lifecycle UX
### 12.1 Outcome
Operationalize serving version management and rollback in product UI.

### 12.2 Non-Negotiable Decisions
1. Alias activation remains the only runtime switch mechanism.
2. Rollback is alias repoint, never in-place mutation.

### 12.3 AI-Agent Task Packets
1. Agent `SRV-1` lifecycle APIs.
Task: list versions, alias history, activate alias endpoints.
Output: serving lifecycle routes.
Verify: contract tests with sqlite fixture.

2. Agent `SRV-2` snapshot diff service.
Task: compare two runtime snapshots and emit structured diff.
Output: diff utility and API.
Verify: diff tests.

3. Agent `SRV-3` frontend registry UI.
Task: version list, detail panel, activation controls, rollback action.
Output: serving lifecycle page.
Verify: integration tests.

### 12.4 Acceptance Tests
1. Operator can rollback alias to prior serving version with audit trail.
2. Snapshot diff is readable and complete.

## 13. Workstream 10: Collaboration and Sharing
### 13.1 Outcome
Enable workspace-level sharing and reproducible handoff.

### 13.2 Non-Negotiable Decisions
1. Shared assets are immutable references with lineage metadata.
2. Permission inheritance is explicit per asset type.

### 13.3 AI-Agent Task Packets
1. Agent `COL-1` asset model.
Task: implement shareable asset metadata for gold specs, experiments, serving specs.
Output: storage schema and backend service.
Verify: asset CRUD tests.

2. Agent `COL-2` permission model.
Task: grant/revoke/share rules with RBAC integration.
Output: permission service.
Verify: permission matrix tests.

3. Agent `COL-3` handoff bundle export/import.
Task: immutable bundle format with checksums and lineage.
Output: bundle service + API.
Verify: export/import parity tests.

### 13.4 Acceptance Tests
1. Shared asset can be imported with lineage intact.
2. Unauthorized workspace cannot access shared private assets.

## 14. Workstream 11: Automated Model Search
### 14.1 Outcome
Add budgeted AutoML-style sweep orchestration.

### 14.2 Non-Negotiable Decisions
1. Search space templates are explicit and bounded.
2. Budget hard-limits are enforced.
3. Search results are reproducible and promotion-compatible.

### 14.3 AI-Agent Task Packets
1. Agent `AUTO-1` search template schema.
Task: define search template models and validation.
Output: schema module.
Verify: template validation tests.

2. Agent `AUTO-2` scheduler integration.
Task: convert search template into orchestrated experiment jobs.
Output: scheduler service.
Verify: job fan-out tests.

3. Agent `AUTO-3` ranking engine.
Task: multi-objective ranking and top-k candidate extraction.
Output: ranking module.
Verify: ranking determinism tests.

4. Agent `AUTO-4` frontend search workflow.
Task: create search setup and results UI.
Output: UI pages and API wiring.
Verify: UI integration tests.

### 14.4 Acceptance Tests
1. Search obeys configured compute/run budget.
2. Top candidates can be promoted via standard flow.

## 15. Workstream 12: Post-Promotion Production Analytics
### 15.1 Outcome
Provide operational confidence via drift, latency SLOs, and incident visibility.

### 15.2 Non-Negotiable Decisions
1. Health metrics are attached to serving alias/version identity.
2. Alerts are driven by explicit threshold policies.
3. Incident timelines must be reconstructable.

### 15.3 Required Interfaces
API:
1. `GET /v1/monitoring/serving/{serving_id}/health`.
2. `GET /v1/monitoring/serving/{serving_id}/drift`.
3. `GET /v1/monitoring/incidents`.

Schema:
1. `drift_metric`.
2. `latency_slo_window`.
3. `incident_event`.

### 15.4 AI-Agent Task Packets
1. Agent `MON-1` metrics collector.
Task: collect per-alias latency and stream quality stats.
Output: monitoring ingestion service.
Verify: metric emission tests.

2. Agent `MON-2` drift evaluator.
Task: compare live feature distributions to reference baselines.
Output: drift engine.
Verify: drift trigger tests.

3. Agent `MON-3` incident manager.
Task: alert thresholds and incident event timeline persistence.
Output: incident service and API.
Verify: incident lifecycle tests.

4. Agent `MON-4` monitoring dashboard UI.
Task: display health, drift, SLO status, incident history.
Output: monitoring UI page.
Verify: UI integration tests.

### 15.5 Acceptance Tests
1. SLO breach creates alert and incident event.
2. Drift threshold breach is visible in dashboard and API.

## 16. End-to-End Validation Matrix
Run these scenarios before each milestone sign-off:
1. Live ingestion session start -> reconnect -> resume -> stop.
2. Create model session full flow -> experiment job -> analyze -> promote.
3. Alias activation/rollback with audit confirmation.
4. Cross-workspace unauthorized access attempts.
5. Config validation failure path in CI and in runtime preflight.
6. Guided assistant recommendation path with no side effects.
7. Monitoring alert generation path for induced latency breach.

## 17. Rollout Strategy
### 17.1 Feature Flags
1. `ff_live_ingestion`.
2. `ff_model_control_plane`.
3. `ff_experiment_jobs`.
4. `ff_guided_assistant`.
5. `ff_gold_dsl`.
6. `ff_monitoring_ops`.

### 17.2 Environments
1. Dev: all flags enabled.
2. Staging: selective cohort testing.
3. Production: phased rollout by workspace allowlist.

### 17.3 Migration Steps
1. Create control-plane schema migrations.
2. Backfill workspace identity for legacy data.
3. Backfill serving alias activation history from existing registry tables.
4. Run config migration for legacy stream role enums.

## 18. Definition of Done (Program Level)
Program is complete when:
1. Live data can feed promoted serving specs in production.
2. Users can create, run, analyze, and promote from UI only.
3. Tenant boundaries, RBAC, and quotas are enforced.
4. Config CI gate prevents runtime-breaking specs.
5. Serving lifecycle and rollback are operator-friendly in UI.
6. Monitoring exposes health/drift/SLO and incident timelines.

## 19. Explicit Assumptions
1. Existing FastAPI app remains primary backend entry point.
2. Existing experiment harness runner remains compute core, wrapped not replaced.
3. Postgres is available for control-plane metadata.
4. Redis is available for job queueing and event fan-out.
5. Existing frontend remains Vite/TypeScript with additional route surfaces.
6. Backward compatibility for current stream consumers is preserved unless a versioned contract is introduced.
