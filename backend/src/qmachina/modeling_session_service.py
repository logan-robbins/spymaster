"""State-machine service for multi-step modeling sessions.

Enforces step ordering, payload validation, and promotion to the serving
registry. The seven modeling steps must be committed in sequence:

    1. dataset_select  -- choose a silver dataset_id
    2. gold_config     -- configure gold feature parameters
    3. signal_select   -- choose signal + hyperparameter ranges
    4. eval_params     -- set TP/SL/cooldown/warmup parameters
    5. run_experiment  -- synthesize spec, submit job, wait for completion
    6. promote_review  -- review configuration with experiment results
    7. promotion       -- finalize and promote to serving alias
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .db.models import ModelingSession, ModelingStepState
from .db.repositories import ModelingSessionRepository
from .serving_config import (
    PublishedServingSpec,
    PublishedServingSource,
    ServingSpec,
)
from .serving_registry import ServingRegistry

logger: logging.Logger = logging.getLogger(__name__)

STEPS_ORDERED: list[str] = [
    "dataset_select",    # 0
    "gold_config",       # 1
    "signal_select",     # 2
    "eval_params",       # 3
    "run_experiment",    # 4
    "promote_review",    # 5
    "promotion",         # 6
]

_STEP_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(STEPS_ORDERED)}


class ModelingSessionService:
    """Orchestrates modeling session lifecycle and step gate enforcement.

    All database operations are delegated to ``ModelingSessionRepository``
    via the ``AsyncSession`` passed to each method. The caller is
    responsible for transaction boundaries (commit/rollback).

    Args:
        lake_root: Root path of the data lake (for dataset registry and
            serving registry operations).
    """

    STEPS_ORDERED: list[str] = STEPS_ORDERED

    def __init__(self, lake_root: Path) -> None:
        self._lake_root = Path(lake_root)
        self._serving_registry = ServingRegistry(lake_root)

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session: Any,
        *,
        workspace_id: uuid.UUID,
        created_by: str = "user",
    ) -> ModelingSession:
        """Create a new draft modeling session.

        Args:
            session: Active ``AsyncSession``.
            workspace_id: Owning workspace UUID.
            created_by: Identity of the actor creating the session.

        Returns:
            The newly created ``ModelingSession`` in ``draft`` status.
        """
        ms = await ModelingSessionRepository.create(
            session,
            workspace_id=workspace_id,
            created_by=created_by,
        )
        logger.info(
            "Created modeling session %s (workspace=%s, actor=%s)",
            ms.session_id,
            workspace_id,
            created_by,
        )
        return ms

    async def get_session(
        self,
        session: Any,
        *,
        session_id: uuid.UUID,
    ) -> ModelingSession | None:
        """Get a modeling session by ID.

        Args:
            session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.

        Returns:
            The ``ModelingSession`` if found, otherwise ``None``.
        """
        return await ModelingSessionRepository.get(
            session, session_id=session_id,
        )

    async def get_session_with_steps(
        self,
        session: Any,
        *,
        session_id: uuid.UUID,
    ) -> tuple[ModelingSession, list[ModelingStepState]]:
        """Get a session and all its step states.

        Args:
            session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.

        Returns:
            Tuple of (ModelingSession, list of ModelingStepState).

        Raises:
            ValueError: If the session does not exist.
        """
        ms = await ModelingSessionRepository.get(
            session, session_id=session_id,
        )
        if ms is None:
            raise ValueError(f"ModelingSession not found: {session_id}")

        steps = await ModelingSessionRepository.get_steps(
            session, session_id=session_id,
        )
        return ms, steps

    # ------------------------------------------------------------------
    # Step commit with gate enforcement
    # ------------------------------------------------------------------

    async def commit_step(
        self,
        session: Any,
        *,
        session_id: uuid.UUID,
        step_name: str,
        payload: dict[str, Any],
    ) -> ModelingStepState:
        """Commit a modeling step with payload data.

        Enforces sequential ordering: all prior steps must be committed
        before a later step can be committed. Re-committing the same
        step is idempotent (updates payload).

        Args:
            session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.
            step_name: One of the six step names in ``STEPS_ORDERED``.
            payload: Arbitrary JSON-serializable step data.

        Returns:
            The committed ``ModelingStepState``.

        Raises:
            ValueError: If ``step_name`` is not recognized or if prior
                steps have not been committed.
        """
        if step_name not in _STEP_INDEX:
            raise ValueError(
                f"Unknown step '{step_name}'. "
                f"Valid steps: {STEPS_ORDERED}"
            )

        ms = await ModelingSessionRepository.get(
            session, session_id=session_id,
        )
        if ms is None:
            raise ValueError(f"ModelingSession not found: {session_id}")

        target_idx: int = _STEP_INDEX[step_name]

        # Gate check: all prior steps must be committed
        if target_idx > 0:
            existing_steps = await ModelingSessionRepository.get_steps(
                session, session_id=session_id,
            )
            committed_names: set[str] = {
                s.step_name
                for s in existing_steps
                if s.status == "committed"
            }
            for prior_idx in range(target_idx):
                prior_name: str = STEPS_ORDERED[prior_idx]
                if prior_name not in committed_names:
                    raise ValueError(
                        f"Cannot commit step '{step_name}' (index {target_idx}): "
                        f"prior step '{prior_name}' (index {prior_idx}) "
                        f"has not been committed yet. "
                        f"Steps must be committed in order: {STEPS_ORDERED}"
                    )

        step = await ModelingSessionRepository.commit_step(
            session,
            session_id=session_id,
            step_name=step_name,
            payload=payload,
        )

        # Transition session from draft to in_progress on first commit
        if ms.status == "draft":
            await ModelingSessionRepository.update_status(
                session, session_id=session_id, status="in_progress",
            )

        logger.info(
            "Committed step '%s' for session %s",
            step_name,
            session_id,
        )
        return step

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    async def promote(
        self,
        session: Any,
        *,
        session_id: uuid.UUID,
        alias: str,
    ) -> dict[str, Any]:
        """Promote a completed modeling session to a serving version.

        All six steps must be committed. Assembles a ``ServingSpec`` from
        the committed payloads, validates it, publishes it via the
        ``ServingRegistry``, and updates the session status to
        ``promoted``.

        Args:
            session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.
            alias: Serving alias to assign (e.g. ``"production"``).

        Returns:
            Dict with ``serving_id``, ``alias``, and ``spec_path``.

        Raises:
            ValueError: If any step is missing or validation fails.
        """
        ms, steps = await self.get_session_with_steps(
            session, session_id=session_id,
        )

        committed_map: dict[str, dict[str, Any]] = {
            s.step_name: s.payload_json or {}
            for s in steps
            if s.status == "committed"
        }

        missing: list[str] = [
            name for name in STEPS_ORDERED
            if name not in committed_map
        ]
        if missing:
            raise ValueError(
                f"Cannot promote session {session_id}: "
                f"steps not committed: {missing}"
            )

        # Gate: run_experiment must be completed before promotion
        run_exp_payload = committed_map.get("run_experiment", {})
        if run_exp_payload.get("status") != "completed":
            raise ValueError(
                f"Cannot promote: run_experiment step has status "
                f"'{run_exp_payload.get('status', 'not_submitted')}'. "
                f"Run and complete the experiment before promoting."
            )

        # Assemble ServingSpec from step payloads
        serving_spec = self._assemble_serving_spec(
            session_id=session_id,
            step_payloads=committed_map,
        )

        # Validate via round-trip
        from .serving_registry import validate_serving_spec_preflight

        validate_serving_spec_preflight(serving_spec)

        # Build published spec
        config_hash = hashlib.sha256(
            json.dumps(
                serving_spec.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]

        run_id = str(session_id).replace("-", "")[:16]
        serving_id = self._serving_registry.build_serving_id(
            experiment_name=f"modeling_session",
            run_id=run_id,
            config_hash=config_hash,
        )

        now_utc = datetime.now(tz=timezone.utc).isoformat()
        published = PublishedServingSpec(
            serving_id=serving_id,
            description=f"Promoted from modeling session {session_id}",
            runtime_snapshot=serving_spec.to_runtime_fields(cell_width_ms=None)
            if not serving_spec.projection.horizons_ms
            else serving_spec.model_dump(mode="json"),
            source=PublishedServingSource(
                run_id=run_id,
                experiment_name="modeling_session",
                config_hash=config_hash,
                promoted_at_utc=now_utc,
                serving_spec_name=serving_spec.name,
                signal_name=(
                    serving_spec.signal.name
                    if serving_spec.signal
                    else None
                ),
            ),
        )

        result = self._serving_registry.promote(
            alias=alias,
            spec=published,
            actor="modeling_studio",
        )

        await ModelingSessionRepository.update_status(
            session, session_id=session_id, status="promoted",
        )

        logger.info(
            "Promoted session %s to alias=%s serving_id=%s",
            session_id,
            result.alias,
            result.serving_id,
        )

        return {
            "serving_id": result.serving_id,
            "alias": result.alias,
            "spec_path": str(result.spec_path),
        }

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------

    async def get_decision_log(
        self,
        session: Any,
        *,
        session_id: uuid.UUID,
    ) -> list[dict[str, Any]]:
        """Return an ordered list of committed step payloads with timestamps.

        Args:
            session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.

        Returns:
            List of dicts with ``step_name``, ``payload``, ``committed_at``,
            and ``step_index`` for each committed step, ordered by step index.

        Raises:
            ValueError: If the session does not exist.
        """
        ms = await ModelingSessionRepository.get(
            session, session_id=session_id,
        )
        if ms is None:
            raise ValueError(f"ModelingSession not found: {session_id}")

        steps = await ModelingSessionRepository.get_steps(
            session, session_id=session_id,
        )

        log: list[dict[str, Any]] = []
        for step in steps:
            if step.status != "committed":
                continue
            log.append({
                "step_name": step.step_name,
                "step_index": _STEP_INDEX.get(step.step_name, -1),
                "payload": step.payload_json,
                "committed_at": (
                    step.committed_at.isoformat()
                    if step.committed_at
                    else None
                ),
            })

        log.sort(key=lambda entry: entry["step_index"])
        return log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_serving_spec(
        self,
        *,
        session_id: uuid.UUID,
        step_payloads: dict[str, dict[str, Any]],
    ) -> ServingSpec:
        """Assemble a ServingSpec from committed step payloads.

        Maps each step's payload to the corresponding ServingSpec fields.

        Args:
            session_id: UUID for naming purposes.
            step_payloads: Map of step_name -> committed payload dict.

        Returns:
            Assembled ServingSpec ready for validation and promotion.
        """
        dataset_payload = step_payloads["dataset_select"]
        gold_payload = step_payloads["gold_config"]
        signal_payload = step_payloads["signal_select"]
        eval_payload = step_payloads["eval_params"]

        # Build signal config
        signal_config: dict[str, Any] | None = None
        signal_name = signal_payload.get("signal_name", "derivative")
        if signal_name:
            signal_config = {
                "name": signal_name,
                "params": {
                    k: v for k, v in signal_payload.items()
                    if k not in ("signal_name",)
                },
            }

        # Build scoring config from eval_params
        scoring_data: dict[str, Any] = {}
        scoring_keys = {
            "zscore_window_bins", "zscore_min_periods",
            "derivative_weights", "tanh_scale", "neutral_threshold",
        }
        for key in scoring_keys:
            if key in eval_payload:
                scoring_data[key] = eval_payload[key]

        # Pipeline reference from dataset
        pipeline_name = dataset_payload.get(
            "pipeline", dataset_payload.get("dataset_id", "unknown"),
        )

        spec_name = f"modeling_{str(session_id).replace('-', '')[:8]}"

        spec_data: dict[str, Any] = {
            "name": spec_name,
            "description": f"Assembled from modeling session {session_id}",
            "pipeline": pipeline_name,
        }

        if scoring_data:
            spec_data["scoring"] = scoring_data

        if signal_config:
            spec_data["signal"] = signal_config

        # Gold config coefficients: store in description and wire DSL hash
        if gold_payload:
            spec_data["description"] += (
                f" | gold_config={json.dumps(gold_payload, sort_keys=True)}"
            )
            from ..gold_dsl.compat import gold_config_to_dsl
            from .gold_config import GoldFeatureConfig
            try:
                gold_cfg_data = dict(gold_payload)
                gold_cfg_data.setdefault("flow_windows", [10])
                gold_cfg_data.setdefault("flow_rollup_weights", [1.0])
                dsl_spec = gold_config_to_dsl(GoldFeatureConfig.model_validate(gold_cfg_data))
                full_hash = dsl_spec.spec_hash()
                spec_data["gold_dsl_spec_id"] = full_hash[:16]
                spec_data["gold_dsl_hash"] = full_hash
            except Exception as exc:
                logger.warning("Gold DSL wiring failed (non-fatal): %s", exc)

        return ServingSpec.model_validate(spec_data)

    # ------------------------------------------------------------------
    # Experiment synthesis + submission
    # ------------------------------------------------------------------

    async def synthesize_and_submit_experiment(
        self,
        db_session: Any,
        *,
        session_id: uuid.UUID,
        workspace_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Synthesize serving + experiment YAML specs and submit to the job queue.

        All four prerequisite steps (dataset_select, gold_config, signal_select,
        eval_params) must be committed before calling this method.

        Args:
            db_session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.
            workspace_id: Owning workspace UUID.

        Returns:
            Dict with ``job_id``, ``spec_ref``, and ``spec_name``.

        Raises:
            ValueError: If required steps are not committed.
        """
        _ms, steps = await self.get_session_with_steps(db_session, session_id=session_id)
        committed_map: dict[str, dict[str, Any]] = {
            s.step_name: s.payload_json or {}
            for s in steps
            if s.status == "committed"
        }

        required = ["dataset_select", "gold_config", "signal_select", "eval_params"]
        missing = [r for r in required if r not in committed_map]
        if missing:
            raise ValueError(
                f"Cannot synthesize experiment: steps not committed: {missing}"
            )

        spec_name = f"wizard_{str(session_id).replace('-', '')[:12]}"
        dataset_id: str = committed_map["dataset_select"].get("dataset_id", "unknown")
        eval_payload = committed_map["eval_params"]

        # Write serving spec YAML
        serving_spec = self._assemble_serving_spec(
            session_id=session_id,
            step_payloads=committed_map,
        )
        serving_configs_dir = (
            self._lake_root / "research" / "harness" / "configs" / "serving"
        )
        serving_configs_dir.mkdir(parents=True, exist_ok=True)
        serving_yaml_path = serving_configs_dir / f"{spec_name}.yaml"
        serving_yaml_path.write_text(
            yaml.dump(serving_spec.model_dump(exclude_none=True), default_flow_style=False)
        )

        # Build experiment spec dict
        experiment_dict: dict[str, Any] = {
            "name": spec_name,
            "description": f"Wizard experiment for session {session_id}",
            "serving": spec_name,
            "inline_datasets": [dataset_id],
            "eval": {
                "tp_ticks": eval_payload.get("tp_ticks", 8),
                "sl_ticks": eval_payload.get("sl_ticks", 4),
                "cooldown_bins": eval_payload.get("cooldown_bins", 20),
                "warmup_bins": eval_payload.get("warmup_bins", 300),
            },
            "tracking": {"backend": "none"},
            "parallel": {"max_workers": 1},
        }

        from .experiment_config import ExperimentSpec

        experiment_configs_dir = ExperimentSpec.configs_dir(self._lake_root)
        experiment_configs_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = f"{spec_name}.yaml"
        spec_path = experiment_configs_dir / spec_filename
        spec_path.write_text(yaml.dump(experiment_dict, default_flow_style=False))

        # Validate round-trip
        ExperimentSpec.from_yaml(spec_path)

        from .db.repositories import ExperimentJobRepository
        from .db.models import Workspace

        # Ensure workspace exists (create if missing)
        ws = await db_session.get(Workspace, workspace_id)
        if ws is None:
            ws = Workspace(workspace_id=workspace_id, name=str(workspace_id))
            db_session.add(ws)
            await db_session.flush()

        job = await ExperimentJobRepository.create(
            db_session,
            workspace_id=workspace_id,
            spec_ref=spec_filename,
            created_by="modeling_studio",
        )
        await db_session.flush()
        job_id = str(job.job_id)

        from ..jobs.queue import get_job_queue

        queue = await get_job_queue()
        await queue.enqueue(job_id, {
            "spec_ref": spec_filename,
            "workspace_id": str(workspace_id),
        })

        logger.info(
            "Submitted wizard experiment job %s (spec=%s) for session %s",
            job_id,
            spec_filename,
            session_id,
        )
        return {"job_id": job_id, "spec_ref": spec_filename, "spec_name": spec_name}

    async def synthesize_and_submit_sensitivity(
        self,
        db_session: Any,
        *,
        session_id: uuid.UUID,
        workspace_id: uuid.UUID,
        sweep_axis: str,
        sweep_values: list[Any],
    ) -> dict[str, Any]:
        """Synthesize and submit a parameter sensitivity sweep experiment.

        Works like ``synthesize_and_submit_experiment`` but adds a sweep
        block to explore one axis. Supported axes: tp_ticks, sl_ticks,
        cooldown_bins (mapped to eval config) or zscore_window_bins,
        tanh_scale, neutral_threshold (mapped to sweep.scoring).

        Args:
            db_session: Active ``AsyncSession``.
            session_id: UUID of the modeling session.
            workspace_id: Owning workspace UUID.
            sweep_axis: Parameter name to sweep.
            sweep_values: List of values to evaluate.

        Returns:
            Dict with ``job_id``, ``spec_ref``, and ``spec_name``.

        Raises:
            ValueError: If required steps are not committed.
        """
        _ms, steps = await self.get_session_with_steps(db_session, session_id=session_id)
        committed_map: dict[str, dict[str, Any]] = {
            s.step_name: s.payload_json or {}
            for s in steps
            if s.status == "committed"
        }

        required = ["dataset_select", "gold_config", "signal_select", "eval_params"]
        missing = [r for r in required if r not in committed_map]
        if missing:
            raise ValueError(
                f"Cannot synthesize sensitivity: steps not committed: {missing}"
            )

        session_id_short = str(session_id).replace("-", "")[:12]
        spec_name = f"wizard_{session_id_short}_sens_{sweep_axis}"
        dataset_id: str = committed_map["dataset_select"].get("dataset_id", "unknown")
        eval_payload = committed_map["eval_params"]

        # Write serving spec YAML (reuse base assembly)
        serving_spec = self._assemble_serving_spec(
            session_id=session_id,
            step_payloads=committed_map,
        )
        serving_configs_dir = (
            self._lake_root / "research" / "harness" / "configs" / "serving"
        )
        serving_configs_dir.mkdir(parents=True, exist_ok=True)
        serving_yaml_path = serving_configs_dir / f"{spec_name}.yaml"
        serving_yaml_path.write_text(
            yaml.dump(serving_spec.model_dump(exclude_none=True), default_flow_style=False)
        )

        # Eval block
        eval_block: dict[str, Any] = {
            "tp_ticks": eval_payload.get("tp_ticks", 8),
            "sl_ticks": eval_payload.get("sl_ticks", 4),
            "cooldown_bins": eval_payload.get("cooldown_bins", 20),
            "warmup_bins": eval_payload.get("warmup_bins", 300),
        }

        # Route sweep_axis to eval or sweep.scoring
        eval_axes = {"tp_ticks", "sl_ticks", "cooldown_bins"}
        sweep_block: dict[str, Any] = {}
        if sweep_axis in eval_axes:
            eval_block[sweep_axis] = sweep_values
        else:
            sweep_block["scoring"] = {sweep_axis: sweep_values}

        experiment_dict: dict[str, Any] = {
            "name": spec_name,
            "description": f"Sensitivity sweep ({sweep_axis}) for session {session_id}",
            "serving": spec_name,
            "inline_datasets": [dataset_id],
            "eval": eval_block,
            "tracking": {"backend": "none"},
            "parallel": {"max_workers": 1},
        }
        if sweep_block:
            experiment_dict["sweep"] = sweep_block

        from .experiment_config import ExperimentSpec

        experiment_configs_dir = ExperimentSpec.configs_dir(self._lake_root)
        experiment_configs_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = f"{spec_name}.yaml"
        spec_path = experiment_configs_dir / spec_filename
        spec_path.write_text(yaml.dump(experiment_dict, default_flow_style=False))

        # Validate round-trip
        ExperimentSpec.from_yaml(spec_path)

        from .db.repositories import ExperimentJobRepository
        from .db.models import Workspace

        ws = await db_session.get(Workspace, workspace_id)
        if ws is None:
            ws = Workspace(workspace_id=workspace_id, name=str(workspace_id))
            db_session.add(ws)
            await db_session.flush()

        job = await ExperimentJobRepository.create(
            db_session,
            workspace_id=workspace_id,
            spec_ref=spec_filename,
            created_by="modeling_studio",
        )
        await db_session.flush()
        job_id = str(job.job_id)

        from ..jobs.queue import get_job_queue

        queue = await get_job_queue()
        await queue.enqueue(job_id, {
            "spec_ref": spec_filename,
            "workspace_id": str(workspace_id),
        })

        logger.info(
            "Submitted sensitivity sweep job %s (axis=%s, spec=%s) for session %s",
            job_id,
            sweep_axis,
            spec_filename,
            session_id,
        )
        return {"job_id": job_id, "spec_ref": spec_filename, "spec_name": spec_name}
