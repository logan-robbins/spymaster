"""REST routes for the Model Studio control plane.

Endpoints:
    POST   /v1/modeling/sessions                                           -- Create session
    GET    /v1/modeling/sessions/{session_id}                              -- Get session + steps
    POST   /v1/modeling/sessions/{session_id}/steps/{step_name}/commit    -- Commit step
    POST   /v1/modeling/sessions/{session_id}/promote                      -- Promote
    GET    /v1/modeling/sessions/{session_id}/decisions                    -- Decision log
    POST   /v1/modeling/validate_yaml                                      -- Validate YAML
    POST   /v1/modeling/sessions/{session_id}/preview                      -- Dataset preview
    GET    /v1/modeling/specs                                              -- Available signals + datasets
    POST   /v1/modeling/sessions/{session_id}/experiment/submit            -- Submit wizard experiment
    GET    /v1/modeling/sessions/{session_id}/experiment/status            -- Experiment job status
    POST   /v1/modeling/sessions/{session_id}/sensitivity/submit           -- Submit sensitivity sweep
    GET    /v1/modeling/sessions/{session_id}/sensitivity/results          -- Sensitivity sweep results
"""
from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..experiment_harness.dataset_registry import DatasetRegistry
from .db.engine import get_db_session
from .modeling_session_service import STEPS_ORDERED, ModelingSessionService

logger: logging.Logger = logging.getLogger(__name__)

router: APIRouter = APIRouter(tags=["modeling"])


# ------------------------------------------------------------------
# Request / response models
# ------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for creating a new modeling session."""

    workspace_id: str
    created_by: str = "user"


class CreateSessionResponse(BaseModel):
    """Response for session creation."""

    session_id: str
    status: str


class StepStateResponse(BaseModel):
    """Serialized representation of a single modeling step."""

    step_name: str
    status: str
    payload: dict[str, Any] | None = None
    committed_at: str | None = None


class SessionResponse(BaseModel):
    """Full session detail including step states."""

    session_id: str
    status: str
    created_by: str
    created_at: str
    steps: list[StepStateResponse]


class CommitStepRequest(BaseModel):
    """Request body for committing a modeling step."""

    payload: dict[str, Any]


class CommitStepResponse(BaseModel):
    """Response after committing a step."""

    step_name: str
    status: str
    committed_at: str | None = None


class PromoteRequest(BaseModel):
    """Request body for promotion."""

    alias: str


class PromoteResponse(BaseModel):
    """Response after successful promotion."""

    serving_id: str
    alias: str
    spec_path: str


class ValidateYamlRequest(BaseModel):
    """Request body for YAML validation."""

    yaml_content: str


class ValidateYamlResponse(BaseModel):
    """Response from YAML validation."""

    valid: bool
    errors: list[str] = Field(default_factory=list)


class DatasetPreviewResponse(BaseModel):
    """Response from dataset preview."""

    n_bins: int
    date_range: dict[str, str]
    signal_distribution: dict[str, float | None]
    mid_price_range: dict[str, float | None]


class SpecsResponse(BaseModel):
    """Available signals and datasets for the modeling UI."""

    signals: list[str]
    datasets: list[str]
    steps: list[str]


class SubmitExperimentResponse(BaseModel):
    """Response after submitting a wizard experiment job."""

    job_id: str
    spec_ref: str
    spec_name: str


class ExperimentStatusResponse(BaseModel):
    """Status of the wizard experiment job."""

    job_id: str | None = None
    status: str
    run_ids: list[str] = Field(default_factory=list)
    n_runs: int = 0
    error_message: str | None = None


class SensitivitySubmitRequest(BaseModel):
    """Request body for submitting a parameter sensitivity sweep."""

    sweep_axis: str
    sweep_values: list[float | int]
    workspace_id: str


class SensitivityResultsResponse(BaseModel):
    """Results from a completed sensitivity sweep."""

    sweep_axis: str
    results: list[dict[str, Any]]
    job_status: str


# ------------------------------------------------------------------
# Route factory
# ------------------------------------------------------------------


def create_modeling_router(lake_root: Path) -> APIRouter:
    """Build and return the modeling API router.

    All routes are bound to the provided ``lake_root`` for dataset
    resolution and serving registry operations.

    Args:
        lake_root: Root path of the data lake.

    Returns:
        Configured ``APIRouter`` with all modeling endpoints.
    """
    service = ModelingSessionService(lake_root)
    dataset_registry = DatasetRegistry(lake_root)

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions",
        response_model=CreateSessionResponse,
    )
    async def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        """Create a new draft modeling session."""
        try:
            ws_id = uuid.UUID(req.workspace_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workspace_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            ms = await service.create_session(
                db,
                workspace_id=ws_id,
                created_by=req.created_by,
            )
            return CreateSessionResponse(
                session_id=str(ms.session_id),
                status=ms.status,
            )

    # ------------------------------------------------------------------
    # GET /v1/modeling/sessions/{session_id}
    # ------------------------------------------------------------------

    @router.get(
        "/v1/modeling/sessions/{session_id}",
        response_model=SessionResponse,
    )
    async def get_session(session_id: str) -> SessionResponse:
        """Get session with all step states."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            try:
                ms, steps = await service.get_session_with_steps(
                    db, session_id=sid,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            step_responses: list[StepStateResponse] = []
            for s in steps:
                step_responses.append(
                    StepStateResponse(
                        step_name=s.step_name,
                        status=s.status,
                        payload=s.payload_json,
                        committed_at=(
                            s.committed_at.isoformat()
                            if s.committed_at
                            else None
                        ),
                    )
                )

            return SessionResponse(
                session_id=str(ms.session_id),
                status=ms.status,
                created_by=ms.created_by,
                created_at=ms.created_at.isoformat(),
                steps=step_responses,
            )

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions/{session_id}/steps/{step_name}/commit
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions/{session_id}/steps/{step_name}/commit",
        response_model=CommitStepResponse,
    )
    async def commit_step(
        session_id: str,
        step_name: str,
        req: CommitStepRequest,
    ) -> CommitStepResponse:
        """Commit a modeling step with gate enforcement."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            try:
                step = await service.commit_step(
                    db,
                    session_id=sid,
                    step_name=step_name,
                    payload=req.payload,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            return CommitStepResponse(
                step_name=step.step_name,
                status=step.status,
                committed_at=(
                    step.committed_at.isoformat()
                    if step.committed_at
                    else None
                ),
            )

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions/{session_id}/promote
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions/{session_id}/promote",
        response_model=PromoteResponse,
    )
    async def promote_session(
        session_id: str,
        req: PromoteRequest,
    ) -> PromoteResponse:
        """Promote a completed session to a serving alias."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            try:
                result = await service.promote(
                    db,
                    session_id=sid,
                    alias=req.alias,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            return PromoteResponse(
                serving_id=result["serving_id"],
                alias=result["alias"],
                spec_path=result["spec_path"],
            )

    # ------------------------------------------------------------------
    # GET /v1/modeling/sessions/{session_id}/decisions
    # ------------------------------------------------------------------

    @router.get("/v1/modeling/sessions/{session_id}/decisions")
    async def get_decisions(session_id: str) -> list[dict[str, Any]]:
        """Return the decision log for a modeling session."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            try:
                return await service.get_decision_log(
                    db, session_id=sid,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=404, detail=str(exc),
                ) from exc

    # ------------------------------------------------------------------
    # POST /v1/modeling/validate_yaml
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/validate_yaml",
        response_model=ValidateYamlResponse,
    )
    async def validate_yaml(req: ValidateYamlRequest) -> ValidateYamlResponse:
        """Validate a YAML string as a ServingSpec.

        Parses the YAML, writes to a temp file, and runs
        ``validate_serving_spec()`` against it.
        """
        errors: list[str] = []
        try:
            parsed = yaml.safe_load(req.yaml_content)
        except yaml.YAMLError as exc:
            return ValidateYamlResponse(
                valid=False,
                errors=[f"YAML parse error: {exc}"],
            )

        if not isinstance(parsed, dict):
            return ValidateYamlResponse(
                valid=False,
                errors=[
                    f"Expected YAML mapping at top level, "
                    f"got {type(parsed).__name__}"
                ],
            )

        try:
            from .serving_config import ServingSpec

            ServingSpec.model_validate(parsed)
        except Exception as exc:
            errors.append(str(exc))

        return ValidateYamlResponse(
            valid=len(errors) == 0,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions/{session_id}/preview
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions/{session_id}/preview",
        response_model=DatasetPreviewResponse,
    )
    async def dataset_preview(session_id: str) -> DatasetPreviewResponse:
        """Generate sample statistics from the selected dataset.

        Only available after the ``dataset_select`` step is committed.
        Reads ``bins.parquet`` and ``grid_clean.parquet`` for the
        selected dataset_id and computes summary statistics.
        """
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid session_id: {exc}",
            ) from exc

        async with get_db_session() as db:
            try:
                _ms, steps = await service.get_session_with_steps(
                    db, session_id=sid,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=404, detail=str(exc),
                ) from exc

            dataset_step = next(
                (
                    s for s in steps
                    if s.step_name == "dataset_select"
                    and s.status == "committed"
                ),
                None,
            )
            if dataset_step is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "dataset_select step must be committed "
                        "before requesting a preview"
                    ),
                )

            dataset_id: str = (dataset_step.payload_json or {}).get(
                "dataset_id", "",
            )
            if not dataset_id:
                raise HTTPException(
                    status_code=400,
                    detail="dataset_select payload missing 'dataset_id'",
                )

        # Read parquet files (outside DB session -- file I/O only)
        try:
            paths = dataset_registry.resolve(dataset_id)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404, detail=str(exc),
            ) from exc

        import pandas as pd

        bins_df = pd.read_parquet(paths.bins_parquet)
        grid_df = pd.read_parquet(paths.grid_clean_parquet)

        n_bins: int = len(bins_df)

        # Date range from bins
        date_range: dict[str, str] = {}
        if "ts_ns" in bins_df.columns and n_bins > 0:
            ts_min = pd.Timestamp(bins_df["ts_ns"].min(), unit="ns")
            ts_max = pd.Timestamp(bins_df["ts_ns"].max(), unit="ns")
            date_range = {
                "start": ts_min.isoformat(),
                "end": ts_max.isoformat(),
            }
        elif "dt" in bins_df.columns and n_bins > 0:
            date_range = {
                "start": str(bins_df["dt"].min()),
                "end": str(bins_df["dt"].max()),
            }

        # Mid price range
        mid_price_range: dict[str, float | None] = {
            "min": None, "max": None,
        }
        if "mid_price" in bins_df.columns and n_bins > 0:
            mid_price_range["min"] = float(bins_df["mid_price"].min())
            mid_price_range["max"] = float(bins_df["mid_price"].max())

        # Signal distribution (if flow_score or composite exists in grid)
        signal_distribution: dict[str, float | None] = {
            "mean": None,
            "std": None,
            "pct25": None,
            "pct50": None,
            "pct75": None,
        }
        signal_col: str | None = None
        for candidate in ("flow_score", "composite", "rest_depth"):
            if candidate in grid_df.columns:
                signal_col = candidate
                break

        if signal_col is not None and len(grid_df) > 0:
            col = grid_df[signal_col].dropna()
            if len(col) > 0:
                signal_distribution["mean"] = float(col.mean())
                signal_distribution["std"] = float(col.std())
                quantiles = col.quantile([0.25, 0.50, 0.75])
                signal_distribution["pct25"] = float(quantiles[0.25])
                signal_distribution["pct50"] = float(quantiles[0.50])
                signal_distribution["pct75"] = float(quantiles[0.75])

        return DatasetPreviewResponse(
            n_bins=n_bins,
            date_range=date_range,
            signal_distribution=signal_distribution,
            mid_price_range=mid_price_range,
        )

    # ------------------------------------------------------------------
    # GET /v1/modeling/specs
    # ------------------------------------------------------------------

    @router.get("/v1/modeling/specs", response_model=SpecsResponse)
    async def get_specs() -> SpecsResponse:
        """List available signals and datasets for the modeling UI."""
        # List datasets
        datasets: list[str] = dataset_registry.list_datasets()

        # List registered signals
        signals: list[str] = []
        try:
            from ..experiment_harness.signals import (
                SIGNAL_REGISTRY,
                ensure_signals_loaded,
            )

            ensure_signals_loaded()
            signals = sorted(SIGNAL_REGISTRY.keys())
        except ImportError:
            logger.warning("Could not import signal registry")

        return SpecsResponse(
            signals=signals,
            datasets=datasets,
            steps=STEPS_ORDERED,
        )

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions/{session_id}/experiment/submit
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions/{session_id}/experiment/submit",
        response_model=SubmitExperimentResponse,
    )
    async def submit_experiment(session_id: str) -> SubmitExperimentResponse:
        """Synthesize serving + experiment specs and submit a job."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid session_id: {exc}") from exc

        # Use a fixed workspace for the studio (single-user mode)
        workspace_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

        async with get_db_session() as db:
            try:
                result = await service.synthesize_and_submit_experiment(
                    db,
                    session_id=sid,
                    workspace_id=workspace_id,
                )
                await db.commit()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SubmitExperimentResponse(
            job_id=result["job_id"],
            spec_ref=result["spec_ref"],
            spec_name=result["spec_name"],
        )

    # ------------------------------------------------------------------
    # GET /v1/modeling/sessions/{session_id}/experiment/status
    # ------------------------------------------------------------------

    @router.get(
        "/v1/modeling/sessions/{session_id}/experiment/status",
        response_model=ExperimentStatusResponse,
    )
    async def get_experiment_status(session_id: str) -> ExperimentStatusResponse:
        """Return status of the wizard experiment job for this session."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid session_id: {exc}") from exc

        async with get_db_session() as db:
            try:
                _ms, steps = await service.get_session_with_steps(db, session_id=sid)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            run_exp_step = next(
                (s for s in steps if s.step_name == "run_experiment" and s.status == "committed"),
                None,
            )
            if run_exp_step is None:
                return ExperimentStatusResponse(status="not_submitted")

            payload = run_exp_step.payload_json or {}
            job_id_str = payload.get("job_id")
            if not job_id_str:
                return ExperimentStatusResponse(status="not_submitted")

            from .db.repositories import ExperimentJobRepository

            try:
                job_uuid = uuid.UUID(job_id_str)
                job = await ExperimentJobRepository.get(db, job_id=job_uuid)
            except (ValueError, Exception):
                job = None

            if job is None:
                return ExperimentStatusResponse(
                    job_id=job_id_str,
                    status=payload.get("status", "unknown"),
                    run_ids=payload.get("run_ids", []),
                    n_runs=payload.get("n_runs", 0),
                )

            return ExperimentStatusResponse(
                job_id=str(job.job_id),
                status=job.status,
                run_ids=payload.get("run_ids", []),
                n_runs=payload.get("n_runs", 0),
                error_message=job.error_message,
            )

    # ------------------------------------------------------------------
    # POST /v1/modeling/sessions/{session_id}/sensitivity/submit
    # ------------------------------------------------------------------

    @router.post(
        "/v1/modeling/sessions/{session_id}/sensitivity/submit",
        response_model=SubmitExperimentResponse,
    )
    async def submit_sensitivity(
        session_id: str,
        req: SensitivitySubmitRequest,
    ) -> SubmitExperimentResponse:
        """Submit a parameter sensitivity sweep for this session."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid session_id: {exc}") from exc

        try:
            workspace_id = uuid.UUID(req.workspace_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid workspace_id: {exc}") from exc

        async with get_db_session() as db:
            try:
                result = await service.synthesize_and_submit_sensitivity(
                    db,
                    session_id=sid,
                    workspace_id=workspace_id,
                    sweep_axis=req.sweep_axis,
                    sweep_values=list(req.sweep_values),
                )
                await db.commit()
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return SubmitExperimentResponse(
            job_id=result["job_id"],
            spec_ref=result["spec_ref"],
            spec_name=result["spec_name"],
        )

    # ------------------------------------------------------------------
    # GET /v1/modeling/sessions/{session_id}/sensitivity/results
    # ------------------------------------------------------------------

    @router.get(
        "/v1/modeling/sessions/{session_id}/sensitivity/results",
        response_model=SensitivityResultsResponse,
    )
    async def get_sensitivity_results(session_id: str) -> SensitivityResultsResponse:
        """Return sensitivity sweep results for this session."""
        try:
            sid = uuid.UUID(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid session_id: {exc}") from exc

        async with get_db_session() as db:
            try:
                _ms, steps = await service.get_session_with_steps(db, session_id=sid)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Find the most recent sensitivity step payload from promote_review
        promote_step = next(
            (s for s in steps if s.step_name == "promote_review" and s.status == "committed"),
            None,
        )
        sens_payload = (promote_step.payload_json or {}) if promote_step else {}
        sens_job_id = sens_payload.get("sensitivity_job_id")
        sweep_axis = sens_payload.get("sensitivity_sweep_axis", "")
        run_ids: list[str] = sens_payload.get("sensitivity_run_ids", [])
        job_status = sens_payload.get("sensitivity_job_status", "not_submitted")

        results: list[dict[str, Any]] = []
        if run_ids:
            try:
                from ..experiment_harness.results_db import ResultsDB

                results_db_root = (
                    lake_root / "research" / "harness" / "results"
                )
                db_inst = ResultsDB(results_db_root)
                import pandas as pd

                for run_id in run_ids:
                    meta = db_inst.query_runs(run_id=run_id)
                    if meta.empty:
                        continue
                    row = meta.iloc[0]
                    params = {}
                    try:
                        import json as _json
                        params = _json.loads(row.get("signal_params_json", "{}"))
                    except Exception:
                        pass
                    results.append({
                        "run_id": run_id,
                        "sweep_value": params.get(sweep_axis),
                        "tp_hit_rate": float(row.get("tp_rate", 0) or 0),
                        "sl_hit_rate": float(row.get("sl_rate", 0) or 0),
                        "n_signals": int(row.get("n_signals", 0) or 0),
                    })
            except Exception as exc:
                logger.warning("Failed to load sensitivity results: %s", exc)

        return SensitivityResultsResponse(
            sweep_axis=sweep_axis,
            results=results,
            job_status=job_status,
        )

    return router
