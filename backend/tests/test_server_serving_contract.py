from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.qmachina.app import create_app
from src.qmachina.serving_config import (
    PublishedServingSource,
    PublishedServingSpec,
)
from src.qmachina.serving_registry import ServingRegistry


def _promote_minimal_spec(
    lake_root: Path,
    *,
    alias: str,
    runtime_snapshot: dict[str, object],
) -> None:
    registry = ServingRegistry(lake_root)
    serving_id = registry.build_serving_id(
        experiment_name="exp",
        run_id="run1234567890",
        config_hash="cfg1234567890",
    )
    spec = PublishedServingSpec(
        serving_id=serving_id,
        description="test",
        runtime_snapshot=runtime_snapshot,
        source=PublishedServingSource(
            run_id="run1234567890",
            experiment_name="exp",
            config_hash="cfg1234567890",
            promoted_at_utc="2026-02-20T00:00:00+00:00",
            serving_spec_name="serving_test",
            signal_name="derivative",
        ),
    )
    registry.promote(alias=alias, spec=spec, actor="test")


def test_stream_rejects_non_serving_query_params(tmp_path: Path) -> None:
    app = create_app(lake_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        "/v1/stream?serving=vp_main&product_type=future_mbo"
    ) as ws:
        payload = json.loads(ws.receive_text())
        assert payload["type"] == "error"
        assert "only 'serving' is allowed" in payload["message"]


def test_stream_rejects_unknown_serving_selector(tmp_path: Path) -> None:
    app = create_app(lake_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect("/v1/stream?serving=does_not_exist") as ws:
        payload = json.loads(ws.receive_text())
        assert payload["type"] == "error"
        assert "Unknown serving selector" in payload["message"]


def test_stream_rejects_published_serving_with_missing_runtime_identity(
    tmp_path: Path,
) -> None:
    _promote_minimal_spec(
        tmp_path,
        alias="vp_missing_identity",
        runtime_snapshot={"symbol": "MNQH6"},
    )

    app = create_app(lake_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        "/v1/stream?serving=vp_missing_identity"
    ) as ws:
        payload = json.loads(ws.receive_text())
        assert payload["type"] == "error"
        assert "missing required key: product_type" in payload["message"]


def test_stream_rejects_published_serving_with_missing_stream_context(
    tmp_path: Path,
) -> None:
    _promote_minimal_spec(
        tmp_path,
        alias="vp_missing_stream_dt",
        runtime_snapshot={
            "product_type": "future_mbo",
            "symbol": "MNQH6",
        },
    )

    app = create_app(lake_root=tmp_path)
    client = TestClient(app)

    with client.websocket_connect(
        "/v1/stream?serving=vp_missing_stream_dt"
    ) as ws:
        payload = json.loads(ws.receive_text())
        assert payload["type"] == "error"
        assert "missing required key: stream_dt" in payload["message"]
