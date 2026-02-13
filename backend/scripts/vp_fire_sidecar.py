"""Vacuum-pressure FIRE outcome sidecar.

Connects to the vacuum-pressure WebSocket stream, detects transitions into
FIRE, and tracks whether price moves by N ticks in the FIRE direction within
the configured horizon.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pyarrow as pa
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from src.vacuum_pressure.fire_sidecar import FireOutcomeTracker


logger = logging.getLogger(__name__)

DEFAULT_WS_BASE_URL = "ws://localhost:8002/v1/vacuum-pressure/stream"
REQUIRED_SIGNALS_FIELDS = ("window_end_ts_ns", "event_state", "event_direction")
REQUIRED_SNAP_FIELDS = ("window_end_ts_ns", "mid_price")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field!r} is not int-coercible: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Field {field!r} must be > 0, got {parsed}")
    return parsed


def _as_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field {field!r} is not float-coercible: {value!r}") from exc
    if not (parsed > 0):
        raise ValueError(f"Field {field!r} must be > 0, got {parsed}")
    return parsed


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Field {field!r} must be a string, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Field {field!r} must be non-empty")
    return cleaned


def _required(row: dict[str, Any], field: str, surface: str) -> Any:
    if field not in row or row[field] is None:
        raise ValueError(
            f"Missing required field {field!r} on {surface} surface."
        )
    return row[field]


def _decode_arrow_first_row(payload: bytes, surface: str) -> dict[str, Any]:
    try:
        table = pa.ipc.open_stream(payload).read_all()
    except Exception as exc:  # pragma: no cover - pyarrow error shape is not stable.
        raise ValueError(f"Failed to decode Arrow IPC for {surface}: {exc}") from exc

    if table.num_rows < 1:
        raise ValueError(f"Expected at least one row on {surface}, received zero.")

    row = table.to_pylist()[0]
    if not isinstance(row, dict):
        raise ValueError(f"Decoded {surface} row is not a dict: {type(row).__name__}")
    return row


def _build_stream_url(args: argparse.Namespace) -> str:
    params: dict[str, str] = {
        "product_type": args.product_type,
        "symbol": args.symbol,
        "dt": args.dt,
        "speed": str(args.speed),
        "mode": args.mode,
    }
    if args.start_time:
        params["start_time"] = args.start_time
    return f"{args.ws_url}?{urlencode(params)}"


class JsonlLogger:
    """Simple structured JSONL writer with explicit flush behavior."""

    def __init__(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = output_path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


async def _recv_or_stop(
    websocket: ClientConnection,
    stop_event: asyncio.Event,
) -> str | bytes | None:
    recv_task = asyncio.create_task(websocket.recv())
    stop_task = asyncio.create_task(stop_event.wait())
    done, pending = await asyncio.wait(
        {recv_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()

    if stop_task in done and stop_event.is_set():
        recv_task.cancel()
        return None

    return recv_task.result()


async def _periodic_metrics_task(
    *,
    stop_event: asyncio.Event,
    tracker: FireOutcomeTracker,
    writer: JsonlLogger,
    print_interval_s: float,
    session: dict[str, Any],
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=print_interval_s)
            return
        except TimeoutError:
            metrics = tracker.metrics()
            record = {
                "ts": _utc_now_iso(),
                "type": "metrics",
                "reason": "interval",
                "session": session,
                "metrics": metrics,
            }
            writer.write(record)
            logger.info("Metrics: %s", json.dumps(metrics, sort_keys=True))


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _set_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _set_stop)
        except NotImplementedError:  # pragma: no cover
            signal.signal(sig, lambda _s, _f: stop_event.set())


async def run_sidecar(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    writer = JsonlLogger(output_path)
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    session = {
        "symbol": args.symbol,
        "dt": args.dt,
        "product_type": args.product_type,
        "speed": args.speed,
        "mode": args.mode,
        "start_time": args.start_time,
        "tick_target": args.tick_target,
        "max_horizon_s": args.max_horizon_s,
    }
    stream_url = _build_stream_url(args)

    pending_surface: str | None = None
    snap_mid_by_window: dict[int, float] = {}
    tracker: FireOutcomeTracker | None = None
    periodic_task: asyncio.Task[None] | None = None

    writer.write({
        "ts": _utc_now_iso(),
        "type": "startup",
        "stream_url": stream_url,
        "session": session,
    })

    logger.info("Connecting to %s", stream_url)

    exit_code = 0
    error_message: str | None = None

    try:
        async with websockets.connect(stream_url, max_size=None) as websocket:
            while not stop_event.is_set():
                message = await _recv_or_stop(websocket, stop_event)
                if message is None:
                    break

                if isinstance(message, str):
                    payload = json.loads(message)
                    msg_type = payload.get("type")

                    if msg_type == "runtime_config":
                        tick_size = _as_float(
                            _required(payload, "tick_size", "runtime_config"),
                            "tick_size",
                        )
                        tracker = FireOutcomeTracker(
                            tick_size=tick_size,
                            tick_target=float(args.tick_target),
                            max_horizon_s=float(args.max_horizon_s),
                        )
                        if periodic_task is None:
                            periodic_task = asyncio.create_task(
                                _periodic_metrics_task(
                                    stop_event=stop_event,
                                    tracker=tracker,
                                    writer=writer,
                                    print_interval_s=float(args.print_interval_s),
                                    session=session,
                                )
                            )
                    elif msg_type == "surface_header":
                        pending_surface = str(payload.get("surface", ""))
                    elif msg_type == "error":
                        raise RuntimeError(
                            f"Server returned error frame: {payload.get('message')}"
                        )

                    continue

                if tracker is None:
                    raise RuntimeError("Received binary data before runtime_config frame.")
                if not pending_surface:
                    raise RuntimeError("Received binary frame without pending surface_header.")

                surface = pending_surface
                pending_surface = None

                if surface == "snap":
                    row = _decode_arrow_first_row(message, surface)
                    for field in REQUIRED_SNAP_FIELDS:
                        _required(row, field, "snap")
                    wid = _as_int(row["window_end_ts_ns"], "window_end_ts_ns")
                    mid_price = _as_float(row["mid_price"], "mid_price")
                    snap_mid_by_window[wid] = mid_price
                elif surface == "signals":
                    row = _decode_arrow_first_row(message, surface)
                    for field in REQUIRED_SIGNALS_FIELDS:
                        _required(row, field, "signals")
                    wid = _as_int(row["window_end_ts_ns"], "window_end_ts_ns")
                    event_state = _as_str(row["event_state"], "event_state")
                    event_direction = _as_str(row["event_direction"], "event_direction")
                    if wid not in snap_mid_by_window:
                        raise ValueError(
                            f"Missing required field 'mid_price' for window {wid} "
                            "when processing signals."
                        )
                    mid_price = snap_mid_by_window.pop(wid)

                    fire_event, outcomes = tracker.update(
                        window_end_ts_ns=wid,
                        mid_price=mid_price,
                        event_state=event_state,
                        event_direction=event_direction,
                    )
                    if fire_event is not None:
                        writer.write({
                            "ts": _utc_now_iso(),
                            "type": "fire_entry",
                            "session": session,
                            "event": {
                                "event_id": fire_event.event_id,
                                "fire_ts_ns": fire_event.fire_ts_ns,
                                "fire_price": fire_event.fire_price,
                                "direction": fire_event.direction,
                                "tick_target": fire_event.tick_target,
                                "tick_size": fire_event.tick_size,
                                "target_price": fire_event.target_price,
                                "deadline_ts_ns": fire_event.deadline_ts_ns,
                            },
                            "metrics": tracker.metrics(),
                        })
                    for outcome in outcomes:
                        writer.write({
                            "ts": _utc_now_iso(),
                            "type": "outcome",
                            "session": session,
                            "outcome": {
                                "event_id": outcome.event_id,
                                "status": outcome.status,
                                "fire_ts_ns": outcome.fire_ts_ns,
                                "resolved_ts_ns": outcome.resolved_ts_ns,
                                "fire_price": outcome.fire_price,
                                "resolved_price": outcome.resolved_price,
                                "direction": outcome.direction,
                                "target_price": outcome.target_price,
                                "time_to_outcome_s": outcome.time_to_outcome_s,
                            },
                            "metrics": tracker.metrics(),
                        })
                else:
                    # flow and unknown surfaces are intentionally ignored.
                    continue

    except ConnectionClosed as exc:
        exit_code = 1
        error_message = f"Connection closed: code={exc.code}, reason={exc.reason!r}"
        logger.error(error_message)
    except Exception as exc:
        exit_code = 1
        error_message = str(exc)
        logger.exception("Sidecar failed")
    finally:
        if periodic_task is not None:
            stop_event.set()
            periodic_task.cancel()
            await asyncio.gather(periodic_task, return_exceptions=True)

        final_metrics = tracker.metrics() if tracker is not None else None
        writer.write({
            "ts": _utc_now_iso(),
            "type": "shutdown",
            "session": session,
            "metrics": final_metrics,
            "error": error_message,
        })
        writer.close()

    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Background sidecar for FIRE-direction +tick-target horizon accuracy."
        )
    )
    parser.add_argument("--symbol", default="QQQ", help="Instrument symbol.")
    parser.add_argument("--dt", default="2026-02-06", help="Session date YYYY-MM-DD.")
    parser.add_argument(
        "--product-type",
        default="equity_mbo",
        choices=["equity_mbo", "future_mbo"],
        help="Product type.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier.",
    )
    parser.add_argument(
        "--mode",
        default="replay",
        choices=["replay", "live"],
        help="Streaming mode.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional emit start time HH:MM ET.",
    )
    parser.add_argument(
        "--tick-target",
        type=float,
        default=8.0,
        help="Directional target in ticks from FIRE entry (default: 8).",
    )
    parser.add_argument(
        "--max-horizon-s",
        type=float,
        default=10.0,
        help="Maximum horizon in seconds to resolve hit/miss.",
    )
    parser.add_argument(
        "--output",
        default=str(backend_root / "logs" / "vp_fire_sidecar.jsonl"),
        help="JSONL output path.",
    )
    parser.add_argument(
        "--print-interval-s",
        type=float,
        default=15.0,
        help="Emit periodic metrics every N seconds.",
    )
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_BASE_URL,
        help="Base websocket URL for vacuum-pressure stream.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    exit_code = asyncio.run(run_sidecar(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
