"""Lightweight web demo server for the custom chat_ui frontend."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from email.parser import BytesParser
from email.policy import default as email_policy_default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

from agent.config import get_config
from agent.data.apple_watch_loader import AppleWatchUploadError
from agent.data.record_repository import get_record_repository
from agent.demo.data_sources import Icentia11kDataSource
from agent.demo.live_session import LiveMonitorSession
from agent.reactive.pipeline import ReactivePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MAX_JSON_BODY_BYTES = 1_000_000
MAX_MULTIPART_BODY_BYTES = 5_000_000

pipeline: ReactivePipeline | None = None


class LiveSessionError(RuntimeError):
    """Base error for live web session operations."""


class NoLiveSessionError(LiveSessionError):
    """Raised when an endpoint requires an active live session."""


class LiveSessionConflictError(LiveSessionError):
    """Raised when a conflicting start or chat operation is running."""


def _sanitize_json(value: Any) -> Any:
    """Recursively convert payloads to strict browser-safe JSON values."""
    if hasattr(value, "item") and callable(value.item):
        return _sanitize_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _snapshot_payload(session: Any) -> dict[str, Any]:
    snapshot = session.snapshot()
    return asdict(snapshot) if hasattr(snapshot, "__dataclass_fields__") else dict(snapshot)


class LiveWebRuntime:
    """Concurrency-safe owner for the demo's one global live session."""

    def __init__(self, *, discrete_maxsize: int = 256) -> None:
        self._session: LiveMonitorSession | None = None
        self._generation = 0
        self._lifecycle_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._chat_lock = threading.Lock()
        self._latest_window: dict[str, Any] | None = None
        self._latest_dirty = False
        self._discrete_maxsize = discrete_maxsize
        self._discrete: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=discrete_maxsize
        )
        self._subscriber_seq = 0
        self._chat_history: list[dict[str, str]] = []

    def start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        patient = str(payload.get("patient") or "01182").strip()
        raw_segments = payload.get("segments", ["00", "01"])
        if not patient:
            raise ValueError("patient is required")
        if not isinstance(raw_segments, list) or not raw_segments or len(raw_segments) > 10:
            raise ValueError("segments must be a non-empty list with at most 10 items")
        segments = [(str(item).strip(), 0.0, None) for item in raw_segments]
        if any(not segment_id for segment_id, _, _ in segments):
            raise ValueError("segment IDs must be non-empty")
        speed = float(payload.get("speed", 30.0))
        window_seconds = float(payload.get("window_seconds", 10.0))
        if speed <= 0 or window_seconds <= 0:
            raise ValueError("speed and window_seconds must be positive")

        with self._start_lock:
            if not self._chat_lock.acquire(blocking=False):
                raise LiveSessionConflictError(
                    "Finish the current question before starting a new session."
                )
            try:
                self._chat_history = []
                with self._lifecycle_lock:
                    old_session = self._session
                    self._generation += 1
                    generation = self._generation
                    self._session = None
                    self._latest_window = None
                    self._latest_dirty = False
                    self._discrete = queue.Queue(maxsize=self._discrete_maxsize)
                    self._subscriber_seq += 1

                if old_session is not None:
                    old_session.stop()
                    old_session.wait(timeout=0.5)

                local_root = str(get_config().datasets.icentia11k_root)
                if not os.path.isdir(local_root):
                    raise ValueError(f"Local Icentia11k root does not exist: {local_root}")
                source = Icentia11kDataSource(
                    patient_id=patient,
                    segments=segments,
                    window_seconds=window_seconds,
                    local_root=local_root,
                )

                def on_window(record: dict[str, Any]) -> None:
                    event = self._window_event(record)
                    with self._lifecycle_lock:
                        if generation != self._generation:
                            return
                        self._latest_window = event
                        self._latest_dirty = True

                def on_alert(event: dict[str, Any]) -> None:
                    self._put_discrete(
                        generation,
                        {"type": "alert", "alert": event},
                    )

                new_session = LiveMonitorSession(
                    source,
                    speed=speed,
                    emit_signal_preview=True,
                    on_window=on_window,
                    on_alert=on_alert,
                )
                new_session.start()
                with self._lifecycle_lock:
                    if generation != self._generation:
                        new_session.stop()
                        raise LiveSessionConflictError("Session start was superseded.")
                    self._session = new_session
                return _snapshot_payload(new_session)
            finally:
                self._chat_lock.release()

    @staticmethod
    def _window_event(record: dict[str, Any]) -> dict[str, Any]:
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        vitals = result.get("vitals") if isinstance(result.get("vitals"), dict) else {}
        return {
            "type": "window",
            "offset_s": record.get("offset_s"),
            "segment_id": record.get("segment_id"),
            "window_index": record.get("window_index"),
            "hr_bpm": vitals.get("hr_bpm"),
            "rhythm_class": result.get("rhythm_class"),
            "intervene": record.get("intervene", False),
            "signal_preview": record.get("signal_preview", []),
            "fs_preview": record.get("fs_preview"),
        }

    def _put_discrete(self, generation: int, event: dict[str, Any]) -> None:
        with self._lifecycle_lock:
            if generation != self._generation:
                return
            target_queue = self._discrete
        try:
            target_queue.put_nowait(event)
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                target_queue.put_nowait(event)
            except queue.Full:
                logger.warning("Dropping live alert event because the queue is full")

    def snapshot(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            session = self._session
        if session is None:
            raise NoLiveSessionError("No live monitoring session is active.")
        return _snapshot_payload(session)

    def control(self, action: str, *, speed: float | None = None) -> dict[str, Any]:
        with self._lifecycle_lock:
            session = self._session
        if session is None:
            raise NoLiveSessionError("No live monitoring session is active.")
        if action == "pause":
            session.pause()
        elif action == "resume":
            session.resume()
        elif action == "next":
            session.fast_forward_to_next_alert()
        elif action == "stop":
            session.stop()
        elif action == "speed":
            if speed is None or float(speed) <= 0:
                raise ValueError("speed must be positive")
            session.set_speed(float(speed))
        else:
            raise ValueError(f"Unknown session action: {action}")
        return _snapshot_payload(session)

    def acquire_chat_session(self) -> LiveMonitorSession:
        if not self._chat_lock.acquire(blocking=False):
            raise LiveSessionConflictError("Another live question is still running.")
        with self._lifecycle_lock:
            session = self._session
        if session is None:
            self._chat_lock.release()
            raise NoLiveSessionError("No live monitoring session is active.")
        if session.snapshot().windows_processed < 1:
            self._chat_lock.release()
            raise LiveSessionConflictError("Wait for the first monitoring window.")
        return session

    def release_chat_session(self) -> None:
        if self._chat_lock.locked():
            self._chat_lock.release()

    def chat_history_for_prompt(self) -> list[dict[str, str]]:
        """Return recent turns while the caller holds the chat lock."""
        return [dict(turn) for turn in self._chat_history[-4:]]

    def record_chat_turn(self, question: str, answer: str) -> None:
        """Store a completed turn while the caller holds the chat lock."""
        self._chat_history.append({"question": question, "answer": answer})
        del self._chat_history[:-12]

    def reset_chat_history(self) -> None:
        if not self._chat_lock.acquire(blocking=False):
            raise LiveSessionConflictError("Finish the current question before clearing chat.")
        try:
            self._chat_history = []
        finally:
            self._chat_lock.release()

    def iter_events(
        self,
        *,
        tick_s: float = 0.25,
        status_interval_s: float = 1.0,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> Iterator[dict[str, Any]]:
        with self._lifecycle_lock:
            session = self._session
            if session is None:
                raise NoLiveSessionError("No live monitoring session is active.")
            self._subscriber_seq += 1
            subscriber_seq = self._subscriber_seq
            generation = self._generation
            discrete = self._discrete
            initial_window = self._latest_window
            self._latest_dirty = False

        latest_sent = initial_window is not None
        if initial_window is not None:
            yield initial_window
        next_status_at = clock()

        while True:
            with self._lifecycle_lock:
                if (
                    generation != self._generation
                    or subscriber_seq != self._subscriber_seq
                ):
                    return
                latest = self._latest_window if self._latest_dirty else None
                if latest is not None:
                    self._latest_dirty = False

            emitted = False
            if latest is not None:
                latest_sent = True
                emitted = True
                yield latest

            while True:
                try:
                    event = discrete.get_nowait()
                except queue.Empty:
                    break
                emitted = True
                yield event

            now = clock()
            snapshot = _snapshot_payload(session)
            if now >= next_status_at:
                emitted = True
                yield {"type": "status", "snapshot": snapshot}
                next_status_at = now + status_interval_s

            if snapshot.get("completed"):
                with self._lifecycle_lock:
                    no_pending_window = not self._latest_dirty
                no_window_expected = int(snapshot.get("windows_processed") or 0) == 0
                if (
                    (latest_sent or no_window_expected)
                    and no_pending_window
                    and discrete.empty()
                ):
                    yield {"type": "done", "snapshot": snapshot}
                    return

            if not emitted:
                sleeper(tick_s)


live_runtime = LiveWebRuntime()


class RequestBodyError(ValueError):
    """Raised when the incoming HTTP request body is invalid."""


class RequestTooLargeError(RequestBodyError):
    """Raised when the incoming HTTP request exceeds configured size limits."""


@dataclass
class ChatResult:
    answer: str
    logs: list[str]
    risk_level: str | None
    retry_count: int


def _get_apple_watch_loader():
    return get_record_repository().apple_watch_loader


def init_pipeline() -> None:
    """Lazily initialise reactive pipeline."""
    global pipeline
    if pipeline is None:
        cfg = get_config()
        logger.info("Initialising Reactive Pipeline (model=%s)…", cfg.llm.model)
        pipeline = ReactivePipeline()


def _serialize_record_summary(record: dict[str, Any]) -> dict[str, Any]:
    label = record.get("label")
    if not label:
        source = record.get("source", "unknown")
        lead_names = record.get("lead_names") or []
        lead_text = ",".join(lead_names) if lead_names else "-"
        sampling_rate = record.get("sampling_rate")
        duration_seconds = record.get("duration_seconds")
        label = (
            f"{record.get('record_id')} | source={source} | lead={lead_text} | "
            f"sr={sampling_rate if sampling_rate is not None else '-'}Hz | "
            f"duration={duration_seconds if duration_seconds is not None else '-'}s"
        )

    return {
        "record_id": record.get("record_id"),
        "source": record.get("source", "ptbxl"),
        "patient_id": record.get("patient_id"),
        "age": record.get("age"),
        "sex": record.get("sex"),
        "recording_date": record.get("recording_date"),
        "diagnostic_class": record.get("diagnostic_class"),
        "lead_names": record.get("lead_names") or [],
        "num_leads": record.get("num_leads"),
        "sampling_rate": record.get("sampling_rate"),
        "duration_seconds": record.get("duration_seconds"),
        "label": label,
    }


def list_records(limit: int = 50) -> list[dict[str, Any]]:
    records = get_record_repository().list_records(limit=limit)
    return [_serialize_record_summary(item) for item in records]


def _build_upload_response(record) -> dict[str, Any]:
    lead_names = list(getattr(record, "lead_names", []))
    metadata = dict(getattr(record, "metadata", {}) or {})
    return {
        "success": True,
        "record_id": f"apple_watch:{record.record_id}",
        "source": metadata.get("source", "apple_watch"),
        "metadata": {
            "patient_id": metadata.get("patient_id"),
            "recorded_at": metadata.get("recorded_at"),
            "device": metadata.get("device"),
            "lead_names": lead_names,
            "num_leads": int(getattr(record, "num_leads", len(lead_names))),
            "sampling_rate": int(record.sampling_rate),
            "duration_seconds": round(float(record.duration_seconds), 1),
        },
        "record": _serialize_record_summary(
            {
                "record_id": f"apple_watch:{record.record_id}",
                "source": metadata.get("source", "apple_watch"),
                "patient_id": metadata.get("patient_id"),
                "age": metadata.get("age"),
                "sex": metadata.get("sex"),
                "recording_date": metadata.get("recorded_at"),
                "diagnostic_class": metadata.get("diagnostic_superclass", []),
                "lead_names": lead_names,
                "num_leads": int(getattr(record, "num_leads", len(lead_names))),
                "sampling_rate": int(record.sampling_rate),
                "duration_seconds": round(float(record.duration_seconds), 1),
            }
        ),
    }


def upload_ecg_json(payload: dict[str, Any]) -> dict[str, Any]:
    extra_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    record = _get_apple_watch_loader().create_record_from_payload(payload, extra_metadata=extra_metadata)
    return _build_upload_response(record)


def upload_ecg_file(*, filename: str, content: bytes, fields: dict[str, str]) -> dict[str, Any]:
    sampling_rate = None
    raw_sampling_rate = (fields.get("sampling_rate") or "").strip()
    if raw_sampling_rate:
        try:
            sampling_rate = int(raw_sampling_rate)
        except ValueError as exc:
            raise AppleWatchUploadError(
                f"Invalid sampling_rate {raw_sampling_rate!r}. Expected an integer Hz value."
            ) from exc

    extra_metadata = {
        key: value
        for key, value in {
            "patient_id": fields.get("patient_id"),
            "recorded_at": fields.get("recorded_at"),
            "device": fields.get("device"),
            "age": fields.get("age"),
            "sex": fields.get("sex"),
        }.items()
        if value not in {None, ""}
    }

    record = _get_apple_watch_loader().create_record_from_upload(
        filename=filename,
        content=content,
        sampling_rate=sampling_rate,
        lead_name=fields.get("lead_name"),
        extra_metadata=extra_metadata,
    )
    return _build_upload_response(record)


def run_chat(user_message: str, record_id: str | None) -> ChatResult:
    init_pipeline()
    logs: list[str] = []
    final_answer = ""
    risk_level: str | None = None
    retry_count = 0

    for response in pipeline.run(  # type: ignore[union-attr]
        user_query=user_message,
        active_record_id=record_id or None,
    ):
        is_status = (
            isinstance(response.context_used, dict)
            and response.context_used.get("event") == "status"
        )
        if is_status:
            logs.append(response.answer)
            continue

        final_answer = response.answer
        retry_count = response.retry_count
        if response.risk_level:
            risk_level = response.risk_level.value.upper()

    return ChatResult(
        answer=final_answer,
        logs=logs,
        risk_level=risk_level,
        retry_count=retry_count,
    )


def run_chat_events(
    user_message: str,
    record_id: str | None,
) -> Iterator[dict[str, Any]]:
    """Stream status and answer events for realtime UI updates."""
    init_pipeline()

    latest_answer = ""
    latest_risk_level: str | None = None
    latest_retry_count = 0

    for response in pipeline.run(  # type: ignore[union-attr]
        user_query=user_message,
        active_record_id=record_id or None,
    ):
        is_status = (
            isinstance(response.context_used, dict)
            and response.context_used.get("event") == "status"
        )

        if is_status:
            yield {
                "type": "status",
                "message": response.answer,
            }
            continue

        latest_answer = response.answer
        latest_retry_count = response.retry_count
        latest_risk_level = (
            response.risk_level.value.upper() if response.risk_level else None
        )

        yield {
            "type": "answer",
            "answer": response.answer,
            "risk_level": latest_risk_level,
            "retry_count": response.retry_count,
        }

    yield {
        "type": "final",
        "answer": latest_answer,
        "risk_level": latest_risk_level,
        "retry_count": latest_retry_count,
    }


class ChatUIHandler(BaseHTTPRequestHandler):
    """Serve chat_ui static files and lightweight JSON APIs."""

    from pathlib import Path
    ui_dir = Path(__file__).resolve().parent / "chat_ui"

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(
            _sanitize_json(payload),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_event(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(
            _sanitize_json(payload),
            ensure_ascii=False,
            allow_nan=False,
        )
        body = f"data: {encoded}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()

    def _send_file(self, file_path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_request_body(self, max_bytes: int) -> bytes:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            content_length = int(raw_length)
        except ValueError as exc:
            raise RequestBodyError("invalid Content-Length header") from exc
        if content_length < 0:
            raise RequestBodyError("invalid Content-Length header")
        if content_length > max_bytes:
            raise RequestTooLargeError(
                f"request body too large ({content_length} bytes); limit is {max_bytes} bytes"
            )
        if content_length == 0:
            return b""

        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            raise RequestBodyError("incomplete request body")
        return raw

    def _read_json_body(self, *, max_bytes: int = MAX_JSON_BODY_BYTES) -> dict[str, Any]:
        raw = self._read_request_body(max_bytes)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _read_multipart_form(
        self,
        *,
        max_bytes: int = MAX_MULTIPART_BODY_BYTES,
    ) -> tuple[str, bytes, dict[str, str]]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise RequestBodyError("expected multipart/form-data content type")

        body = self._read_request_body(max_bytes)
        parser = BytesParser(policy=email_policy_default)
        message = parser.parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        if not message.is_multipart():
            raise RequestBodyError("invalid multipart request body")

        filename: str | None = None
        file_content: bytes | None = None
        fields: dict[str, str] = {}

        for part in message.iter_parts():
            field_name = part.get_param("name", header="content-disposition")
            if not field_name:
                continue
            payload = part.get_payload(decode=True) or b""
            part_filename = part.get_filename()
            if part_filename is not None:
                if field_name == "file":
                    filename = str(part_filename)
                    file_content = payload
                continue
            fields[field_name] = payload.decode(part.get_content_charset() or "utf-8").strip()

        if not filename or file_content is None:
            raise AppleWatchUploadError("Multipart upload must include a file field named 'file'.")

        return filename, file_content, fields

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send_file(self.ui_dir / "index.html", "text/html; charset=utf-8")
            return

        if self.path == "/style.css":
            self._send_file(self.ui_dir / "style.css", "text/css; charset=utf-8")
            return

        if self.path == "/script.js":
            self._send_file(self.ui_dir / "script.js", "application/javascript; charset=utf-8")
            return

        if self.path == "/api/records":
            try:
                self._send_json({"records": list_records()})
            except Exception as exc:
                logger.exception("Failed to list records")
                self._send_json(
                    {"error": f"Failed to list records: {exc}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if self.path == "/api/session/events":
            self._handle_session_events()
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/session/start":
            self._handle_session_start()
            return

        if self.path == "/api/session/chat_stream":
            self._handle_session_chat_stream()
            return

        if self.path == "/api/session/reset_chat":
            self._handle_session_reset_chat()
            return

        session_actions = {
            "/api/session/pause": "pause",
            "/api/session/resume": "resume",
            "/api/session/next": "next",
            "/api/session/stop": "stop",
            "/api/session/speed": "speed",
        }
        if self.path in session_actions:
            self._handle_session_control(session_actions[self.path])
            return

        if self.path == "/api/chat_stream":
            self._handle_chat_stream()
            return

        if self.path == "/api/upload_ecg":
            self._handle_upload_ecg()
            return

        if self.path != "/api/chat":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self._read_json_body()
            message = str(payload.get("message", "")).strip()
            record_id_raw = payload.get("record_id", None)
            record_id = str(record_id_raw).strip() if record_id_raw else None

            if not message:
                self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                return

            result = run_chat(message, record_id)
            self._send_json(
                {
                    "answer": result.answer,
                    "logs": result.logs,
                    "risk_level": result.risk_level,
                    "retry_count": result.retry_count,
                }
            )
        except RequestTooLargeError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        except RequestBodyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("Chat request failed")
            self._send_json(
                {"error": f"chat failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_session_start(self) -> None:
        try:
            payload = self._read_json_body()
            snapshot = live_runtime.start_session(payload)
            self._send_json({"success": True, "snapshot": snapshot}, HTTPStatus.CREATED)
        except LiveSessionConflictError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.CONFLICT)
        except (RequestBodyError, ValueError) as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._send_json({"success": False, "error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("Live session start failed")
            self._send_json(
                {"success": False, "error": f"session start failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_session_control(self, action: str) -> None:
        try:
            payload = self._read_json_body()
            speed = payload.get("speed") if action == "speed" else None
            snapshot = live_runtime.control(action, speed=speed)
            self._send_json({"success": True, "snapshot": snapshot})
        except NoLiveSessionError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.CONFLICT)
        except (RequestBodyError, ValueError, TypeError) as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._send_json({"success": False, "error": "invalid json body"}, HTTPStatus.BAD_REQUEST)

    def _handle_session_reset_chat(self) -> None:
        try:
            self._read_json_body()
            live_runtime.reset_chat_history()
            self._send_json({"success": True})
        except LiveSessionConflictError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.CONFLICT)
        except (RequestBodyError, json.JSONDecodeError) as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _handle_session_events(self) -> None:
        try:
            live_runtime.snapshot()
            events = live_runtime.iter_events()
            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            for event in events:
                self._send_sse_event(event)
        except NoLiveSessionError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.CONFLICT)
        except BrokenPipeError:
            logger.info("Client disconnected from /api/session/events")
        except Exception as exc:
            logger.exception("Live session event stream failed")
            try:
                self._send_sse_event({"type": "error", "message": str(exc)})
            except Exception:
                pass

    def _handle_session_chat_stream(self) -> None:
        acquired = False
        try:
            payload = self._read_json_body()
            message = str(payload.get("message") or "").strip()
            if not message:
                self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                return
            session = live_runtime.acquire_chat_session()
            acquired = True

            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            final_response = None
            history = live_runtime.chat_history_for_prompt()
            for response in session.ask_stream(message, history=history):
                if response.context_used.get("event") == "status":
                    self._send_sse_event({"type": "status", "message": response.answer})
                    continue
                final_response = response
                self._send_sse_event(
                    {
                        "type": "answer",
                        "answer": response.answer,
                        "risk_level": (
                            response.risk_level.value.upper()
                            if response.risk_level
                            else None
                        ),
                        "retry_count": response.retry_count,
                    }
                )

            if final_response is None:
                raise RuntimeError("Reactive pipeline produced no final response.")
            self._send_sse_event(
                {
                    "type": "final",
                    "answer": final_response.answer,
                    "risk_level": (
                        final_response.risk_level.value.upper()
                        if final_response.risk_level
                        else None
                    ),
                    "retry_count": final_response.retry_count,
                    "tool_results": final_response.context_used.get("tool_results", []),
                }
            )
            live_runtime.record_chat_turn(message, final_response.answer)
            self._send_sse_event({"type": "done"})
        except LiveSessionConflictError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except NoLiveSessionError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (RequestBodyError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            logger.info("Client disconnected from /api/session/chat_stream")
        except Exception as exc:
            logger.exception("Live session chat failed")
            try:
                self._send_sse_event({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            if acquired:
                live_runtime.release_chat_session()

    def _handle_upload_ecg(self) -> None:
        try:
            content_type = self.headers.get("Content-Type", "")
            if content_type.startswith("application/json"):
                payload = self._read_json_body(max_bytes=MAX_JSON_BODY_BYTES)
                response = upload_ecg_json(payload)
            elif content_type.startswith("multipart/form-data"):
                filename, content, fields = self._read_multipart_form(
                    max_bytes=MAX_MULTIPART_BODY_BYTES
                )
                response = upload_ecg_file(filename=filename, content=content, fields=fields)
            else:
                self._send_json(
                    {"error": "upload must use application/json or multipart/form-data"},
                    HTTPStatus.BAD_REQUEST,
                )
                return

            self._send_json(response, HTTPStatus.CREATED)
        except RequestTooLargeError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        except RequestBodyError as exc:
            self._send_json({"success": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except AppleWatchUploadError as exc:
            self._send_json(
                {"success": False, "error": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("Upload request failed")
            self._send_json(
                {"success": False, "error": f"upload failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_chat_stream(self) -> None:
        try:
            payload = self._read_json_body()
            message = str(payload.get("message", "")).strip()
            record_id_raw = payload.get("record_id", None)
            record_id = str(record_id_raw).strip() if record_id_raw else None

            if not message:
                self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                return

            self.send_response(HTTPStatus.OK.value)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()

            for event in run_chat_events(message, record_id):
                self._send_sse_event(event)

            self._send_sse_event({"type": "done"})
        except RequestTooLargeError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        except RequestBodyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json body"}, HTTPStatus.BAD_REQUEST)
        except BrokenPipeError:
            logger.info("Client disconnected during /api/chat_stream")
        except Exception as exc:
            logger.exception("Chat stream request failed")
            try:
                self._send_sse_event({"type": "error", "message": f"chat failed: {exc}"})
            except Exception:
                pass


def serve(host: str | None = None, port: int | None = None) -> None:
    host = host or os.getenv("WEB_DEMO_HOST", "127.0.0.1")
    port = port or int(os.getenv("WEB_DEMO_PORT", "7860"))

    try:
        server = ThreadingHTTPServer((host, port), ChatUIHandler)
    except OSError as exc:
        logger.error("Port %s unavailable: %s", port, exc)
        logger.error("Please stop the process on port %s, then retry.", port)
        raise

    logger.info("chat_ui server started at http://%s:%s", host, port)
    logger.info("Serving static UI from: %s", ChatUIHandler.ui_dir)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the VitalAgent web demo.")
    parser.add_argument(
        "--host",
        default=os.getenv("WEB_DEMO_HOST", "127.0.0.1"),
        help="Host interface to bind. Defaults to WEB_DEMO_HOST or 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WEB_DEMO_PORT", "7860")),
        help="Port to bind. Defaults to WEB_DEMO_PORT or 7860.",
    )
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
