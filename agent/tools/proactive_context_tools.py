"""Reactive tools for reading proactive PatientContext snapshots."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

from agent.tools.registry import register_tool

_CONTEXTS: dict[str, dict[str, Any]] = {}
_CONTEXTS_LOCK = threading.Lock()


def _context_patient_id(context: dict[str, Any]) -> str:
    return str(context.get("patient_id") or "").strip()


def _context_summary(context: dict[str, Any]) -> dict[str, Any]:
    patient_id = _context_patient_id(context)
    alerts = context.get("recent_alerts") or []
    alert_summary = context.get("alert_summary") or {}
    return {
        "patient_id": patient_id,
        "alert_count": int(alert_summary.get("total_alerts") or len(alerts)),
        "false_alert_count": int(alert_summary.get("false_alert_count") or 0),
        "matched_alert_count": int(alert_summary.get("matched_alert_count") or 0),
        "available_sections": sorted(context.keys()),
    }


def _load_context_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Patient context path does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        contexts: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("record_type") != "patient_context":
                    continue
                context = {k: v for k, v in record.items() if k != "record_type"}
                contexts.append(context)
        return contexts

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "patient_context" in payload:
        nested = payload["patient_context"]
        return [nested] if isinstance(nested, dict) else []
    if isinstance(payload, dict) and "recent_alerts" in payload:
        return [payload]
    return []


def _store_context(context: dict[str, Any]) -> None:
    patient_id = _context_patient_id(context)
    if not patient_id:
        raise ValueError("Patient context is missing patient_id.")
    _CONTEXTS[patient_id] = context


def register_patient_context(
    context: dict[str, Any],
    *,
    clear_existing: bool = False,
) -> None:
    """Register an immutable snapshot for live reactive access."""
    snapshot = copy.deepcopy(context)
    with _CONTEXTS_LOCK:
        if clear_existing:
            _CONTEXTS.clear()
        _store_context(snapshot)


def _resolve_context(patient_id: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    with _CONTEXTS_LOCK:
        if patient_id:
            context = _CONTEXTS.get(str(patient_id))
            if context is None:
                return None, f"No proactive patient context found for patient_id={patient_id!r}."
            return copy.deepcopy(context), None
        if not _CONTEXTS:
            return None, "No proactive patient context is loaded."
        if len(_CONTEXTS) > 1:
            return None, "Multiple proactive patient contexts are loaded; provide patient_id."
        return copy.deepcopy(next(iter(_CONTEXTS.values()))), None


def _limit_alerts(alerts: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    if limit < 1:
        limit = 1
    return alerts[-limit:]


def _select_alert(
    alerts: list[dict[str, Any]],
    *,
    offset_s: float | None = None,
    alert_id: int | None = None,
) -> dict[str, Any]:
    """Pick the alert to explain: by stable id, else nearest offset, else latest."""
    if alert_id is not None:
        for alert in alerts:
            if alert.get("alert_id") == alert_id:
                return alert
    if offset_s is not None:
        try:
            target = float(offset_s)
        except (TypeError, ValueError):
            target = None
        if target is not None:
            return min(
                alerts,
                key=lambda alert: abs(float(alert.get("offset_s") or 0.0) - target),
            )
    return alerts[-1]


def _alert_explanation(alert: dict[str, Any]) -> dict[str, Any]:
    rules = alert.get("triggered_rules") or []
    matched_label = alert.get("matched_episode_label")
    latency = alert.get("latency_seconds")
    reason = str(alert.get("reason") or "")

    summary_parts: list[str] = []
    if rules:
        summary_parts.append(f"Triggered rule(s): {', '.join(str(rule) for rule in rules)}.")
    if reason:
        summary_parts.append(reason)
    if matched_label:
        episode_text = f"The alert aligned with a {matched_label} episode"
        if latency is not None:
            episode_text += f" about {float(latency):.0f} seconds after episode onset"
        summary_parts.append(episode_text + ".")
    if alert.get("is_false_alert"):
        summary_parts.append("Offline replay marked this alert as outside annotated abnormal episodes.")

    return {
        "alert_id": alert.get("alert_id"),
        "patient_id": alert.get("patient_id"),
        "offset_s": alert.get("offset_s"),
        "urgency": alert.get("urgency"),
        "triggered_rules": rules,
        "reason": reason,
        "matched_episode_label": matched_label,
        "latency_seconds": latency,
        "is_false_alert": bool(alert.get("is_false_alert")),
        "plain_language_summary": " ".join(summary_parts).strip(),
    }


@register_tool(
    name="proactive_load_patient_context",
    description=(
        "Load proactive PatientContext JSON/JSONL snapshots produced by intervention replay."
    ),
    parameters={
        "context_path": {
            "type": "string",
            "description": "Path to a PatientContext JSON file or replay debug JSONL.",
            "required": True,
        },
        "clear_existing": {
            "type": "boolean",
            "description": "Clear already-loaded proactive contexts before loading.",
            "default": False,
        },
    },
)
def proactive_load_patient_context(
    context_path: str,
    clear_existing: bool = False,
) -> dict[str, Any]:
    try:
        contexts = _load_context_records(Path(context_path))
        with _CONTEXTS_LOCK:
            if clear_existing:
                _CONTEXTS.clear()
            for context in contexts:
                _store_context(copy.deepcopy(context))
        return {
            "success": True,
            "context_count": len(contexts),
            "contexts": [_context_summary(context) for context in contexts],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@register_tool(
    name="proactive_list_patient_contexts",
    description="List loaded proactive PatientContext snapshots.",
)
def proactive_list_patient_contexts() -> dict[str, Any]:
    with _CONTEXTS_LOCK:
        contexts = [copy.deepcopy(context) for context in _CONTEXTS.values()]
    return {
        "success": True,
        "contexts": [_context_summary(context) for context in contexts],
    }


@register_tool(
    name="proactive_get_recent_alerts",
    description="Return recent proactive intervention alerts for follow-up QA.",
    parameters={
        "patient_id": {
            "type": "string",
            "description": "Patient ID. Optional when only one proactive context is loaded.",
            "required": False,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of recent alerts to return.",
            "default": 5,
        },
        "include_false_alerts": {
            "type": "boolean",
            "description": "Whether to include alerts marked false by offline replay.",
            "default": True,
        },
    },
)
def proactive_get_recent_alerts(
    patient_id: str | None = None,
    limit: int = 5,
    include_false_alerts: bool = True,
) -> dict[str, Any]:
    context, error = _resolve_context(patient_id)
    if error:
        return {"success": False, "error": error}
    alerts = list(context.get("recent_alerts") or [])
    if not include_false_alerts:
        alerts = [alert for alert in alerts if not alert.get("is_false_alert")]
    alerts = _limit_alerts(alerts, limit)
    return {
        "success": True,
        "patient_id": _context_patient_id(context),
        "alert_count": len(alerts),
        "alerts": alerts,
        "alert_summary": context.get("alert_summary") or {},
    }


@register_tool(
    name="proactive_explain_last_alert",
    description="Explain the most recent proactive alert using PatientContext memory.",
    parameters={
        "patient_id": {
            "type": "string",
            "description": "Patient ID. Optional when only one proactive context is loaded.",
            "required": False,
        },
    },
)
def proactive_explain_last_alert(
    patient_id: str | None = None,
    offset_s: float | None = None,
    alert_id: int | None = None,
) -> dict[str, Any]:
    # offset_s / alert_id are optional locators used to explain a SPECIFIC stored
    # alert (e.g. the exact card the user clicked); when both are omitted this
    # explains the most recent alert, preserving the original behaviour.
    context, error = _resolve_context(patient_id)
    if error:
        return {"success": False, "error": error}
    alerts = list(context.get("recent_alerts") or [])
    if not alerts:
        return {
            "success": False,
            "error": "No proactive alerts are stored for this patient context.",
        }
    alert = _select_alert(alerts, offset_s=offset_s, alert_id=alert_id)
    return {
        "success": True,
        "patient_id": _context_patient_id(context),
        "alert": alert,
        "explanation": _alert_explanation(alert),
    }
