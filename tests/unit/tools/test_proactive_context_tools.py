from __future__ import annotations

import json

import pytest

import agent.tools.proactive_context_tools as proactive_tools
from agent.tools.registry import list_tools


@pytest.fixture(autouse=True)
def clear_contexts():
    proactive_tools._CONTEXTS.clear()
    yield
    proactive_tools._CONTEXTS.clear()


def _context(patient_id: str = "01182") -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "alert_summary": {
            "total_alerts": 2,
            "false_alert_count": 1,
            "matched_alert_count": 1,
            "by_rule": {"sustained_tachycardia": 2},
        },
        "recent_alerts": [
            {
                "alert_id": 0,
                "patient_id": patient_id,
                "segment_id": "00",
                "window_index": 75,
                "offset_s": 750.0,
                "urgency": "medium",
                "triggered_rules": ["sustained_tachycardia"],
                "reason": "Heart rate has exceeded 100 bpm in 80% of recent readings.",
                "matched_episode_id": 0,
                "matched_episode_label": "AF",
                "matched_episode_start_offset_s": 590.0,
                "matched_episode_duration_s": 190.0,
                "latency_seconds": 160.0,
                "is_false_alert": False,
            },
            {
                "alert_id": 1,
                "patient_id": patient_id,
                "segment_id": "00",
                "window_index": 220,
                "offset_s": 2200.0,
                "urgency": "medium",
                "triggered_rules": ["sustained_tachycardia"],
                "reason": "Heart rate has exceeded 100 bpm in 80% of recent readings.",
                "latency_seconds": None,
                "is_false_alert": True,
            },
        ],
    }


def test_proactive_context_tools_are_registered() -> None:
    names = {tool["name"] for tool in list_tools()}

    assert {
        "proactive_load_patient_context",
        "proactive_list_patient_contexts",
        "proactive_get_recent_alerts",
        "proactive_explain_last_alert",
    }.issubset(names)


def test_load_patient_context_from_replay_jsonl(tmp_path) -> None:
    path = tmp_path / "replay.jsonl"
    records = [
        {"record_type": "patient_context", **_context()},
        {"record_type": "window", "window_index": 1},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = proactive_tools.proactive_load_patient_context(str(path), clear_existing=True)

    assert result["success"] is True
    assert result["context_count"] == 1
    assert result["contexts"][0]["patient_id"] == "01182"
    assert result["contexts"][0]["alert_count"] == 2


def test_get_recent_alerts_filters_false_alerts() -> None:
    proactive_tools._CONTEXTS["01182"] = _context()

    result = proactive_tools.proactive_get_recent_alerts(
        patient_id="01182",
        include_false_alerts=False,
    )

    assert result["success"] is True
    assert result["alert_count"] == 1
    assert result["alerts"][0]["matched_episode_label"] == "AF"


def test_explain_last_alert_returns_plain_language_summary() -> None:
    proactive_tools._CONTEXTS["01182"] = _context()

    result = proactive_tools.proactive_explain_last_alert(patient_id="01182")

    assert result["success"] is True
    assert result["alert"]["alert_id"] == 1
    assert result["explanation"]["is_false_alert"] is True
    assert "sustained_tachycardia" in result["explanation"]["plain_language_summary"]


def test_explain_last_alert_requires_patient_id_when_ambiguous() -> None:
    proactive_tools._CONTEXTS["01182"] = _context("01182")
    proactive_tools._CONTEXTS["00012"] = _context("00012")

    result = proactive_tools.proactive_explain_last_alert()

    assert result == {
        "success": False,
        "error": "Multiple proactive patient contexts are loaded; provide patient_id.",
    }
