"""VitalBench reactive evaluation"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from agent.benchmarks.vitalbench.eval_loader import (
    AgentInput,
    EvalSpec,
    QASample,
    load_qa_jsonl as stream_qa_jsonl,
)
from agent.benchmarks.vitalbench.evaluator import (
    boolean_match,  # noqa: F401 - re-exported for existing evaluation tests/helpers
    evaluate_prediction,
    evaluate_prediction_details,
    normalized_match,  # noqa: F401 - re-exported for existing evaluation tests/helpers
    numeric_tolerance_match,  # noqa: F401 - re-exported for existing evaluation tests/helpers
)
from agent.config import get_config, resolved_config_snapshot
from agent.llm import (
    merge_usage,
    zero_usage,
)
from agent.mhealth.schemas import MonitoringState
from agent.mhealth.schemas import canonicalize_dataset_name
from agent.reactive.previous_window_compare import PREVIOUS_WINDOW_COMPARISON_RESULT
from agent.reactive.validation import ValidationGate
from agent.state.mhealth_state_store import get_global_state_store
from agent.tools.registry import call_tool, list_tools, register_tool

import agent.tools.datasets.icentia11k  # noqa: F401
import agent.tools.datasets.ppg_dalia  # noqa: F401
import agent.tools.datasets.wesad  # noqa: F401
import agent.tools.datasets.wesad_stress_classifier  # noqa: F401
import agent.tools.ecg.tools  # noqa: F401
import agent.tools.mhealth_signal_state_build_tools  # noqa: F401
import agent.tools.mhealth_state_build_tools  # noqa: F401
import agent.tools.mhealth_state_tools  # noqa: F401
import agent.tools.ppg.tools  # noqa: F401
import agent.tools.proactive_evaluate_tools  # noqa: F401


QUERY_TOOL_PREFIX = "state_get_"
SAFE_BUILD_TOOL = "eval_state_build_missing_context"
LOCATOR_STATE_TOOL_NAMES = {
    "state_get_current_monitoring_state",
    "state_get_monitoring_window",
    "state_get_longitudinal_trend",
    "state_get_dataset_capabilities",
}
# Signal-mode: datasets where agent calls raw-signal tools directly (no state store).
SIGNAL_TOOL_NAMES = {
    "list_ppg_records",
    "get_ppg_description",
    "get_ppg_metadata",
    "analyze_pulse_rate",
    "analyze_prv",
    "assess_ppg_signal_quality",
    "analyze_ppg_rhythm_irregularity",
}
RAW_SIGNAL_TOOL_NAMES = SIGNAL_TOOL_NAMES | {
    "list_ecg_records",
    "get_ecg_description",
    "get_ecg_metadata",
    "analyze_heart_rate",
    "analyze_hrv",
    "assess_signal_quality",
    "ecg_diagnosis",
    "analyze_icentia11k_ecg_window_signal",
    "analyze_ppg_dalia_window_signal",
    "analyze_wesad_window_signal",
    "classify_wesad_stress_state",
}
RAW_SIGNAL_TOOL_NAMES_BY_DATASET = {
    "afppgecg": SIGNAL_TOOL_NAMES
    | {
        "evaluate_proactive_rules",
        "analyze_afppgecg_rhythm_context",
    },
    "ppg": SIGNAL_TOOL_NAMES,
    "ppg_dalia": {
        "analyze_ppg_dalia_window_signal",
        "evaluate_proactive_rules",
    },
    "wesad": {
        "analyze_wesad_window_signal",
        "evaluate_proactive_rules",
    },
    "icentia11k": {
        "analyze_icentia11k_ecg_window_signal",
        "evaluate_proactive_rules",
    },
}
ADAPTIVE_SCOPE_TOOL_NAMES = {
    "analyze_afppgecg_rhythm_context",
}
ADAPTIVE_SCOPE_TARGETS = {
    "af_presence_monitoring_window",
    "af_frequency_monitoring_window",
    "rhythm_variability_monitoring_window",
}
RAW_TOOL_DESCRIPTION_OVERRIDES = {
    "analyze_ppg_dalia_window_signal": (
        "Analyze raw synchronized PPG-DaLiA signals for one subject window. "
        "Returns signal-derived wrist BVP pulse metrics including peak pulse-rate "
        "evidence for HR spike/threshold questions, BVP quality, optional chest "
        "ECG analysis, wrist motion, EDA, temperature, and caveats. "
        "Reference labels and protocol activity labels are hidden during eval."
    ),
    "analyze_wesad_window_signal": (
        "Analyze raw synchronized WESAD signals for one subject window. "
        "Returns signal-derived wrist BVP pulse metrics, wrist motion, EDA, "
        "temperature, respiration summary, optional chest ECG summary, and caveats. "
        "Protocol labels are hidden during eval."
    ),
    "classify_wesad_stress_state": (
        "Predict a WESAD window's baseline / stress / amusement / meditation state "
        "with a supervised classifier trained from raw-derived BVP, EDA, ACC, "
        "temperature, and respiration features. The tool enforces subject-wise "
        "training isolation and excludes release eval subjects by default; if no "
        "non-leaking training subjects are available, it returns a leakage-guard error."
    ),
    "analyze_icentia11k_ecg_window_signal": (
        "Analyze a raw single-lead Icentia11k ECG signal window. Returns signal-derived "
        "R-peak heart-rate statistics, annotation-compatible mean heart rate when "
        "Icentia11k beat annotations are available, RR-irregularity screening metrics, "
        "signal quality caveats, and an AF-screening heuristic. It does not expose "
        "WFDB rhythm annotations."
    ),
    "evaluate_proactive_rules": (
        "Replay the proactive engine's guideline-grounded rule layer over raw ECG "
        "sub-windows in the locator range. Returns rule triggers, highest urgency, "
        "guideline citations, HR/rhythm traces, and aggregate rhythm_summary fields "
        "without exposing dataset labels. For PPG-DaLiA HR spike questions, prefer "
        "the PPG-DaLiA raw-window tool's wrist BVP peak-rate evidence."
    ),
    "analyze_afppgecg_rhythm_context": (
        "Adaptive-scope rhythm analysis for afppgecg using signal-derived "
        "classification on aligned-ECG R-peak positions. Returns anchor "
        "window + sampled-recording extended context. Does not read clinical "
        "AF annotation labels."
    ),
}
RAW_HIDDEN_PARAMETER_NAMES = {
    "min_label_fraction",
}
RAW_LABEL_LEAK_KEYS = {
    "activity",
    "dominant_activity_fraction",
    "dominant_activity_label",
    "dominant_activity_label_id",
    "dominant_label_fraction",
    "dominant_protocol_label",
    "dominant_protocol_label_id",
    "ecg_reference_hr_label",
    "label_counts",
    "protocol_label",
}
LABEL_GENERATED_STATE_FIELDS = frozenset(
    {
        "rhythm_class",
        "previous_rhythm_class",
        "stress_label",
        "previous_stress_label",
        "protocol_label_id",
        "previous_protocol_label_id",
        "activity_label",
        "previous_activity_label",
        "af_burden_ratio",
        "stress_burden_ratio",
        "stress_duration_s",
        "dominant_stress_label",
        "rhythm_transition_count_per_hour",
        "stress_transition_count_per_hour",
        "rhythm_transition_count",
        "stress_transition_count",
    }
)
LABEL_GENERATED_METADATA_KEY_SUBSTRINGS = (
    "annotation",
    "ground_truth",
    "label",
    "protocol",
    "rhythm_gt",
    "rhythm_class_source",
    "rhythm_transition_source",
)
RAW_SIGNAL_OBSERVABILITY_BY_TARGET = {
    # Directly signal-derived or heuristic-screening targets available from
    # eval-safe raw-signal tools.
    "af_presence_current": "raw_signal_observable",
    "heart_rate_category_current": "raw_signal_observable",
    "heart_rate_numeric_current": "raw_signal_observable",
    "heart_rate_higher_than_previous_window": "raw_signal_observable",
    "rhythm_changed_from_previous_window": "raw_signal_observable",
    # Dataset/protocol/annotation labels are not fairly recoverable from raw
    # sensor summaries alone without exposing hidden labels.
    "stress_presence_current": "annotation_grounded",
    "stress_state_current": "annotation_grounded",
    "protocol_label_id_query_current": "annotation_grounded",
    "stress_state_changed_from_previous_window": "annotation_grounded",
}
PPG_DATASETS = {"afppgecg"}
ALLOWED_ERROR_CLASSES = {
    "context_store_issue",
    "planner_issue",
    "answer_formatting_issue",
    "benchmark_evidence_issue",
    "model_reasoning_issue",
    "build_unavailable",
    "tool_error",
    "unknown",
}
_ICENTIA_SEGMENT_RE = re.compile(r"(?:^|_)segment_(\d+)(?:_|$)")
TOKEN_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
)
TRACE_PREDICTION_SUMMARY_KEYS = (
    "validation_first_passed",
    "validation_first_coverage",
    "validation_last_passed",
    "validation_last_coverage",
    "coverage_improved",
    "validation_issue_types",
    "replan_count",
    "replan_triggered",
    "initial_tool_names",
    "initial_plan_signature",
    "final_tool_names",
    "final_plan_signature",
    "plan_changed_after_replan",
    "perturbation_count",
    "tool_call_count",
    "llm_call_count",
)
MODEL_COST_KEYS = (
    "planning_model_call_count",
    "answer_model_call_count",
    "planning_input_tokens",
    "planning_output_tokens",
    "answer_input_tokens",
    "answer_output_tokens",
)
EVALUATION_PROTOCOLS = ("paper_v1", "rebuttal_v2")
AGENT_LIKE_CONDITIONS = {
    "agent",
    "agent_no_validation",
    "agent_no_replan",
    "agent_no_planner",
    "agent_no_planner_paper_v1",
    "agent_no_tools",
    "agent_all_tools",
}


def effective_raw_tool_names(
    dataset: str | None,
    *,
    adaptive_scope_enabled: bool,
) -> set[str]:
    """Return the exact raw tool pool exposed for one evaluation sample."""

    pool = set(
        RAW_SIGNAL_TOOL_NAMES_BY_DATASET.get(str(dataset), RAW_SIGNAL_TOOL_NAMES)
        if dataset is not None
        else RAW_SIGNAL_TOOL_NAMES
    )
    if not adaptive_scope_enabled:
        pool -= ADAPTIVE_SCOPE_TOOL_NAMES
    return pool
DEFAULT_VITALBENCH_DIR = get_config().datasets.vitalbench_root
DEFAULT_QA_JSONL = DEFAULT_VITALBENCH_DIR / "qa.jsonl"
DEFAULT_STATES_JSONL = DEFAULT_VITALBENCH_DIR / "states.jsonl"
DEFAULT_SOURCE_MANIFEST = DEFAULT_VITALBENCH_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = Path(".runtime/vitalbench_eval")
VALID_EVAL_CONDITIONS = (
    "agent",
    "agent_no_validation",
    "agent_no_replan",
    "agent_no_planner",
    "agent_no_planner_paper_v1",
    "agent_no_tools",
    "agent_all_tools",
)


def log_info(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ContextKey:
    dataset: str
    subject_id: str
    recording_id: str


@dataclass
class ModelCallTracker:
    planning_model_call_count: int = 0
    answer_model_call_count: int = 0
    planning_usage: dict[str, int] = field(default_factory=zero_usage)
    answer_usage: dict[str, int] = field(default_factory=zero_usage)

    def begin(self, role: str) -> None:
        if role == "planner":
            self.planning_model_call_count += 1
        else:
            self.answer_model_call_count += 1

    def add_usage(self, role: str, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        if role == "planner":
            self.planning_usage = merge_usage(self.planning_usage, usage)
        else:
            self.answer_usage = merge_usage(self.answer_usage, usage)

    def to_fields(self) -> dict[str, int]:
        return {
            "planning_model_call_count": self.planning_model_call_count,
            "answer_model_call_count": self.answer_model_call_count,
            "planning_input_tokens": int(self.planning_usage.get("input_tokens", 0)),
            "planning_output_tokens": int(self.planning_usage.get("output_tokens", 0)),
            "answer_input_tokens": int(self.answer_usage.get("input_tokens", 0)),
            "answer_output_tokens": int(self.answer_usage.get("output_tokens", 0)),
        }


def attach_model_call_tracker(service: Any, tracker: ModelCallTracker) -> None:
    """Track logical model invocations on this pipeline-local LLM service."""

    original_complete = service.complete
    original_stream = service.stream

    def tracked_complete(*, profile: Any, messages: list[dict[str, str]]) -> Any:
        role = str(getattr(profile, "role", ""))
        tracker.begin(role)
        result = original_complete(profile=profile, messages=messages)
        tracker.add_usage(role, getattr(result, "usage", None))
        return result

    def tracked_stream(*, profile: Any, messages: list[dict[str, str]]) -> Iterator[Any]:
        role = str(getattr(profile, "role", ""))
        tracker.begin(role)
        final_usage: dict[str, int] | None = None
        for event in original_stream(profile=profile, messages=messages):
            if getattr(event, "usage", None) is not None:
                final_usage = event.usage
            yield event
        tracker.add_usage(role, final_usage)

    service.complete = tracked_complete
    service.stream = tracked_stream


@dataclass
class ManifestLookup:
    build_supported: bool
    builder_tool: str | None = None
    builder_args: dict[str, Any] | None = None
    entry: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class SplitResult:
    preloaded_contexts: set[ContextKey]
    missing_contexts: set[ContextKey]
    routing_analysis_limited: bool


@dataclass(frozen=True)
class PerturbationConfig:
    rate: float = 0.0
    mode: str = "tool_failure"
    target: str = "critical"
    seed: int = 42

    @property
    def enabled(self) -> bool:
        return self.rate > 0.0

    def normalized_rate(self) -> float:
        return min(1.0, max(0.0, float(self.rate)))

    def allows_tool(self, tool_name: str) -> bool:
        if self.target == "critical":
            return tool_name in ValidationGate.CRITICAL_TOOLS
        return True

    def should_inject(
        self,
        *,
        question_id: str,
        tool_name: str,
        occurrence_index: int,
    ) -> bool:
        if not self.enabled or not self.allows_tool(tool_name):
            return False
        key = f"{self.seed}\0{question_id}\0{tool_name}\0{occurrence_index}"
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        score = int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
        return score < self.normalized_rate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": self.rate,
            "mode": self.mode,
            "target": self.target,
            "seed": self.seed,
        }


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(value))


def _truncate_for_trace(value: Any, max_chars: int) -> Any:
    """Keep trace payloads readable while preserving the shape of tool outputs."""
    if max_chars <= 0:
        return "[omitted]"
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=False)
    if len(text) <= max_chars:
        return json.loads(text)
    return {
        "truncated": True,
        "original_chars": len(text),
        "preview": text[:max_chars],
    }


def _plan_signature(plan_payload: dict[str, Any] | None) -> str | None:
    """Stable hash of an ordered plan's tool names and arguments."""
    if not isinstance(plan_payload, dict):
        return None
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return None
    normalized_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        normalized_steps.append(
            {
                "tool_name": str(step.get("tool_name") or ""),
                "tool_args": step.get("tool_args") or {},
            }
        )
    payload = json.dumps(
        normalized_steps,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _plan_tool_names(plan_payload: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan_payload, dict):
        return []
    steps = plan_payload.get("steps")
    if not isinstance(steps, list):
        return []
    return [
        str(step.get("tool_name"))
        for step in steps
        if isinstance(step, dict) and step.get("tool_name")
    ]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


class EvalTraceRecorder:
    """Per-sample structured trace for diagnosing slow eval stages."""

    def __init__(
        self,
        question_id: str,
        *,
        condition: str | None = None,
        evaluation_protocol: str | None = None,
        tier: str | None = None,
        template_id: str | None = None,
        target: str | None = None,
        dataset: str | None = None,
        quiet: bool,
        result_max_chars: int,
    ):
        self.question_id = question_id
        self.condition = condition
        self.evaluation_protocol = evaluation_protocol
        self.tier = tier
        self.template_id = template_id
        self.target = target
        self.dataset = dataset
        self.quiet = quiet
        self.result_max_chars = result_max_chars
        self.started = time.monotonic()
        self.current_attempt = 0
        self.events: list[dict[str, Any]] = []

    def add(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        duration_sec: float | None = None,
        stdout: str | None = None,
    ) -> None:
        now = time.monotonic()
        record: dict[str, Any] = {
            "question_id": self.question_id,
            "event": event,
            "attempt": self.current_attempt,
            "elapsed_sec": round(now - self.started, 6),
        }
        if duration_sec is not None:
            record["duration_sec"] = round(duration_sec, 6)
        if payload:
            record.update(payload)
        self.events.append(record)

        if stdout and not self.quiet:
            log_info(
                f"[VitalBench trace] {self.question_id} "
                f"+{record['elapsed_sec']:.1f}s {stdout}",
            )

    def summarize(self) -> dict[str, Any]:
        tool_events = [event for event in self.events if event.get("event") == "tool_done"]
        orchestration_events = [
            event for event in self.events if event.get("event") == "orchestration_done"
        ]
        perturbation_events = [
            event
            for event in self.events
            if event.get("event") == "tool_perturbation_injected"
        ]
        planner_events = [
            event for event in self.events
            if event.get("event") in {"planner_done", "replan_done"}
        ]
        answer_done = next(
            (event for event in reversed(self.events) if event.get("event") == "answer_done"),
            None,
        )
        validation_events = [
            event for event in self.events if event.get("event") == "validation_done"
        ]
        initial_validation = next(
            (event for event in validation_events if int(event.get("attempt", -1)) == 0),
            None,
        )
        final_validation = validation_events[-1] if validation_events else None
        validation_first_coverage = (
            None
            if initial_validation is None or initial_validation.get("coverage") is None
            else float(initial_validation.get("coverage"))
        )
        validation_last_coverage = (
            None
            if final_validation is None or final_validation.get("coverage") is None
            else float(final_validation.get("coverage"))
        )
        coverage_improved = (
            None
            if validation_first_coverage is None or validation_last_coverage is None
            else validation_last_coverage > validation_first_coverage
        )
        replan_events = [event for event in self.events if event.get("event") == "replan_done"]
        plan_done_events = [
            event
            for event in self.events
            if event.get("event") in {"planner_done", "replan_done"}
        ]
        initial_plan_event = next(
            (event for event in plan_done_events if int(event.get("attempt", -1)) == 0),
            None,
        )
        executed_attempts = {
            int(event.get("attempt", 0))
            for event in tool_events
            if event.get("attempt") is not None
        }
        final_attempt = max(executed_attempts) if executed_attempts else None
        final_plan_event = (
            next(
                (
                    event
                    for event in reversed(plan_done_events)
                    if int(event.get("attempt", -1)) == final_attempt
                ),
                None,
            )
            if final_attempt is not None
            else (plan_done_events[-1] if plan_done_events else None)
        )
        initial_plan_signature = (
            initial_plan_event.get("plan_signature")
            if isinstance(initial_plan_event, dict)
            else None
        )
        final_plan_signature = (
            final_plan_event.get("plan_signature")
            if isinstance(final_plan_event, dict)
            else None
        )
        validation_issue_types = sorted(
            {
                str(issue_type)
                for event in validation_events
                for issue_type in (event.get("issue_types") or [])
            }
        )
        replan_count = len(replan_events)
        llm_call_count = (
            len([event for event in self.events if event.get("event") == "planner_done"])
            + replan_count
            + len([event for event in self.events if event.get("event") == "answer_done"])
        )
        return {
            "duration_sec": round(time.monotonic() - self.started, 6),
            "event_count": len(self.events),
            "planner_duration_sec": round(
                sum(float(event.get("duration_sec") or 0.0) for event in planner_events),
                6,
            ),
            "tool_duration_sec": round(
                sum(float(event.get("duration_sec") or 0.0) for event in tool_events),
                6,
            ),
            "orchestration_duration_sec": round(
                sum(
                    float(event.get("duration_sec") or 0.0)
                    for event in orchestration_events
                ),
                6,
            ),
            "answer_duration_sec": answer_done.get("answer_duration_sec") if answer_done else None,
            "tool_calls": [
                {
                    "tool_name": event.get("tool_name"),
                    "success": event.get("success"),
                    "duration_sec": event.get("duration_sec"),
                }
                for event in tool_events
            ],
            "validation_first_passed": (
                None if initial_validation is None else bool(initial_validation.get("passed"))
            ),
            "validation_first_coverage": validation_first_coverage,
            "validation_last_passed": (
                None if final_validation is None else bool(final_validation.get("passed"))
            ),
            "validation_last_coverage": validation_last_coverage,
            "coverage_improved": coverage_improved,
            "validation_critical_first": (
                None if initial_validation is None else bool(initial_validation.get("has_critical"))
            ),
            "validation_issue_types": validation_issue_types,
            "replan_count": replan_count,
            "replan_triggered": replan_count > 0,
            "initial_tool_names": (
                list(initial_plan_event.get("tool_names") or [])
                if isinstance(initial_plan_event, dict)
                else []
            ),
            "final_tool_names": (
                list(final_plan_event.get("tool_names") or [])
                if isinstance(final_plan_event, dict)
                else []
            ),
            "initial_plan_signature": initial_plan_signature,
            "final_plan_signature": final_plan_signature,
            "plan_changed_after_replan": (
                initial_plan_signature is not None
                and final_plan_signature is not None
                and initial_plan_signature != final_plan_signature
            ),
            "perturbation_count": len(perturbation_events),
            "tool_call_count": len(tool_events),
            "llm_call_count": llm_call_count,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "condition": self.condition,
            "evaluation_protocol": self.evaluation_protocol,
            "tier": self.tier,
            "template_id": self.template_id,
            "target": self.target,
            "dataset": self.dataset,
            "summary": self.summarize(),
            "events": self.events,
        }


class SourceManifest:
    """Lookup safe builder arguments by dataset/subject/recording."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.entries = list(payload.get("entries") or [])

    @classmethod
    def load(cls, path: str | Path) -> "SourceManifest":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def lookup(
        self,
        dataset: str,
        subject_id: str | None,
        recording_id: str | None = None,
    ) -> ManifestLookup:
        subject = "" if subject_id is None else str(subject_id)
        recording = "" if recording_id is None else str(recording_id)
        candidates: list[dict[str, Any]] = []
        for entry in self.entries:
            if canonicalize_dataset_name(entry.get("dataset")) != canonicalize_dataset_name(dataset):
                continue
            entry_subject = str(entry.get("subject_id") or "")
            entry_recording = str(entry.get("recording_id") or entry_subject)
            if subject and entry_subject != subject:
                continue
            if recording and entry_recording != recording:
                continue
            candidates.append(entry)

        if not candidates:
            registry_lookup = lookup_builder_registry(dataset, subject_id, recording_id)
            if registry_lookup.build_supported:
                return registry_lookup
            if recording:
                retry_lookup = self.lookup(dataset, subject_id, None)
                if retry_lookup.build_supported:
                    return retry_lookup
            return ManifestLookup(
                build_supported=False,
                builder_tool=registry_lookup.builder_tool,
                error=(
                    "No source manifest entry for "
                    f"dataset={dataset}, subject_id={subject}, recording_id={recording}. "
                    f"{registry_lookup.error or ''}"
                ),
            )

        entry = candidates[0]
        build_supported = bool(entry.get("build_supported", entry.get("builder_tool") is not None))
        builder_tool = entry.get("builder_tool")
        if not build_supported or not builder_tool:
            return ManifestLookup(
                build_supported=False,
                builder_tool=builder_tool,
                builder_args=dict(entry.get("builder_args") or {}),
                entry=entry,
                error="Build is not supported for this manifest entry.",
            )
        return ManifestLookup(
            build_supported=True,
            builder_tool=str(builder_tool),
            builder_args=dict(entry.get("builder_args") or {}),
            entry=entry,
        )


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BuilderRegistryEntry:
    builder_tool: str
    builder_args_factory: Callable[[str, str], dict[str, Any]]


def _normalize_s_subject(value: str | int | None) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("subject_id is required.")
    return text if text.upper().startswith("S") else f"S{text}"


def _normalize_digit_id(value: str | int | None, *, width: int) -> str:
    text = "" if value is None else str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError("numeric id is required.")
    return digits.zfill(width)


def _normalize_icentia_segment(value: str | int | None) -> str:
    text = "00" if value is None else str(value).strip()
    if text.lower().startswith("s"):
        text = text[1:]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError("segment id is required.")
    return digits.zfill(2)


def _existing_path(candidates: Iterable[Path]) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No source file found. Checked: " + ", ".join(checked))


def _wesad_builder_args(subject_id: str, recording_id: str) -> dict[str, Any]:
    del recording_id
    subject = _normalize_s_subject(subject_id)
    path = _existing_path(
        [
            get_config().datasets.wesad_root / subject / f"{subject}.pkl",
            PROJECT_ROOT / "dataset" / "wesad_sample" / "WESAD" / subject / f"{subject}.pkl",
        ]
    )
    return {"pickle_path": str(path)}


def _ppg_dalia_builder_args(subject_id: str, recording_id: str) -> dict[str, Any]:
    del recording_id
    subject = _normalize_s_subject(subject_id)
    path = _existing_path(
        [
            get_config().datasets.ppg_dalia_root / subject / f"{subject}.pkl",
            PROJECT_ROOT / "dataset" / "ppg_dalia_sample" / "converted_original_like" / subject / f"{subject}.pkl",
        ]
    )
    return {"pickle_path": str(path)}


def _icentia11k_builder_args(subject_id: str, recording_id: str) -> dict[str, Any]:
    patient = _normalize_digit_id(subject_id, width=5)
    segment = _normalize_icentia_segment(recording_id)
    local_root = get_config().datasets.icentia11k_root
    record_stem = local_root / f"p{patient[:2]}" / f"p{patient}" / f"p{patient}_s{segment}"
    _existing_path([record_stem.with_suffix(".hea")])
    return {
        "record_id": str(record_stem),
        "sampling_rate": 250,
        "lead": "I",
    }


def _afppgecg_builder_args(subject_id: str, recording_id: str) -> dict[str, Any]:
    del recording_id
    patient = _normalize_digit_id(subject_id, width=3)
    dataset_root = get_config().datasets.afppgecg_root
    _existing_path([dataset_root / f"{patient}_ECG.mat"])
    _existing_path([dataset_root / f"{patient}_PPG.mat"])
    return {"patient_id": patient, "dataset_root": str(dataset_root)}


BUILDER_REGISTRY: dict[str, BuilderRegistryEntry] = {
    "wesad": BuilderRegistryEntry(
        "state_build_from_wesad_pickle",
        _wesad_builder_args,
    ),
    "ppg_dalia": BuilderRegistryEntry(
        "state_build_from_ppg_dalia_pickle",
        _ppg_dalia_builder_args,
    ),
    "icentia11k": BuilderRegistryEntry(
        "state_build_from_ecg_record",
        _icentia11k_builder_args,
    ),
    "afppgecg": BuilderRegistryEntry(
        "state_build_from_ppg_patient",
        _afppgecg_builder_args,
    ),
}


def lookup_builder_registry(
    dataset: str,
    subject_id: str | None,
    recording_id: str | None = None,
) -> ManifestLookup:
    entry = BUILDER_REGISTRY.get(str(dataset))
    subject = "" if subject_id is None else str(subject_id)
    recording = "" if recording_id is None else str(recording_id)
    if entry is None:
        return ManifestLookup(
            build_supported=False,
            error=f"No builder registry entry for dataset={dataset}.",
        )
    try:
        return ManifestLookup(
            build_supported=True,
            builder_tool=entry.builder_tool,
            builder_args=entry.builder_args_factory(subject, recording),
            entry={
                "dataset": str(dataset),
                "subject_id": subject,
                "recording_id": recording or subject,
                "builder_tool": entry.builder_tool,
                "build_supported": True,
                "source": "builder_registry",
            },
        )
    except Exception as exc:
        return ManifestLookup(
            build_supported=False,
            builder_tool=entry.builder_tool,
            error=(
                "Builder registry could not resolve source for "
                f"dataset={dataset}, subject_id={subject}, recording_id={recording}: {exc}"
            ),
        )


def is_builder_registry_supported_context(key: ContextKey) -> bool:
    return lookup_builder_registry(key.dataset, key.subject_id, key.recording_id).build_supported


_ACTIVE_MANIFEST: SourceManifest | None = None
_HIDE_LABEL_GENERATED_STATES = False


def set_active_source_manifest(manifest: SourceManifest | None) -> None:
    global _ACTIVE_MANIFEST
    _ACTIVE_MANIFEST = manifest


def set_label_generated_state_visibility_filter(enabled: bool) -> None:
    global _HIDE_LABEL_GENERATED_STATES
    _HIDE_LABEL_GENERATED_STATES = bool(enabled)


def effective_subject_id(state: MonitoringState) -> str:
    return str(
        state.subject_id
        or state.metadata.get("subject_id")
        or state.dataset_specific.get("subject_id")
        or state.patient_id
    )


def effective_recording_id(state: MonitoringState) -> str:
    return str(
        state.recording_id
        or state.metadata.get("recording_id")
        or state.metadata.get("record_id")
        or state.metadata.get("segment_id")
        or state.dataset_specific.get("recording_id")
        or state.dataset_specific.get("record_id")
        or state.dataset_specific.get("segment_id")
        or effective_subject_id(state)
    )


def context_key_for_state(state: MonitoringState) -> ContextKey:
    return ContextKey(
        dataset=str(state.dataset),
        subject_id=effective_subject_id(state),
        recording_id=effective_recording_id(state),
    )


def load_states_jsonl(path: str | Path) -> list[MonitoringState]:
    states: list[MonitoringState] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                states.append(MonitoringState.from_dict(json.loads(line)))
    return states


def load_qa_jsonl(path: str | Path) -> list[QASample]:
    return list(stream_qa_jsonl(path))


def group_states_by_context(states: Iterable[MonitoringState]) -> dict[ContextKey, list[MonitoringState]]:
    grouped: dict[ContextKey, list[MonitoringState]] = defaultdict(list)
    for state in states:
        grouped[context_key_for_state(state)].append(state)
    return dict(grouped)


def split_preload_contexts(
    states: list[MonitoringState],
    miss_ratio: float,
    seed: int,
    *,
    build_supported_contexts: set[ContextKey] | None = None,
) -> SplitResult:
    contexts = sorted(
        group_states_by_context(states),
        key=lambda key: (key.dataset, key.subject_id, key.recording_id),
    )
    if len(contexts) < 2:
        if contexts and build_supported_contexts and contexts[0] in build_supported_contexts and miss_ratio > 0:
            return SplitResult(set(), {contexts[0]}, True)
        return SplitResult(set(contexts), set(), True)

    rng = random.Random(seed)
    shuffled = contexts[:]
    rng.shuffle(shuffled)
    missing_count = round(len(shuffled) * float(miss_ratio))
    missing_count = max(1 if miss_ratio > 0 else 0, missing_count)
    missing_count = min(len(shuffled) - 1, missing_count)
    missing = set(shuffled[:missing_count])
    preloaded = set(shuffled[missing_count:])
    return SplitResult(preloaded, missing, False)


def load_preloaded_states(states: list[MonitoringState], preloaded_contexts: set[ContextKey]) -> int:
    store = get_global_state_store()
    store.clear()
    selected = [state for state in states if context_key_for_state(state) in preloaded_contexts]
    store.add_states(selected)
    return len(selected)


def filter_samples(
    samples: list[QASample],
    *,
    tier: str,
    dataset: str | None,
    question_type: str | None,
    max_samples: int | None,
    limit_per_target: int | None,
    seed: int,
) -> list[QASample]:
    dataset_filter = None if dataset is None else canonicalize_dataset_name(dataset)
    filtered = [
        sample
        for sample in samples
        if str(sample.gt_metadata.get("tier")).upper() == tier.upper()
        and (
            dataset_filter is None
            or sample.agent_input.window_locator.dataset == dataset_filter
        )
        and (
            question_type is None
            or str(sample.gt_metadata.get("question_type")) == question_type
        )
    ]
    rng = random.Random(seed)
    rng.shuffle(filtered)
    if limit_per_target is not None:
        counts: Counter[str] = Counter()
        limited: list[QASample] = []
        for sample in filtered:
            target = str(sample.gt_metadata.get("target"))
            if counts[target] >= limit_per_target:
                continue
            counts[target] += 1
            limited.append(sample)
        filtered = limited
    if max_samples is not None:
        filtered = filtered[:max_samples]
    return filtered


def score_prediction(eval_spec: EvalSpec, prediction: str) -> tuple[bool, str | None]:
    return evaluate_prediction(eval_spec, prediction)


def score_prediction_details(
    eval_spec: EvalSpec,
    prediction: str,
) -> tuple[bool, str | None, str | None]:
    return evaluate_prediction_details(eval_spec, prediction)


def raw_signal_observability(target: str | None) -> str:
    return RAW_SIGNAL_OBSERVABILITY_BY_TARGET.get(str(target), "unknown")


def normalize_token_usage(token_usage: dict[str, Any] | None) -> dict[str, int]:
    usage = zero_usage()
    if not isinstance(token_usage, dict):
        return usage
    for key in TOKEN_USAGE_KEYS:
        usage[key] = int(token_usage.get(key, 0) or 0)
    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def trace_prediction_fields(trace_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(trace_summary, dict):
        return {}
    return {
        f"trace_{key}": trace_summary.get(key)
        for key in TRACE_PREDICTION_SUMMARY_KEYS
    }


def model_cost_prediction_fields(agent_result: dict[str, Any]) -> dict[str, int]:
    return {key: int(agent_result.get(key, 0) or 0) for key in MODEL_COST_KEYS}


@register_tool(
    name=SAFE_BUILD_TOOL,
    description=(
        "Eval-safe builder wrapper for missing VitalBench contexts. Looks up "
        "source_manifest args and preserves already-loaded StateStore contents."
    ),
    parameters={
        "dataset": {"type": "string", "description": "Dataset name.", "required": True},
        "subject_id": {"type": "string", "description": "Subject ID.", "required": True},
        "recording_id": {"type": "string", "description": "Recording ID.", "required": False},
    },
)
def eval_state_build_missing_context(
    dataset: str,
    subject_id: str,
    recording_id: str | None = None,
) -> dict[str, Any]:
    if _ACTIVE_MANIFEST is None:
        return {"success": False, "error": "No active source manifest.", "error_class": "build_unavailable"}
    lookup = _ACTIVE_MANIFEST.lookup(dataset, subject_id, recording_id)
    if not lookup.build_supported or not lookup.builder_tool:
        return {
            "success": False,
            "error": lookup.error or "Build is unavailable for this context.",
            "error_class": "build_unavailable",
            "dataset": dataset,
            "subject_id": subject_id,
            "recording_id": recording_id,
        }
    builder_args = dict(lookup.builder_args or {})
    builder_args["clear_existing"] = False
    store = get_global_state_store()
    before_contexts = store.list_contexts()
    requested_recording = "" if recording_id is None else str(recording_id)
    for ctx in before_contexts:
        if ctx.get("dataset") != dataset:
            continue
        if str(ctx.get("subject_id") or "") != str(subject_id or ""):
            continue
        if requested_recording and str(ctx.get("recording_id") or "") != requested_recording:
            continue
        if int(ctx.get("state_count") or 0) <= 0:
            continue
        return {
            "success": True,
            "requested": {
                "dataset": dataset,
                "subject_id": subject_id,
                "recording_id": recording_id,
            },
            "cached": True,
            "builder_tool": lookup.builder_tool,
            "builder_args": builder_args,
            "build_success": True,
            "generated_state_count": 0,
            "hidden_label_generated_state_count": 0,
            "contexts": before_contexts,
            "builder_result": {"success": True, "cached": True},
            "error": "",
            "error_class": None,
        }

    before = sum(int(ctx.get("state_count") or 0) for ctx in before_contexts)
    result = call_tool(lookup.builder_tool, **builder_args)
    hidden_label_generated_state_count = 0
    if _HIDE_LABEL_GENERATED_STATES:
        hidden_label_generated_state_count = enforce_label_generated_state_visibility()
    after_contexts = store.list_contexts()
    after = sum(int(ctx.get("state_count") or 0) for ctx in after_contexts)
    return {
        "success": result.get("success") is True,
        "requested": {
            "dataset": dataset,
            "subject_id": subject_id,
            "recording_id": recording_id,
        },
        "builder_tool": lookup.builder_tool,
        "builder_args": builder_args,
        "build_success": result.get("success") is True,
        "generated_state_count": max(0, after - before),
        "hidden_label_generated_state_count": hidden_label_generated_state_count,
        "contexts": after_contexts,
        "builder_result": result,
        "error": result.get("error") or "",
        "error_class": None if result.get("success") is True else "tool_error",
    }


def _filtered_tools_prompt(tools: list[dict[str, Any]]) -> str:
    lines = ["Available tools:\n"]
    for info in tools:
        lines.append(f"- **{info['name']}**: {info['description']}")
        for pname, pinfo in (info.get("parameters") or {}).items():
            default = f" (default: {pinfo['default']})" if "default" in pinfo else ""
            lines.append(
                f"    - `{pname}` ({pinfo.get('type', 'any')}): "
                f"{pinfo.get('description', '')}{default}"
            )
    return "\n".join(lines)


def sanitize_raw_tool_result(value: Any) -> Any:
    """Remove known reference-label fields from raw-mode tool outputs."""
    if isinstance(value, dict):
        return {
            key: sanitize_raw_tool_result(item)
            for key, item in value.items()
            if key not in RAW_LABEL_LEAK_KEYS
        }
    if isinstance(value, list):
        return [sanitize_raw_tool_result(item) for item in value]
    return value


def _is_label_generated_metadata_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(marker in lowered for marker in LABEL_GENERATED_METADATA_KEY_SUBSTRINGS)


def is_label_generated_state(state: MonitoringState) -> bool:
    """Return true when a state exposes dataset label/annotation-derived facts."""

    if any(getattr(state, field_name) is not None for field_name in LABEL_GENERATED_STATE_FIELDS):
        return True
    for payload in (state.metadata, state.dataset_specific):
        for key in payload:
            if _is_label_generated_metadata_key(str(key)):
                return True
    return False


def project_label_generated_state(state: MonitoringState) -> MonitoringState:
    """Return an agent-visible state with label/annotation facts removed.

    Some benchmark states mix signal-derived measurements with dataset labels,
    e.g. PPG-DaLiA has BVP-derived HR plus activity-label metadata. State-mode
    should keep the signal facts while hiding label-generated fields.
    """

    return replace(
        state,
        **{field_name: None for field_name in LABEL_GENERATED_STATE_FIELDS},
        metadata={
            key: value
            for key, value in state.metadata.items()
            if not _is_label_generated_metadata_key(str(key))
        },
        dataset_specific={
            key: value
            for key, value in state.dataset_specific.items()
            if not _is_label_generated_metadata_key(str(key))
        },
    )


def project_label_generated_states(
    states: Iterable[MonitoringState],
) -> list[MonitoringState]:
    """Project states into the label-safe view exposed in state-mode eval."""

    return [project_label_generated_state(state) for state in states]


def _collect_global_state_store_states() -> list[MonitoringState]:
    store = get_global_state_store()
    seen: dict[str, MonitoringState] = {}
    for context in store.list_contexts():
        start_s = context.get("start_s")
        end_s = context.get("end_s")
        if start_s is None or end_s is None:
            continue
        states = store.get_window(
            context.get("dataset"),
            context.get("subject_id"),
            context.get("recording_id"),
            float(start_s),
            float(end_s),
        )
        for state in states:
            seen[state.state_id] = state
    return list(seen.values())


def enforce_label_generated_state_visibility() -> int:
    """Project label-generated facts out of the global store and return count."""

    states = _collect_global_state_store_states()
    projected_count = sum(1 for state in states if is_label_generated_state(state))
    if projected_count > 0:
        visible_states = project_label_generated_states(states)
        store = get_global_state_store()
        store.clear()
        store.add_states(visible_states)
    return projected_count


def _raw_tool_info(tool: dict[str, Any]) -> dict[str, Any]:
    """Return eval-safe tool metadata for the planner prompt."""
    info = dict(tool)
    name = str(info.get("name"))
    if name in RAW_TOOL_DESCRIPTION_OVERRIDES:
        info["description"] = RAW_TOOL_DESCRIPTION_OVERRIDES[name]
    parameters = dict(info.get("parameters") or {})
    for hidden_name in RAW_HIDDEN_PARAMETER_NAMES:
        parameters.pop(hidden_name, None)
    info["parameters"] = parameters
    return info


def _tool_parameter_names(name: str, tools: list[dict[str, Any]] | None = None) -> set[str] | None:
    for tool in tools if tools is not None else list_tools():
        if tool.get("name") == name:
            return set((tool.get("parameters") or {}).keys())
    return None


def _filter_known_tool_kwargs(
    name: str,
    kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    allowed = _tool_parameter_names(name, tools)
    if allowed is None:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in allowed}


def _compact_dataclass_dict(value: Any) -> dict[str, Any]:
    return {key: item for key, item in asdict(value).items() if item is not None}


def _plan_trace_payload(plan: Any) -> dict[str, Any]:
    if hasattr(plan, "model_dump"):
        try:
            plan_payload = plan.model_dump(mode="json")
        except TypeError:
            plan_payload = _truncate_for_trace(plan.model_dump(), 1_000_000)
    else:
        plan_payload = _truncate_for_trace(asdict(plan), 1_000_000)
    steps = plan_payload.get("steps") or []
    return {
        "intent": str(plan_payload.get("intent")),
        "reasoning": plan_payload.get("reasoning"),
        "step_count": len(steps),
        "tool_names": _plan_tool_names(plan_payload),
        "plan_signature": _plan_signature(plan_payload),
        "plan": plan_payload,
    }


@contextlib.contextmanager
def traced_reactive_planner(trace: EvalTraceRecorder | None) -> Iterator[None]:
    """Record planner and re-planner calls made inside ReactivePipeline."""
    if trace is None:
        yield
        return

    import agent.reactive.planner as planner_mod

    original_create_plan = planner_mod.Planner.create_plan
    original_replan = planner_mod.Planner.replan

    def traced_create_plan(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace.current_attempt = 0
        trace.add(
            "planner_start",
            {"model": getattr(self, "model", None)},
            stdout=f"planner start model={getattr(self, 'model', None)}",
        )
        started = time.monotonic()
        try:
            plan, usage = original_create_plan(self, *args, **kwargs)
        except Exception as exc:
            trace.add(
                "planner_error",
                {"error": str(exc)},
                duration_sec=time.monotonic() - started,
                stdout=f"planner error elapsed={time.monotonic() - started:.1f}s error={exc}",
            )
            raise
        duration = time.monotonic() - started
        payload = _plan_trace_payload(plan)
        payload["usage"] = usage
        trace.add(
            "planner_done",
            payload,
            duration_sec=duration,
            stdout=(
                f"planner done elapsed={duration:.1f}s "
                f"steps={payload['step_count']} tools={payload['tool_names']}"
            ),
        )
        return plan, usage

    def traced_replan(self: Any, *args: Any, **kwargs: Any) -> Any:
        from_attempt = trace.current_attempt
        previous_plan = kwargs.get("previous_plan")
        if previous_plan is None and args:
            previous_plan = args[0]
        issues = kwargs.get("issues")
        if issues is None and len(args) > 1:
            issues = args[1]
        trace.add(
            "replan_start",
            {"model": getattr(self, "model", None), "from_attempt": from_attempt},
            stdout=f"replan start model={getattr(self, 'model', None)}",
        )
        started = time.monotonic()
        try:
            plan, usage = original_replan(self, *args, **kwargs)
        except Exception as exc:
            trace.add(
                "replan_error",
                {"error": str(exc)},
                duration_sec=time.monotonic() - started,
                stdout=f"replan error elapsed={time.monotonic() - started:.1f}s error={exc}",
            )
            raise
        duration = time.monotonic() - started
        to_attempt = from_attempt + 1
        trace.current_attempt = to_attempt
        previous_payload = _plan_trace_payload(previous_plan) if previous_plan is not None else {}
        previous_tool_names = [
            str(name)
            for name in previous_payload.get("tool_names", [])
            if name
        ]
        payload = _plan_trace_payload(plan)
        payload["usage"] = usage
        new_tool_names = [
            str(name)
            for name in payload.get("tool_names", [])
            if name
        ]
        payload.update(
            {
                "attempt": to_attempt,
                "from_attempt": from_attempt,
                "to_attempt": to_attempt,
                "previous_tool_names": previous_tool_names,
                "new_tool_names": new_tool_names,
                "added_tools": [
                    name for name in new_tool_names if name not in previous_tool_names
                ],
                "removed_tools": [
                    name for name in previous_tool_names if name not in new_tool_names
                ],
                "issues_used_for_replan": [
                    str(issue) for issue in (issues or [])
                ],
            }
        )
        trace.add(
            "replan_done",
            payload,
            duration_sec=duration,
            stdout=(
                f"replan done elapsed={duration:.1f}s "
                f"steps={payload['step_count']} tools={payload['tool_names']}"
            ),
        )
        return plan, usage

    planner_mod.Planner.create_plan = traced_create_plan
    planner_mod.Planner.replan = traced_replan
    try:
        yield
    finally:
        planner_mod.Planner.create_plan = original_create_plan
        planner_mod.Planner.replan = original_replan


@contextlib.contextmanager
def traced_validation_gate(trace: EvalTraceRecorder | None) -> Iterator[None]:
    """Record validation outcomes made inside ReactivePipeline."""
    if trace is None:
        yield
        return

    import agent.reactive.validation as validation_mod

    original_validate = validation_mod.ValidationGate.validate

    def traced_validate(self: Any, *args: Any, **kwargs: Any) -> Any:
        plan = kwargs.get("plan")
        if plan is None and args:
            plan = args[0]
        started = time.monotonic()
        try:
            result = original_validate(self, *args, **kwargs)
        except Exception as exc:
            trace.add(
                "validation_error",
                {"error": str(exc)},
                duration_sec=time.monotonic() - started,
                stdout=f"validation error elapsed={time.monotonic() - started:.1f}s error={exc}",
            )
            raise

        issues = list(getattr(result, "issues", []) or [])
        tool_names = [
            str(step.tool_name)
            for step in getattr(plan, "steps", []) or []
            if getattr(step, "tool_name", None)
        ]
        critical_issues = list(getattr(result, "critical_issues", []) or [])
        warnings = list(getattr(result, "warnings", []) or [])
        trace.add(
            "validation_done",
            {
                "passed": bool(getattr(result, "passed", False)),
                "coverage": float(getattr(result, "coverage", 0.0) or 0.0),
                "has_critical": bool(getattr(result, "has_critical", False)),
                "issue_count": len(issues),
                "critical_issue_count": len(critical_issues),
                "warning_issue_count": len(warnings),
                "issue_messages": [
                    str(getattr(issue, "message", "")) for issue in issues
                ],
                "issue_types": [
                    _enum_value(getattr(issue, "issue_type", ""))
                    for issue in issues
                ],
                "issue_severities": [
                    _enum_value(getattr(issue, "severity", ""))
                    for issue in issues
                ],
                "tool_names": tool_names,
                "issue_tool_names": [
                    str(getattr(issue, "tool_name", ""))
                    for issue in issues
                    if getattr(issue, "tool_name", "")
                ],
            },
            duration_sec=time.monotonic() - started,
            stdout=(
                "validation done "
                f"passed={bool(getattr(result, 'passed', False))} "
                f"issues={len(issues)}"
            ),
        )
        return result

    validation_mod.ValidationGate.validate = traced_validate
    try:
        yield
    finally:
        validation_mod.ValidationGate.validate = original_validate


def _trace_tool_call(
    trace: EvalTraceRecorder | None,
    *,
    name: str,
    args: dict[str, Any],
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if trace is None:
        return call()

    trace.add(
        "tool_start",
        {
            "tool_name": name,
            "tool_args": _truncate_for_trace(args, trace.result_max_chars),
        },
        stdout=f"tool start {name} args={json.dumps(_truncate_for_trace(args, 300), ensure_ascii=False, default=str)}",
    )
    started = time.monotonic()
    try:
        result = call()
    except Exception as exc:
        duration = time.monotonic() - started
        trace.add(
            "tool_error",
            {
                "tool_name": name,
                "tool_args": _truncate_for_trace(args, trace.result_max_chars),
                "error": str(exc),
            },
            duration_sec=duration,
            stdout=f"tool error {name} elapsed={duration:.1f}s error={exc}",
        )
        raise

    duration = time.monotonic() - started
    success = result.get("success")
    error = result.get("error")
    result_chars = _json_size(result)
    trace.add(
        "tool_done",
        {
            "tool_name": name,
            "tool_args": _truncate_for_trace(args, trace.result_max_chars),
            "success": success,
            "error": error,
            "result_chars": result_chars,
            "result": _truncate_for_trace(result, trace.result_max_chars),
        },
        duration_sec=duration,
        stdout=(
            f"tool done {name} success={success} "
            f"elapsed={duration:.1f}s result_chars={result_chars}"
            + (f" error={error}" if error else "")
        ),
    )
    return result


def _perturbed_tool_result(
    result: dict[str, Any],
    *,
    tool_name: str,
    mode: str,
) -> dict[str, Any]:
    if mode == "tool_failure":
        return {
            "success": False,
            "error": f"injected: tool_failure for {tool_name}",
        }

    perturbed = dict(result)
    perturbed["success"] = True
    required_fields = ValidationGate.REQUIRED_FIELDS.get(tool_name, [])
    for field_name in required_fields:
        perturbed[field_name] = None
    if not required_fields and "data" in perturbed:
        perturbed["data"] = None
    return perturbed


def _maybe_perturb_tool_result(
    result: dict[str, Any],
    *,
    trace: EvalTraceRecorder | None,
    perturbation: PerturbationConfig | None,
    question_id: str,
    tool_name: str,
    occurrence_index: int,
) -> dict[str, Any]:
    if perturbation is None or not perturbation.should_inject(
        question_id=question_id,
        tool_name=tool_name,
        occurrence_index=occurrence_index,
    ):
        return result

    perturbed = _perturbed_tool_result(
        result,
        tool_name=tool_name,
        mode=perturbation.mode,
    )
    if trace is not None:
        trace.add(
            "tool_perturbation_injected",
            {
                "tool_name": tool_name,
                "occurrence_index": occurrence_index,
                "mode": perturbation.mode,
                "target": perturbation.target,
                "rate": perturbation.rate,
                "seed": perturbation.seed,
                "original_success": result.get("success"),
                "perturbed_success": perturbed.get("success"),
            },
            stdout=(
                "perturbation injected "
                f"tool={tool_name} occurrence={occurrence_index} "
                f"mode={perturbation.mode}"
            ),
        )
    return perturbed


def _trace_perturbable_tool_call(
    trace: EvalTraceRecorder | None,
    *,
    perturbation: PerturbationConfig | None,
    question_id: str,
    call_counts: Counter[str],
    name: str,
    args: dict[str, Any],
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    occurrence_index = call_counts[name]
    call_counts[name] += 1

    def call_with_perturbation() -> dict[str, Any]:
        result = call()
        return _maybe_perturb_tool_result(
            result,
            trace=trace,
            perturbation=perturbation,
            question_id=question_id,
            tool_name=name,
            occurrence_index=occurrence_index,
        )

    return _trace_tool_call(
        trace,
        name=name,
        args=args,
        call=call_with_perturbation,
    )


@contextlib.contextmanager
def restricted_mhealth_tool_pool(
    trace: EvalTraceRecorder | None = None,
    *,
    perturbation: PerturbationConfig | None = None,
    question_id: str | None = None,
) -> Iterator[None]:
    """Expose state query tools plus the eval-safe builder wrapper during eval (state mode)."""
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod
    import agent.tools.registry as registry_mod

    original_planner_list = planner_mod.list_tools
    original_planner_prompt = planner_mod.get_tools_prompt
    original_ecg_policy = planner_mod._apply_ecg_diagnosis_policy
    original_pipeline_call = pipeline_mod.call_tool
    call_counts: Counter[str] = Counter()
    sample_question_id = question_id or (trace.question_id if trace is not None else "")

    def allowed_tools() -> list[dict[str, Any]]:
        return [
            tool
            for tool in list_tools()
            if tool["name"] in LOCATOR_STATE_TOOL_NAMES
            or tool["name"] in {"state_list_contexts", SAFE_BUILD_TOOL}
        ]

    def guarded_call(name: str, **kwargs: Any) -> dict[str, Any]:
        allowed_names = {tool["name"] for tool in allowed_tools()}
        effective_args = dict(kwargs)
        if name not in allowed_names:
            return _trace_tool_call(
                trace,
                name=name,
                args=effective_args,
                call=lambda: {
                    "success": False,
                    "error": f"Tool '{name}' is disabled during VitalBench eval.",
                },
            )
        if name == "state_list_contexts":
            effective_args = {}
            return _trace_perturbable_tool_call(
                trace,
                perturbation=perturbation,
                question_id=sample_question_id,
                call_counts=call_counts,
                name=name,
                args=effective_args,
                call=lambda: registry_mod.call_tool(name),
            )
        if name.startswith("state_build_from_"):
            return _trace_tool_call(
                trace,
                name=name,
                args=effective_args,
                call=lambda: {
                    "success": False,
                    "error": (
                        "Direct state_build_from_* tools are disabled in eval; "
                        "use eval_state_build_missing_context."
                    ),
                },
            )
        effective_args = _filter_known_tool_kwargs(name, kwargs, allowed_tools())
        return _trace_perturbable_tool_call(
            trace,
            perturbation=perturbation,
            question_id=sample_question_id,
            call_counts=call_counts,
            name=name,
            args=effective_args,
            call=lambda: registry_mod.call_tool(name, **effective_args),
        )

    planner_mod.list_tools = allowed_tools
    planner_mod.get_tools_prompt = lambda: _filtered_tools_prompt(allowed_tools())
    planner_mod._apply_ecg_diagnosis_policy = lambda plan, user_query, active_record_id=None: plan
    pipeline_mod.call_tool = guarded_call
    try:
        yield
    finally:
        planner_mod.list_tools = original_planner_list
        planner_mod.get_tools_prompt = original_planner_prompt
        planner_mod._apply_ecg_diagnosis_policy = original_ecg_policy
        pipeline_mod.call_tool = original_pipeline_call


@contextlib.contextmanager
def restricted_signal_tool_pool(
    trace: EvalTraceRecorder | None = None,
    dataset: str | None = None,
    adaptive_scope_enabled: bool = False,
    perturbation: PerturbationConfig | None = None,
    question_id: str | None = None,
) -> Iterator[None]:
    """Expose sanitized raw-signal analysis tools during eval."""
    import agent.reactive.pipeline as pipeline_mod
    import agent.reactive.planner as planner_mod
    import agent.tools.registry as registry_mod

    original_planner_list = planner_mod.list_tools
    original_planner_prompt = planner_mod.get_tools_prompt
    original_ecg_policy = planner_mod._apply_ecg_diagnosis_policy
    original_pipeline_call = pipeline_mod.call_tool
    original_previous_window_compare = pipeline_mod.compare_with_previous_window
    call_counts: Counter[str] = Counter()
    sample_question_id = question_id or (trace.question_id if trace is not None else "")
    allowed_tool_names = effective_raw_tool_names(
        dataset,
        adaptive_scope_enabled=adaptive_scope_enabled,
    )

    def allowed_tools() -> list[dict[str, Any]]:
        return [
            _raw_tool_info(tool)
            for tool in list_tools()
            if tool["name"] in allowed_tool_names
        ]

    def guarded_call(name: str, **kwargs: Any) -> dict[str, Any]:
        effective_args = dict(kwargs)
        if name not in allowed_tool_names:
            return _trace_tool_call(
                trace,
                name=name,
                args=effective_args,
                call=lambda: {
                    "success": False,
                    "error": (
                        f"Tool '{name}' is disabled during VitalBench raw-data eval"
                        + (f" for dataset '{dataset}'." if dataset else ".")
                    ),
                },
            )
        effective_args = _filter_known_tool_kwargs(name, kwargs, allowed_tools())
        result = _trace_perturbable_tool_call(
            trace,
            perturbation=perturbation,
            question_id=sample_question_id,
            call_counts=call_counts,
            name=name,
            args=effective_args,
            call=lambda: sanitize_raw_tool_result(
                registry_mod.call_tool(name, **effective_args)
            ),
        )
        return result

    def orchestrated_previous_window_compare(**kwargs: Any) -> dict[str, Any]:
        if trace is not None:
            trace.add(
                "orchestration_start",
                {
                    "orchestration": PREVIOUS_WINDOW_COMPARISON_RESULT,
                    "args": _truncate_for_trace(kwargs, trace.result_max_chars),
                },
                stdout="orchestration start previous_window_comparison",
            )
        started = time.monotonic()
        result = sanitize_raw_tool_result(original_previous_window_compare(**kwargs))
        if trace is not None:
            trace.add(
                "orchestration_done",
                {
                    "orchestration": PREVIOUS_WINDOW_COMPARISON_RESULT,
                    "success": result.get("success"),
                    "error": result.get("error"),
                    "result": _truncate_for_trace(result, trace.result_max_chars),
                },
                duration_sec=time.monotonic() - started,
                stdout=(
                    "orchestration done previous_window_comparison "
                    f"success={result.get('success')}"
                ),
            )
        return result

    planner_mod.list_tools = allowed_tools
    planner_mod.get_tools_prompt = lambda: _filtered_tools_prompt(allowed_tools())
    planner_mod._apply_ecg_diagnosis_policy = lambda plan, user_query, active_record_id=None: plan
    pipeline_mod.call_tool = guarded_call
    pipeline_mod.compare_with_previous_window = orchestrated_previous_window_compare
    try:
        yield
    finally:
        planner_mod.list_tools = original_planner_list
        planner_mod.get_tools_prompt = original_planner_prompt
        planner_mod._apply_ecg_diagnosis_policy = original_ecg_policy
        pipeline_mod.call_tool = original_pipeline_call
        pipeline_mod.compare_with_previous_window = original_previous_window_compare


def _adaptive_scope_enabled(
    *,
    dataset: str,
    tier: str | None,
    target: str | None,
) -> bool:
    return (
        str(dataset) == "afppgecg"
        and str(tier or "").upper() == "B"
        and str(target or "") in ADAPTIVE_SCOPE_TARGETS
    )


def _agent_input_with_observation_scope_note(
    agent_input: AgentInput,
    *,
    adaptive_scope_enabled: bool,
) -> AgentInput:
    if adaptive_scope_enabled:
        note = (
            "\n\nObservation scope policy: this is an aggregate rhythm/AF question. "
            "Use the locator as the anchor window, but you may choose a broader "
            "sampled-recording observation scope when a tool explicitly supports it. "
            "Do not manually invent timestamps; use tool scope parameters."
        )
    else:
        note = (
            "\n\nObservation scope policy: use the locator window only. Do not use "
            "recording-wide or expanded-scope tools for current or Tier A style questions."
        )
    if note in agent_input.question:
        return agent_input
    return replace(agent_input, question=f"{agent_input.question}{note}")


def infer_actual_path(tool_results: list[dict[str, Any]]) -> tuple[str, list[str], bool | None, bool | None]:
    counted_results = [
        item
        for item in tool_results
        if item.get("tool_name") != PREVIOUS_WINDOW_COMPARISON_RESULT
    ]
    names = [str(item.get("tool_name")) for item in counted_results]
    # Raw-data mode: check if any signal analysis tool was called successfully.
    signal_tool_names = RAW_SIGNAL_TOOL_NAMES | ADAPTIVE_SCOPE_TOOL_NAMES
    signal_results = [
        item for item in counted_results if item.get("tool_name") in signal_tool_names
    ]
    if signal_results:
        signal_success = any(item.get("success") is True for item in signal_results)
        return ("signal_query" if signal_success else "failure"), names, None, signal_success
    # State-mode: check build / query paths.
    build_results = [
        item for item in counted_results if item.get("tool_name") == SAFE_BUILD_TOOL
    ]
    query_results = [
        item
        for item in counted_results
        if str(item.get("tool_name", "")).startswith(QUERY_TOOL_PREFIX)
    ]
    build_success = None if not build_results else any(item.get("success") is True for item in build_results)
    query_success = None if not query_results else any(item.get("success") is True for item in query_results)
    if query_results and query_success is True:
        if build_results and build_success is True:
            return "build_then_query", names, build_success, query_success
        return "query", names, build_success, query_success
    if build_results and build_success is True:
        return "build", names, build_success, query_success
    if build_results or query_results:
        return "failure", names, build_success, query_success
    return "failure", names, build_success, query_success


def sample_context_key(
    sample: QASample,
    states_by_id: dict[str, MonitoringState],
) -> ContextKey:
    source_state_id = sample.gt_metadata.get("source_state_id")
    state = states_by_id.get(str(source_state_id)) if source_state_id is not None else None
    if state is not None:
        return context_key_for_state(state)
    evidence = sample.gt_metadata.get("evidence") or {}
    locator = sample.agent_input.window_locator
    subject_id = str(evidence.get("subject_id") or locator.patient_id)
    source_state_id = str(sample.gt_metadata.get("source_state_id") or "")
    icentia_segment = _ICENTIA_SEGMENT_RE.search(source_state_id)
    recording_id = str(
        evidence.get("recording_id")
        or evidence.get("record_id")
        or getattr(locator, "recording_id", None)
        or (icentia_segment.group(1).zfill(2) if icentia_segment else None)
        or subject_id
    )
    return ContextKey(str(locator.dataset), subject_id, recording_id)


def run_agent_sample(
    agent_input: AgentInput,
    *,
    dry_run: bool,
    data_mode: str = "raw",
    trace: EvalTraceRecorder | None = None,
    benchmark_tier: str | None = None,
    benchmark_target: str | None = None,
    evaluation_protocol: str = "rebuttal_v2",
    no_validation_baseline: bool = False,
    no_replan_baseline: bool = False,
    no_planner_baseline: bool = False,
    no_planner_implementation: str = "rebuttal_v2",
    no_planner_selection_mode: str = "random",
    no_planner_seed: int = 42,
    no_planner_tool_count: int = 1,
    no_planner_tool_count_max: int | None = None,
    perturbation: PerturbationConfig | None = None,
) -> dict[str, Any]:
    if dry_run:
        if trace is not None:
            trace.add("dry_run", stdout="agent dry-run skipped")
        return {
            "prediction": "[DRY RUN] agent not executed",
            "raw_response": "",
            "tool_results": [],
            "token_usage": zero_usage(),
            "actual_path": "failure",
            "actual_tools_called": [],
            "build_success": None,
            "query_success": None,
            "error_class": "unknown",
            "error_message": None,
            **ModelCallTracker().to_fields(),
            "trace_summary": trace.summarize() if trace is not None else None,
        }

    from agent.reactive.pipeline import ReactivePipeline

    final = None
    error_message = None
    raw_dataset = str(agent_input.window_locator.dataset)
    adaptive_scope_enabled = _adaptive_scope_enabled(
        dataset=raw_dataset,
        tier=benchmark_tier,
        target=benchmark_target,
    )
    effective_agent_input = (
        _agent_input_with_observation_scope_note(
            agent_input,
            adaptive_scope_enabled=adaptive_scope_enabled,
        )
        if data_mode == "raw"
        else agent_input
    )
    answer_started: float | None = None
    answer_first_token_recorded = False
    model_call_tracker = ModelCallTracker()
    try:
        if trace is not None:
            trace.add(
                "agent_start",
                {
                    "data_mode": data_mode,
                    "dataset": raw_dataset,
                    "benchmark_tier": benchmark_tier,
                    "benchmark_target": benchmark_target,
                    "evaluation_protocol": evaluation_protocol,
                    "adaptive_scope_enabled": adaptive_scope_enabled,
                    "no_validation_baseline": no_validation_baseline,
                    "no_replan_baseline": no_replan_baseline,
                    "no_planner_baseline": no_planner_baseline,
                    "no_planner_implementation": no_planner_implementation,
                    "no_planner_selection_mode": no_planner_selection_mode,
                    "no_planner_seed": no_planner_seed,
                    "no_planner_tool_count": no_planner_tool_count,
                    "no_planner_tool_count_max": no_planner_tool_count_max,
                    "perturbation": (
                        None if perturbation is None else perturbation.to_dict()
                    ),
                },
                stdout=(
                    f"agent start data_mode={data_mode} dataset={raw_dataset} "
                    f"no_validation_baseline={no_validation_baseline} "
                    f"no_replan_baseline={no_replan_baseline} "
                    f"no_planner_baseline={no_planner_baseline}"
                ),
            )
        if data_mode == "raw":
            tool_pool_context = restricted_signal_tool_pool(
                trace,
                dataset=raw_dataset,
                adaptive_scope_enabled=adaptive_scope_enabled,
                perturbation=perturbation,
                question_id=agent_input.question_id,
            )
        else:
            tool_pool_context = restricted_mhealth_tool_pool(
                trace,
                perturbation=perturbation,
                question_id=agent_input.question_id,
            )
        with traced_reactive_planner(trace), traced_validation_gate(trace), tool_pool_context:
            pipeline = ReactivePipeline()
            if evaluation_protocol == "paper_v1":
                from agent.evaluation.reactive.protocols import PaperV1ValidationGate

                pipeline.validator = PaperV1ValidationGate()
            if getattr(pipeline, "llm_service", None) is not None:
                attach_model_call_tracker(pipeline.llm_service, model_call_tracker)
            if no_validation_baseline:
                pipeline.disable_validation = True
                if trace is not None:
                    trace.add(
                        "no_validation_baseline_config",
                        stdout="no-validation baseline enabled",
                    )
            if no_replan_baseline:
                pipeline.disable_validation = False
                pipeline.max_retries = 0
                if trace is not None:
                    trace.add(
                        "no_replan_baseline_config",
                        stdout="no-replan baseline enabled",
                    )
            if no_planner_baseline:
                if no_planner_implementation == "paper_v1":
                    from agent.evaluation.reactive.protocols import (
                        PaperV1NoPlannerPlanner,
                    )

                    pipeline.planner = PaperV1NoPlannerPlanner(
                        llm_service=pipeline.llm_service,
                        seed=no_planner_seed,
                        sample_key=effective_agent_input.question_id,
                        tool_count=no_planner_tool_count,
                        tool_count_max=no_planner_tool_count_max,
                        tool_pool_names=(
                            RAW_SIGNAL_TOOL_NAMES if data_mode == "raw" else None
                        ),
                    )
                else:
                    from agent.evaluation.reactive.ablation_no_planner import (
                        NoPlannerPlanner,
                    )

                    pipeline.planner = NoPlannerPlanner(
                        llm_service=pipeline.llm_service,
                        seed=no_planner_seed,
                        sample_key=effective_agent_input.question_id,
                        tool_count=no_planner_tool_count,
                        tool_count_max=no_planner_tool_count_max,
                        selection_mode=no_planner_selection_mode,
                        canonical_context=pipeline._active_context_from_agent_input(
                            effective_agent_input
                        ),
                        tool_pool_names=(
                            effective_raw_tool_names(
                                raw_dataset,
                                adaptive_scope_enabled=adaptive_scope_enabled,
                            )
                            if data_mode == "raw"
                            else None
                        ),
                    )
                pipeline.disable_validation = True
                pipeline.max_retries = 0
                if trace is not None:
                    trace.add(
                        "no_planner_baseline_config",
                        {
                            "seed": no_planner_seed,
                            "tool_count": no_planner_tool_count,
                            "tool_count_max": no_planner_tool_count_max,
                            "implementation": no_planner_implementation,
                            "selection_mode": no_planner_selection_mode,
                        },
                        stdout=(
                            "no-planner tool strategy enabled "
                            f"implementation={no_planner_implementation} "
                            f"mode={no_planner_selection_mode} "
                            f"seed={no_planner_seed} count={no_planner_tool_count}"
                            + (
                                ""
                                if no_planner_tool_count_max is None
                                else f"..{no_planner_tool_count_max}"
                            )
                        ),
                    )
            for event in pipeline.run_agent_input(effective_agent_input):
                event_context = event.context_used or {}
                is_status = event_context.get("event") == "status"
                if trace is not None and is_status:
                    trace.add(
                        "status",
                        {"message": event.answer},
                        stdout=f"status {event.answer}",
                    )
                    if "Generating answer" in event.answer:
                        answer_started = time.monotonic()
                        trace.add("answer_start", stdout="answer generation start")
                elif trace is not None and event.answer:
                    if answer_started is None:
                        answer_started = time.monotonic()
                        trace.add("answer_start", stdout="answer generation start")
                    if not answer_first_token_recorded:
                        answer_first_token_recorded = True
                        trace.add(
                            "answer_first_chunk",
                            {"chars": len(event.answer)},
                            stdout=f"answer first chunk chars={len(event.answer)}",
                        )
                # Only treat non-status events as candidate final answers.
                # Status events ("✅ Generating answer...", "🧪 Validation: ...")
                # would otherwise leak into the prediction field when an API
                # timeout cuts the stream off before any answer chunk arrives.
                if not is_status:
                    final = event
    except Exception as exc:
        error_message = str(exc)
        if trace is not None:
            trace.add("agent_error", {"error": error_message}, stdout=f"agent error {error_message}")

    if final is None:
        if trace is not None:
            trace.add("agent_done", {"success": False}, stdout="agent done success=False")
        return {
            "prediction": "",
            "raw_response": "",
            "tool_results": [],
            "token_usage": zero_usage(),
            "actual_path": "failure",
            "actual_tools_called": [],
            "build_success": None,
            "query_success": None,
            "error_class": "planner_issue" if error_message else "unknown",
            "error_message": error_message,
            **model_call_tracker.to_fields(),
            "trace_summary": trace.summarize() if trace is not None else None,
        }

    tool_results = list(final.context_used.get("tool_results") or [])
    token_usage = normalize_token_usage(final.context_used.get("token_usage"))
    actual_path, names, build_success, query_success = infer_actual_path(tool_results)
    if no_planner_baseline and no_planner_selection_mode == "none":
        actual_path = "direct_answer"
    if trace is not None:
        answer_duration = None if answer_started is None else time.monotonic() - answer_started
        trace.add(
            "answer_done",
            {
                "chars": len(final.answer),
                "answer_duration_sec": None if answer_duration is None else round(answer_duration, 6),
            },
            stdout=(
                f"answer done chars={len(final.answer)}"
                + ("" if answer_duration is None else f" elapsed={answer_duration:.1f}s")
            ),
        )
        trace.add(
            "agent_done",
            {
                "success": True,
                "actual_path": actual_path,
                "tool_names": names,
            },
            stdout=f"agent done actual_path={actual_path} tools={names}",
        )
    return {
        "prediction": final.answer,
        "raw_response": final.model_dump(),
        "tool_results": tool_results,
        "token_usage": token_usage,
        "actual_path": actual_path,
        "actual_tools_called": names,
        "build_success": build_success,
        "query_success": query_success,
        "error_class": None,
        "error_message": error_message,
        **model_call_tracker.to_fields(),
        "trace_summary": trace.summarize() if trace is not None else None,
    }


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _sec(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}s"


def _accuracy(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return sum(1 for record in records if record.get("score") is True) / len(records)


def aggregate_predictions(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [record for record in records if record["condition"] == condition]
    aggregate: dict[str, Any] = {
        "sample_count": len(subset),
        "accuracy": _accuracy(subset) or 0.0,
        "accuracy_by_question_type": {},
        "accuracy_by_target": {},
    }
    for key in TOKEN_USAGE_KEYS:
        aggregate[f"{key}_total"] = sum(int(record.get(key, 0) or 0) for record in subset)
    if subset:
        aggregate["avg_total_tokens"] = aggregate["total_tokens_total"] / len(subset)
    else:
        aggregate["avg_total_tokens"] = 0.0
    for key, output_name in [
        ("question_type", "accuracy_by_question_type"),
        ("target", "accuracy_by_target"),
    ]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in subset:
            groups[str(record.get(key))].append(record)
        aggregate[output_name] = {name: _accuracy(items) for name, items in sorted(groups.items())}

    if condition in AGENT_LIKE_CONDITIONS:
        routing = Counter(str(record.get("actual_path") or "failure") for record in subset)
        aggregate["routing_distribution"] = {
            "query": routing.get("query", 0),
            "build": routing.get("build", 0),
            "build_then_query": routing.get("build_then_query", 0),
            "signal_query": routing.get("signal_query", 0),
            "direct_answer": routing.get("direct_answer", 0),
            "failure": routing.get("failure", 0),
        }
        aggregate["accuracy_by_actual_path"] = {
            path: _accuracy([record for record in subset if record.get("actual_path") == path])
            for path in [
                "query",
                "build",
                "build_then_query",
                "signal_query",
                "direct_answer",
                "failure",
            ]
        }
        aggregate["accuracy_by_expected_path"] = {
            path: _accuracy([record for record in subset if record.get("expected_path") == path])
            for path in ["query", "build", "raw_signal"]
        }
    return aggregate


def aggregate_trace_performance(trace_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace_records:
        return {}

    slowest_samples = sorted(
        [
            {
                "question_id": record["question_id"],
                "duration_sec": record.get("summary", {}).get("duration_sec"),
                "planner_duration_sec": record.get("summary", {}).get("planner_duration_sec"),
                "tool_duration_sec": record.get("summary", {}).get("tool_duration_sec"),
                "answer_duration_sec": record.get("summary", {}).get("answer_duration_sec"),
                "tool_calls": record.get("summary", {}).get("tool_calls", []),
            }
            for record in trace_records
        ],
        key=lambda item: float(item.get("duration_sec") or 0.0),
        reverse=True,
    )[:10]

    by_tool: dict[str, dict[str, Any]] = {}
    for record in trace_records:
        for event in record.get("events", []):
            if event.get("event") != "tool_done":
                continue
            name = str(event.get("tool_name"))
            bucket = by_tool.setdefault(
                name,
                {
                    "count": 0,
                    "success_count": 0,
                    "total_duration_sec": 0.0,
                    "max_duration_sec": 0.0,
                    "total_result_chars": 0,
                },
            )
            duration = float(event.get("duration_sec") or 0.0)
            bucket["count"] += 1
            bucket["success_count"] += 1 if event.get("success") is True else 0
            bucket["total_duration_sec"] += duration
            bucket["max_duration_sec"] = max(bucket["max_duration_sec"], duration)
            bucket["total_result_chars"] += int(event.get("result_chars") or 0)

    for bucket in by_tool.values():
        count = max(1, int(bucket["count"]))
        bucket["avg_duration_sec"] = round(bucket["total_duration_sec"] / count, 6)
        bucket["total_duration_sec"] = round(bucket["total_duration_sec"], 6)
        bucket["max_duration_sec"] = round(bucket["max_duration_sec"], 6)
        bucket["avg_result_chars"] = round(bucket["total_result_chars"] / count, 2)

    return {
        "sample_count": len(trace_records),
        "slowest_samples": slowest_samples,
        "tools": dict(sorted(by_tool.items())),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


ERROR_ANALYSIS_FIELDNAMES = [
    "condition",
    "question_id",
    "dataset",
    "question_type",
    "target",
    "expected_path",
    "actual_path",
    "score",
    "error_attribution",
    "error_attribution_note",
    "actual_tools_called",
    "gold_answer",
    "prediction_preview",
    "error_message",
    "elapsed_sec",
]


def _tool_results_from_prediction(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw = record.get("raw_response")
    if not isinstance(raw, dict):
        return []
    context = raw.get("context_used")
    if not isinstance(context, dict):
        return []
    tool_results = context.get("tool_results")
    return [item for item in tool_results if isinstance(item, dict)] if isinstance(tool_results, list) else []


def classify_vitalbench_failure(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("score") is True and not record.get("error_message"):
        return "", ""

    if record.get("error_message"):
        return "runtime_error", str(record.get("error_message"))

    tool_results = _tool_results_from_prediction(record)
    failed_tools = [
        item for item in tool_results
        if item.get("success") is not True
    ]
    successful_tools = [
        item for item in tool_results
        if item.get("success") is True
    ]
    if failed_tools:
        first = failed_tools[0]
        return "tool_execution_failure", (
            f"{first.get('tool_name')}: {first.get('error') or 'tool returned success=false'}"
        )
    if not tool_results and not record.get("actual_tools_called"):
        return "tool_selection_failure", "Agent produced no tool evidence for this sample."
    if str(record.get("actual_path") or "") == "failure":
        return "tool_selection_failure", "Agent failed to reach a successful eval path."
    if successful_tools:
        return "answer_mismatch", "Tool evidence was collected, but the final answer was incorrect."
    return "unknown_failure", "Failure could not be attributed from available prediction fields."


def save_vitalbench_error_analysis(
    output_dir: Path,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [
        record for record in predictions
        if record.get("score") is not True or record.get("error_message")
    ]
    rows: list[dict[str, Any]] = []
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_target: dict[str, Counter[str]] = defaultdict(Counter)
    by_attribution: Counter[str] = Counter()

    for record in failures:
        attribution, note = classify_vitalbench_failure(record)
        attribution = attribution or "unlabeled"
        condition = str(record.get("condition") or "")
        dataset = str(record.get("dataset") or "")
        target = str(record.get("target") or "")
        by_attribution[attribution] += 1
        by_condition[condition][attribution] += 1
        by_dataset[dataset][attribution] += 1
        by_target[target][attribution] += 1
        rows.append(
            {
                "condition": condition,
                "question_id": record.get("question_id"),
                "dataset": dataset,
                "question_type": record.get("question_type"),
                "target": target,
                "expected_path": record.get("expected_path"),
                "actual_path": record.get("actual_path"),
                "score": record.get("score"),
                "error_attribution": attribution,
                "error_attribution_note": note,
                "actual_tools_called": json.dumps(record.get("actual_tools_called") or [], ensure_ascii=False),
                "gold_answer": json.dumps(record.get("gold_answer"), ensure_ascii=False),
                "prediction_preview": str(record.get("prediction") or "")[:300],
                "error_message": record.get("error_message"),
                "elapsed_sec": record.get("elapsed_sec"),
            }
        )

    summary = {
        "total_predictions": len(predictions),
        "total_failures": len(failures),
        "by_error_attribution": dict(by_attribution),
        "by_condition": {
            key: dict(counter)
            for key, counter in sorted(by_condition.items())
        },
        "by_dataset": {
            key: dict(counter)
            for key, counter in sorted(by_dataset.items())
        },
        "by_target": {
            key: dict(counter)
            for key, counter in sorted(by_target.items())
        },
    }
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "error_analysis_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ERROR_ANALYSIS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_jsonl(output_dir / "error_analysis_failures.jsonl", rows)
    return summary


def git_commit() -> str | None:
    import subprocess

    result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def build_summary_markdown(
    *,
    command: str,
    eval_payload: dict[str, Any],
    state_count: int,
    qa_count: int,
    predictions: list[dict[str, Any]],
) -> str:
    aggregates = eval_payload["aggregate"]
    agent = aggregates.get("agent", {})
    agent_acc = agent.get("accuracy")
    lines = [
        "# VitalBench Reactive Eval Summary",
        "",
        "## Run command",
        "",
        f"`{command}`",
        "",
        "## Resolved Configuration",
        "",
        "```json",
        json.dumps(
            eval_payload.get("resolved_config", {}),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "```",
        "",
        "## Counts",
        "",
        f"- States: {state_count}",
        f"- QA records: {qa_count}",
        f"- Evaluated samples: {eval_payload['max_samples']}",
        "",
        "## Accuracy",
        "",
    ]
    for condition in eval_payload.get("conditions") or sorted(aggregates):
        condition_acc = aggregates.get(condition, {}).get("accuracy")
        lines.append(f"- {condition}: {_pct(condition_acc)}")
    if "agent" in aggregates:
        for condition, item in sorted(aggregates.items()):
            if condition == "agent":
                continue
            condition_acc = item.get("accuracy")
            delta = None if agent_acc is None or condition_acc is None else agent_acc - condition_acc
            lines.append(f"- Delta agent - {condition}: {_pct(delta)}")
    lines += [
        "",
        "## Agent Routing",
        "",
        f"- Distribution: `{json.dumps(agent.get('routing_distribution', {}), sort_keys=True)}`",
        f"- Accuracy by actual path: `{json.dumps(agent.get('accuracy_by_actual_path', {}), sort_keys=True)}`",
        f"- Accuracy by expected path: `{json.dumps(agent.get('accuracy_by_expected_path', {}), sort_keys=True)}`",
    ]
    if eval_payload.get("routing_analysis_limited"):
        lines += ["", "> Warning: routing analysis is limited for this run."]
    error_analysis = eval_payload.get("error_analysis") or {}
    if error_analysis:
        lines += [
            "",
            "## Error Analysis",
            "",
            f"- Failures: {error_analysis.get('total_failures', 0)} / {error_analysis.get('total_predictions', 0)}",
            f"- By attribution: `{json.dumps(error_analysis.get('by_error_attribution', {}), sort_keys=True)}`",
            "- Files: `error_analysis_summary.json`, `error_analysis_failures.csv`, `error_analysis_failures.jsonl`",
        ]
    trace_perf = eval_payload.get("trace_performance") or {}
    if trace_perf:
        lines += ["", "## Agent Trace Performance"]
        slowest = trace_perf.get("slowest_samples") or []
        if slowest:
            lines += ["", "Slowest samples:"]
            for item in slowest[:5]:
                lines.append(
                    "- "
                    f"`{item.get('question_id')}` total={_sec(item.get('duration_sec'))} "
                    f"planner={_sec(item.get('planner_duration_sec'))} "
                    f"tools={_sec(item.get('tool_duration_sec'))} "
                    f"answer={_sec(item.get('answer_duration_sec'))}"
                )
        tools = trace_perf.get("tools") or {}
        if tools:
            lines += ["", "Tool timings:"]
            for name, item in sorted(
                tools.items(),
                key=lambda pair: float(pair[1].get("total_duration_sec") or 0.0),
                reverse=True,
            )[:10]:
                lines.append(
                    "- "
                    f"`{name}` count={item.get('count')} "
                    f"avg={_sec(item.get('avg_duration_sec'))} "
                    f"max={_sec(item.get('max_duration_sec'))} "
                    f"success={item.get('success_count')}/{item.get('count')}"
                )
    failures = [record for record in predictions if record.get("score") is not True][:5]
    lines += ["", "## Top Failure Examples"]
    if not failures:
        lines.append("")
        lines.append("No failures recorded.")
    for record in failures:
        lines += [
            "",
            f"- `{record['condition']}` `{record['question_id']}` expected `{record['gold_answer']}` got `{record['prediction'][:160]}`",
        ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-jsonl", default=DEFAULT_QA_JSONL, type=Path)
    parser.add_argument(
        "--states-jsonl",
        default=DEFAULT_STATES_JSONL,
        type=Path,
        help=(
            "MonitoringState JSONL. Optional in default raw mode; required for "
            "--agent-data-mode state."
        ),
    )
    parser.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST, type=Path)
    parser.add_argument("--tier", default="A")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--miss-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--perturb-rate",
        type=float,
        default=0.0,
        help="Probability of eval-side tool-result perturbation per eligible tool call.",
    )
    parser.add_argument(
        "--perturb-mode",
        choices=["tool_failure", "empty_field"],
        default="tool_failure",
        help="How to corrupt an eligible tool result when perturbation fires.",
    )
    parser.add_argument(
        "--perturb-target",
        choices=["critical", "any"],
        default="critical",
        help=(
            "critical: only ValidationGate.CRITICAL_TOOLS. any: every exposed "
            "eval tool call is eligible."
        ),
    )
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=None,
        help="Seed for deterministic perturbation decisions. Defaults to --seed.",
    )
    parser.add_argument(
        "--evaluation-protocol",
        choices=EVALUATION_PROTOCOLS,
        default="rebuttal_v2",
        help=(
            "paper_v1 reproduces the submitted evaluation semantics; "
            "rebuttal_v2 enables the revised rebuttal protocol."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        metavar="CONDITION",
        default=["agent"],
        help=(
            "Reactive evaluation conditions. Choices: "
            + ", ".join(VALID_EVAL_CONDITIONS)
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--question-type", choices=["single_verify", "single_choose", "single_query"])
    parser.add_argument("--limit-per-target", type=int)
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--agent-no-planner-tool-count",
        dest="agent_no_planner_tool_count",
        type=int,
        default=1,
        help=(
            "Minimum/fixed number of tools to select per sample for the "
            "No Planner ablation."
        ),
    )
    parser.add_argument(
        "--agent-no-planner-tool-count-max",
        dest="agent_no_planner_tool_count_max",
        type=int,
        help=(
            "If set above --agent-no-planner-tool-count, sample a per-sample "
            "tool count uniformly from the configured min/max range."
        ),
    )
    parser.add_argument(
        "--trace-agent",
        action="store_true",
        help=(
            "Emit detailed per-sample Agent traces to stderr and "
            "write agent_traces.jsonl in the output directory."
        ),
    )
    parser.add_argument(
        "--trace-result-chars",
        type=int,
        default=2000,
        help="Maximum characters to keep for each traced tool argument/result preview.",
    )
    parser.add_argument(
        "--agent-data-mode",
        choices=["state", "raw"],
        default="raw",
        help=(
            "raw: default fair eval mode; hide MonitoringState and expose "
            "sanitized raw-signal tools only. state: legacy/debug mode with "
            "MonitoringState tools visible to the agent."
        ),
    )
    parser.add_argument(
        "--include-label-generated-states",
        action="store_true",
        help=(
            "Legacy/debug override for --agent-data-mode state: expose dataset "
            "label or annotation facts in MonitoringState records. By default, "
            "state-mode projects those fields out before exposing states."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    return parser


def _validate_conditions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    invalid = [condition for condition in args.conditions if condition not in VALID_EVAL_CONDITIONS]
    if invalid:
        parser.error(
            "invalid --conditions value(s): "
            + ", ".join(invalid)
            + ". Choices: "
            + ", ".join(VALID_EVAL_CONDITIONS)
        )


def _validate_protocol_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.evaluation_protocol == "paper_v1":
        incompatible = sorted(
            set(args.conditions) & {"agent_no_planner", "agent_no_tools", "agent_all_tools"}
        )
        if incompatible:
            parser.error(
                "paper_v1 does not support rebuttal-v2 condition(s): "
                + ", ".join(incompatible)
                + ". Use agent_no_planner_paper_v1 for the submitted random-tool ablation."
            )
        if args.perturb_rate != 0.0:
            parser.error("paper_v1 requires --perturb-rate 0")
    elif "agent_no_planner_paper_v1" in args.conditions:
        parser.error(
            "agent_no_planner_paper_v1 requires --evaluation-protocol paper_v1"
        )


def _validate_perturbation_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if not 0.0 <= args.perturb_rate <= 1.0:
        parser.error("--perturb-rate must be between 0.0 and 1.0")
    if args.perturb_seed is None:
        args.perturb_seed = args.seed


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(cli_argv)
    _validate_conditions(args, parser)
    _validate_protocol_args(args, parser)
    _validate_perturbation_args(args, parser)
    perturbation_config = PerturbationConfig(
        rate=args.perturb_rate,
        mode=args.perturb_mode,
        target=args.perturb_target,
        seed=args.perturb_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.monotonic()
    log_info("[VitalBench eval] Starting run", quiet=args.quiet)
    log_info(f"[VitalBench eval] Output dir: {args.output_dir}", quiet=args.quiet)
    log_info(
        "[VitalBench eval] Conditions: "
        + ", ".join(args.conditions)
        + f" protocol={args.evaluation_protocol}"
        + f" data_mode={args.agent_data_mode}"
        + (" trace_agent=on" if args.trace_agent else "")
        + (
            f" perturb_rate={args.perturb_rate}"
            if perturbation_config.enabled
            else ""
        )
        + (" (dry-run)" if args.dry_run else ""),
        quiet=args.quiet,
    )

    manifest = SourceManifest.load(args.source_manifest)
    set_active_source_manifest(manifest)
    log_info(
        f"[VitalBench eval] Loaded manifest: {args.source_manifest} "
        f"({len(manifest.entries)} entries)",
        quiet=args.quiet,
    )

    if args.agent_data_mode == "state" and args.states_jsonl is None:
        parser = build_parser()
        parser.error("--states-jsonl is required when --agent-data-mode state")
    states = load_states_jsonl(args.states_jsonl) if args.states_jsonl is not None else []
    project_label_generated_state_facts = (
        args.agent_data_mode == "state" and not args.include_label_generated_states
    )
    set_label_generated_state_visibility_filter(project_label_generated_state_facts)
    label_generated_state_count = (
        sum(1 for state in states if is_label_generated_state(state))
        if project_label_generated_state_facts
        else 0
    )
    visible_states = (
        project_label_generated_states(states)
        if project_label_generated_state_facts
        else states
    )
    samples_all = load_qa_jsonl(args.qa_jsonl)
    log_info(
        f"[VitalBench eval] Loaded states={len(states)} QA={len(samples_all)}",
        quiet=args.quiet,
    )
    if project_label_generated_state_facts:
        log_info(
            "[VitalBench eval] State visibility filter enabled: projecting "
            f"label-generated fields out of {label_generated_state_count} "
            "condition-visible states.",
            quiet=args.quiet,
        )
    states_by_id = {state.state_id: state for state in states}
    build_supported_contexts = {
        key
        for key in {context_key_for_state(state) for state in states}
        if is_builder_registry_supported_context(key)
    }
    if args.agent_data_mode == "raw":
        store = get_global_state_store()
        store.clear()
        split = SplitResult(set(), set(), True)
        preloaded_state_count = 0
        log_info(
            "[VitalBench eval] Raw-data mode: MonitoringState records are not "
            "preloaded or exposed to the agent.",
            quiet=args.quiet,
        )
    else:
        split = split_preload_contexts(
            states,
            args.miss_ratio,
            args.seed,
            build_supported_contexts=build_supported_contexts,
        )
        preloaded_state_count = load_preloaded_states(visible_states, split.preloaded_contexts)
        log_info(
            "[VitalBench eval] Partial preload: "
            f"contexts preloaded={len(split.preloaded_contexts)} "
            f"missing={len(split.missing_contexts)} "
            f"states_loaded={preloaded_state_count} "
            f"routing_limited={split.routing_analysis_limited}",
            quiet=args.quiet,
        )

    samples = filter_samples(
        samples_all,
        tier=args.tier,
        dataset=args.dataset,
        question_type=args.question_type,
        max_samples=args.max_samples,
        limit_per_target=args.limit_per_target,
        seed=args.seed,
    )
    log_info(
        f"[VitalBench eval] Evaluating {len(samples)} samples "
        f"(tier={args.tier}, max_samples={args.max_samples}, seed={args.seed})",
        quiet=args.quiet,
    )

    predictions: list[dict[str, Any]] = []
    agent_trace_records: list[dict[str, Any]] = []
    prompts: dict[str, str] = {}
    started = time.monotonic()
    for index, sample in enumerate(samples, start=1):
        agent_input = sample.agent_input
        eval_spec = sample.eval_spec
        locator = agent_input.window_locator
        target = sample.gt_metadata.get("target")
        key = sample_context_key(sample, states_by_id)
        expected_path = (
            "raw_signal"
            if args.agent_data_mode == "raw"
            else "build" if key in split.missing_contexts else "query"
        )
        log_info(
            f"[VitalBench eval] Sample {index}/{len(samples)} "
            f"{agent_input.question_id} expected_path={expected_path} "
            f"target={target}",
            quiet=args.quiet,
        )
        base = {
            "question_id": agent_input.question_id,
            "evaluation_protocol": args.evaluation_protocol,
            "dataset": locator.dataset,
            "patient_id": locator.patient_id,
            "modality": sample.gt_metadata.get("modality"),
            "subject_id": key.subject_id,
            "recording_id": key.recording_id,
            "template_id": sample.gt_metadata.get("template_id"),
            "question_type": sample.gt_metadata.get("question_type"),
            "target": target,
            "raw_observability": raw_signal_observability(str(target)),
            "question": agent_input.question,
            "window_locator": _compact_dataclass_dict(locator),
            "agent_input": asdict(agent_input),
            "gold_answer": eval_spec.answer,
            "evaluation_method": eval_spec.evaluation_method,
            "eval_spec": asdict(eval_spec),
            "gt_metadata": sample.gt_metadata,
            "expected_path": expected_path,
        }

        if "agent" in args.conditions:
            condition_started = time.monotonic()
            log_info(
                "[VitalBench eval]   agent start",
                quiet=args.quiet,
            )
            trace = (
                EvalTraceRecorder(
                    agent_input.question_id,
                    condition="agent",
                    evaluation_protocol=args.evaluation_protocol,
                    tier=str(sample.gt_metadata.get("tier") or ""),
                    template_id=str(sample.gt_metadata.get("template_id") or ""),
                    target=str(target or ""),
                    dataset=str(locator.dataset or ""),
                    quiet=args.quiet,
                    result_max_chars=args.trace_result_chars,
                )
                if args.trace_agent
                else None
            )
            agent_result = run_agent_sample(
                agent_input,
                dry_run=args.dry_run,
                data_mode=args.agent_data_mode,
                trace=trace,
                benchmark_tier=sample.gt_metadata.get("tier"),
                benchmark_target=str(target or ""),
                evaluation_protocol=args.evaluation_protocol,
                perturbation=perturbation_config,
            )
            if trace is not None:
                agent_trace_records.append(trace.to_record())
            score, error_class, numeric_match_method = score_prediction_details(
                eval_spec,
                agent_result["prediction"],
            )
            condition_elapsed = time.monotonic() - condition_started
            log_info(
                "[VitalBench eval]   agent done "
                f"score={score} actual_path={agent_result['actual_path']} "
                f"tools={agent_result['actual_tools_called']} "
                f"elapsed={condition_elapsed:.1f}s",
                quiet=args.quiet,
            )
            trace_summary = agent_result.get("trace_summary")
            predictions.append(
                {
                    **base,
                    "condition": "agent",
                    "prediction": agent_result["prediction"],
                    "score": score,
                    "numeric_match_method": numeric_match_method,
                    **normalize_token_usage(agent_result.get("token_usage")),
                    **model_cost_prediction_fields(agent_result),
                    "actual_path": agent_result["actual_path"],
                    "actual_tools_called": agent_result["actual_tools_called"],
                    "build_success": agent_result["build_success"],
                    "query_success": agent_result["query_success"],
                    "error_class": agent_result["error_class"] or error_class,
                    "error_message": agent_result["error_message"],
                    "elapsed_sec": condition_elapsed,
                    "trace_summary": trace_summary,
                    **trace_prediction_fields(trace_summary),
                    "raw_response": agent_result["raw_response"],
                }
            )

        agent_ablation_specs = [
            (
                "agent_no_validation",
                {"no_validation_baseline": True},
                {},
            ),
            (
                "agent_no_replan",
                {"no_replan_baseline": True},
                {},
            ),
        ]
        for condition_name, run_kwargs, extra_record_fields in agent_ablation_specs:
            if condition_name not in args.conditions:
                continue
            condition_started = time.monotonic()
            log_info(
                f"[VitalBench eval]   {condition_name} start",
                quiet=args.quiet,
            )
            trace = (
                EvalTraceRecorder(
                    agent_input.question_id,
                    condition=condition_name,
                    evaluation_protocol=args.evaluation_protocol,
                    tier=str(sample.gt_metadata.get("tier") or ""),
                    template_id=str(sample.gt_metadata.get("template_id") or ""),
                    target=str(target or ""),
                    dataset=str(locator.dataset or ""),
                    quiet=args.quiet,
                    result_max_chars=args.trace_result_chars,
                )
                if args.trace_agent
                else None
            )
            agent_result = run_agent_sample(
                agent_input,
                dry_run=args.dry_run,
                data_mode=args.agent_data_mode,
                trace=trace,
                benchmark_tier=sample.gt_metadata.get("tier"),
                benchmark_target=str(target or ""),
                evaluation_protocol=args.evaluation_protocol,
                perturbation=perturbation_config,
                **run_kwargs,
            )
            if trace is not None:
                agent_trace_records.append(trace.to_record())
            score, error_class, numeric_match_method = score_prediction_details(
                eval_spec,
                agent_result["prediction"],
            )
            condition_elapsed = time.monotonic() - condition_started
            log_info(
                f"[VitalBench eval]   {condition_name} done "
                f"score={score} actual_path={agent_result['actual_path']} "
                f"tools={agent_result['actual_tools_called']} "
                f"elapsed={condition_elapsed:.1f}s",
                quiet=args.quiet,
            )
            trace_summary = agent_result.get("trace_summary")
            predictions.append(
                {
                    **base,
                    "condition": condition_name,
                    "prediction": agent_result["prediction"],
                    "score": score,
                    "numeric_match_method": numeric_match_method,
                    **normalize_token_usage(agent_result.get("token_usage")),
                    **model_cost_prediction_fields(agent_result),
                    "actual_path": agent_result["actual_path"],
                    "actual_tools_called": agent_result["actual_tools_called"],
                    "build_success": agent_result["build_success"],
                    "query_success": agent_result["query_success"],
                    "error_class": agent_result["error_class"] or error_class,
                    "error_message": agent_result["error_message"],
                    "elapsed_sec": condition_elapsed,
                    "trace_summary": trace_summary,
                    **trace_prediction_fields(trace_summary),
                    "raw_response": agent_result["raw_response"],
                    **extra_record_fields,
                }
            )

        tool_strategy_conditions = (
            ("agent_no_planner_paper_v1", "random", "paper_v1"),
            ("agent_no_planner", "random", "rebuttal_v2"),
            ("agent_no_tools", "none", "rebuttal_v2"),
            ("agent_all_tools", "all", "rebuttal_v2"),
        )
        for condition_name, selection_mode, no_planner_implementation in tool_strategy_conditions:
            if condition_name not in args.conditions:
                continue
            condition_started = time.monotonic()
            log_info(
                f"[VitalBench eval]   {condition_name} start "
                f"mode={selection_mode} seed={args.seed} "
                f"count={args.agent_no_planner_tool_count}"
                + (
                    ""
                    if args.agent_no_planner_tool_count_max is None
                    else f"..{args.agent_no_planner_tool_count_max}"
                ),
                quiet=args.quiet,
            )
            trace = (
                EvalTraceRecorder(
                    agent_input.question_id,
                    condition=condition_name,
                    evaluation_protocol=args.evaluation_protocol,
                    tier=str(sample.gt_metadata.get("tier") or ""),
                    template_id=str(sample.gt_metadata.get("template_id") or ""),
                    target=str(target or ""),
                    dataset=str(locator.dataset or ""),
                    quiet=args.quiet,
                    result_max_chars=args.trace_result_chars,
                )
                if args.trace_agent
                else None
            )
            agent_result = run_agent_sample(
                agent_input,
                dry_run=args.dry_run,
                data_mode=args.agent_data_mode,
                trace=trace,
                benchmark_tier=sample.gt_metadata.get("tier"),
                benchmark_target=str(target or ""),
                evaluation_protocol=args.evaluation_protocol,
                no_planner_baseline=True,
                no_planner_implementation=no_planner_implementation,
                no_planner_selection_mode=selection_mode,
                no_planner_seed=args.seed,
                no_planner_tool_count=args.agent_no_planner_tool_count,
                no_planner_tool_count_max=args.agent_no_planner_tool_count_max,
                perturbation=perturbation_config,
            )
            if trace is not None:
                agent_trace_records.append(trace.to_record())
            score, error_class, numeric_match_method = score_prediction_details(
                eval_spec,
                agent_result["prediction"],
            )
            condition_elapsed = time.monotonic() - condition_started
            log_info(
                f"[VitalBench eval]   {condition_name} done "
                f"score={score} actual_path={agent_result['actual_path']} "
                f"tools={agent_result['actual_tools_called']} "
                f"elapsed={condition_elapsed:.1f}s",
                quiet=args.quiet,
            )
            trace_summary = agent_result.get("trace_summary")
            predictions.append(
                {
                    **base,
                    "condition": condition_name,
                    "prediction": agent_result["prediction"],
                    "score": score,
                    "numeric_match_method": numeric_match_method,
                    **normalize_token_usage(agent_result.get("token_usage")),
                    **model_cost_prediction_fields(agent_result),
                    "actual_path": agent_result["actual_path"],
                    "actual_tools_called": agent_result["actual_tools_called"],
                    "build_success": agent_result["build_success"],
                    "query_success": agent_result["query_success"],
                    "error_class": agent_result["error_class"] or error_class,
                    "error_message": agent_result["error_message"],
                    "elapsed_sec": condition_elapsed,
                    "trace_summary": trace_summary,
                    **trace_prediction_fields(trace_summary),
                    "raw_response": agent_result["raw_response"],
                    "agent_no_planner_seed": args.seed,
                    "agent_no_planner_tool_count": args.agent_no_planner_tool_count,
                    "agent_no_planner_tool_count_max": args.agent_no_planner_tool_count_max,
                    "agent_no_planner_implementation": no_planner_implementation,
                    "agent_no_planner_selection_mode": selection_mode,
                }
            )

    run_id = datetime.now(timezone.utc).strftime("vitalbench_%Y%m%dT%H%M%SZ")
    aggregate = {
        condition: aggregate_predictions(predictions, condition)
        for condition in args.conditions
    }
    error_analysis = save_vitalbench_error_analysis(args.output_dir, predictions)
    command = "uv run python -m agent.evaluation.reactive.vitalbench_eval " + " ".join(
        [
            f"--qa-jsonl {args.qa_jsonl}",
            *([] if args.states_jsonl is None else [f"--states-jsonl {args.states_jsonl}"]),
            f"--source-manifest {args.source_manifest}",
            f"--tier {args.tier}",
            f"--max-samples {args.max_samples}",
            f"--miss-ratio {args.miss_ratio}",
            f"--seed {args.seed}",
            f"--perturb-rate {args.perturb_rate}",
            f"--perturb-mode {args.perturb_mode}",
            f"--perturb-target {args.perturb_target}",
            f"--perturb-seed {args.perturb_seed}",
            f"--evaluation-protocol {args.evaluation_protocol}",
            "--conditions " + " ".join(args.conditions),
            f"--output-dir {args.output_dir}",
            *([] if args.dataset is None else [f"--dataset {args.dataset}"]),
            *([] if args.question_type is None else [f"--question-type {args.question_type}"]),
            *([] if args.limit_per_target is None else [f"--limit-per-target {args.limit_per_target}"]),
            f"--agent-data-mode {args.agent_data_mode}",
            f"--agent-no-planner-tool-count {args.agent_no_planner_tool_count}",
            *(
                []
                if args.agent_no_planner_tool_count_max is None
                else [f"--agent-no-planner-tool-count-max {args.agent_no_planner_tool_count_max}"]
            ),
            *(["--trace-agent", f"--trace-result-chars {args.trace_result_chars}"] if args.trace_agent else []),
            *(["--include-label-generated-states"] if args.include_label_generated_states else []),
        ]
    )
    eval_payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "resolved_config": resolved_config_snapshot(),
        "qa_jsonl": str(args.qa_jsonl),
        "states_jsonl": None if args.states_jsonl is None else str(args.states_jsonl),
        "source_manifest": str(args.source_manifest),
        "tier": args.tier,
        "max_samples": len(samples),
        "limit_per_target": args.limit_per_target,
        "miss_ratio": args.miss_ratio,
        "seed": args.seed,
        "perturb_rate": args.perturb_rate,
        "perturb_mode": args.perturb_mode,
        "perturb_target": args.perturb_target,
        "perturb_seed": args.perturb_seed,
        "perturbation": perturbation_config.to_dict(),
        "evaluation_protocol": args.evaluation_protocol,
        "evaluation_protocol_source": (
            "paper_v1_frozen"
            if args.evaluation_protocol == "paper_v1"
            else "rebuttal_v2"
        ),
        "conditions": args.conditions,
        "agent_data_mode": args.agent_data_mode,
        "project_label_generated_state_facts": project_label_generated_state_facts,
        "label_generated_states_projected": label_generated_state_count,
        "include_label_generated_states": args.include_label_generated_states,
        "agent_no_planner_tool_count": args.agent_no_planner_tool_count,
        "agent_no_planner_tool_count_max": args.agent_no_planner_tool_count_max,
        "routing_analysis_limited": split.routing_analysis_limited,
        "elapsed_sec": time.monotonic() - started,
        "preloaded_contexts": [
            asdict(key)
            for key in sorted(
                split.preloaded_contexts,
                key=lambda item: (item.dataset, item.subject_id, item.recording_id),
            )
        ],
        "missing_contexts": [
            asdict(key)
            for key in sorted(
                split.missing_contexts,
                key=lambda item: (item.dataset, item.subject_id, item.recording_id),
            )
        ],
        "agent_trace_enabled": args.trace_agent,
        "trace_performance": aggregate_trace_performance(agent_trace_records),
        "error_analysis": error_analysis,
        "aggregate": aggregate,
    }
    (args.output_dir / "eval.json").write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    if args.trace_agent:
        write_jsonl(args.output_dir / "agent_traces.jsonl", agent_trace_records)
    (args.output_dir / "summary.md").write_text(
        build_summary_markdown(
            command=command,
            eval_payload=eval_payload,
            state_count=len(states),
            qa_count=len(samples_all),
            predictions=predictions,
        ),
        encoding="utf-8",
    )
    if args.save_prompts:
        (args.output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.dry_run:
            for key, prompt in prompts.items():
                print(f"\n[{key}]\n{prompt}")
    log_info(
        "[VitalBench eval] Finished. "
        f"Wrote eval.json, predictions.jsonl, summary.md, error analysis in {args.output_dir}. "
        f"Elapsed={time.monotonic() - run_started:.1f}s",
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
