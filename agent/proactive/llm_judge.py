"""
Proactive LLM judge — contextual decision layer for the Proactive Engine.

This module is the non-rule half of the Proactive Engine's decision
logic. Where ``RuleBasedJudge`` fires on unambiguous, guideline-
grounded events (extreme HR, long AF episodes), ``ProactiveLLMJudge``
handles events whose clinical action depends on context the rule layer
cannot encode — for example:

  - AHRE episodes of 5 minutes to 24 hours, where the guideline's
    recommendation is conditional on CHA2DS2-VASc score.
  - Rhythm transitions from N to AF / AFL that warrant user-facing
    advice but may or may not warrant an alert depending on burden.
  - Heart rates between 50 and 100 bpm that are normal in one context
    and abnormal in another.

The judge is invoked only when a coarse gating check (see
``should_invoke_judge``) indicates the current window is worth the
LLM's attention. It is given the current health state plus the full
executive summaries of all three clinical guidelines, and it may also
request specific guideline sections via a tool.

Safety:
    By default the judge runs in STUB mode and never calls the LLM.
    Live LLM calls happen only when the env var ``LLM_PROACTIVE_ENABLED``
    is true. This prevents accidental token burn during long streaming
    runs.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from agent.proactive.intervention_decision import (
    InterventionResult,
    Modality,
    Urgency,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class CitedSection:
    """A guideline section the LLM cited when producing a decision."""

    guideline_id: str
    section_id: str

    def fully_qualified_id(self) -> str:
        return f"{self.guideline_id}.{self.section_id}"


@dataclass
class JudgeDecision:
    """
    Structured decision from the LLM judge.

    Aligns with ``InterventionResult`` (so pipeline merge logic is
    uniform) but adds free-form advice plus the guideline sections the
    LLM drew on — both are Phase-2 inputs for LLM-as-judge evaluation
    of citation faithfulness and advice quality.
    """

    intervene: bool = False
    urgency: Urgency = Urgency.NONE
    reason: str = ""
    advice: str = ""
    cited_sections: list[CitedSection] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=dict)
    # How many guideline-section tool calls the LLM made for this decision.
    # Useful for compute-cost analysis and evaluation ablation.
    tool_call_count: int = 0

    def to_intervention_result(self) -> InterventionResult:
        """Convert to the pipeline's shared result type."""
        return InterventionResult(
            intervene=self.intervene,
            urgency=self.urgency,
            reason=self.reason,
            triggered_rules=(
                [f"llm:{c.fully_qualified_id()}" for c in self.cited_sections]
                if self.cited_sections
                else None
            ),
        )

    @staticmethod
    def no_alert() -> "JudgeDecision":
        return JudgeDecision(intervene=False, urgency=Urgency.NONE)


# ---------------------------------------------------------------------------
# Coarse gating (non-LLM, just "is this worth the LLM's attention")
# ---------------------------------------------------------------------------
#
# The gate's job is compute-budget management, not clinical judgement.
# Thresholds here are broad enough to surface any state the LLM should
# look at. These are intentionally NOT guideline-derived — they just
# filter out obviously-normal windows. The downstream LLM is responsible
# for the clinical call.


_GATE_HR_LOW_BPM = 50.0
_GATE_HR_HIGH_BPM = 100.0


def should_invoke_judge(health_state: dict[str, Any]) -> bool:
    """
    Return True if the LLM judge should be called for the current window.

    Invoke if any of:
      - HR outside [50, 100] bpm
      - Current rhythm class is not "N" (i.e. AF, AFL, Other)
      - Short window flags an irregular rhythm
      - Sustained tachycardia statistic has crossed 50% (worth a look
        before it reaches the hard-rule threshold of 80%)

    Otherwise skip, saving LLM calls on normal sinus windows.

    Phase 1 note: the ``rhythm_class`` path never triggers today — the
    tracker does not yet populate that field (scheduled for Day 5 /
    rhythm classifier). The check is written defensively so it
    activates automatically once the tracker catches up.
    """
    short = health_state.get("short") or {}

    hr = short.get("hr_bpm")
    if hr is not None:
        if hr < _GATE_HR_LOW_BPM or hr > _GATE_HR_HIGH_BPM:
            return True

    rhythm_class = short.get("rhythm_class")
    if rhythm_class is not None and rhythm_class != "N":
        return True

    rhythm_regular = short.get("rhythm_regular")
    if rhythm_regular is False:
        return True

    tachy_ratio = short.get("tachycardia_ratio_5min")
    if tachy_ratio is not None and tachy_ratio >= 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_system_prompt(modality: Modality, guideline_summaries: str) -> str:
    """
    Construct the system prompt for the Proactive LLM judge.

    The prompt contains:
      - role description
      - input contract (what the judge will receive)
      - output contract (structured JSON fields)
      - available tool (read_guideline_section)
      - consumer-device action ceiling
      - the executive summaries of all enabled guidelines
    """
    signal_phrase = (
        "electrocardiogram (ECG)" if modality == Modality.ECG else "photoplethysmogram (PPG)"
    )

    return f"""You are the contextual decision layer of an mHealth cardiac monitoring agent.

On every window of {signal_phrase} data, the agent evaluates whether to
alert the user. Two layers make that call: a guideline-grounded rule
layer (handles unambiguous extremes and sustained tachycardia at rest)
and you, the LLM judge. Your role is to handle the events where the
guidelines' recommendations are conditional on context the rule layer
cannot encode.

## Your input

On each invocation you receive a structured snapshot of the user's
current health state:
- `hr_bpm`: current heart rate, or null
- `rhythm_class`: "N", "AF", "AFL", "Other", or null
- `af_episode_duration_s`: seconds the current AF episode has lasted,
  or null if not in AF
- `tachycardia_ratio_5min`: fraction of the trailing 5-minute window
  with HR > 100 bpm (0.0 to 1.0), or null
- `tachycardia_sample_count`: number of HR samples in that trailing
  window

You also receive a small patient context (known conditions, if any),
which may be empty in Phase 1.

## Your tools

You have one tool: `read_guideline_section(guideline_id, section_id)`.
Use it when you need the full text of a specific guideline section. The
executive summaries of all guidelines are already in your context below;
fetch full text only when a decision genuinely depends on detail the
summary does not cover.

Limit yourself to at most 3 tool calls per decision. Each unnecessary
call costs latency and tokens.

## Your output

Respond with a single JSON object (no markdown fences, no prose
outside the JSON) of this shape:

{{
  "intervene": true | false,
  "urgency": "none" | "low" | "medium" | "high" | "critical",
  "reason": "one-sentence clinical justification",
  "advice": "short plain-language guidance for the user",
  "cited_sections": [
    {{"guideline_id": "af-2023", "section_id": "ahre-5min-to-24h"}}
  ]
}}

Rules for each field:

- `intervene`: true only if the state warrants user-facing alerting.
  When false, the other fields should be empty strings or empty lists.
- `urgency`: pick the lowest tier consistent with the guidance. "low"
  is informational, "medium" is "consider seeing a clinician soon",
  "high" is "see a clinician promptly", "critical" is
  "seek immediate medical attention". Prefer "medium" unless the
  evidence genuinely warrants escalation.
- `reason`: a single sentence naming the clinical concern in plain
  terms (e.g. "Atrial fibrillation episode of 18 minutes in a user
  without a known AF history, within the guideline's AHRE decision
  band.").
- `advice`: short, non-diagnostic user-facing guidance. Never
  recommend medication changes. Never give a diagnosis. Always route
  the user to a qualified clinician when the recommended action is
  more than reassurance.
- `cited_sections`: zero or more guideline sections your reasoning
  relied on. Use the exact guideline_id and section_id from the
  summaries below. Empty list is acceptable for straightforward cases.

## Consumer-device constraints

This system is a consumer monitoring agent, not a diagnostic device.
Per FDA consumer-device framework:
- Never provide a diagnosis.
- Never recommend starting, stopping, or changing medication.
- When advice is needed beyond reassurance, the action ceiling is
  "consult a healthcare professional".

## When to return no alert

If the signals are within normal ranges, or within a range that is
expected for routine activity, or if the evidence is insufficient to
justify concern, return `intervene: false` with empty fields. Alert
fatigue is a real harm; do not alert on every marginal reading.

## Clinical guidelines (executive summaries)

{guideline_summaries}
"""


# ---------------------------------------------------------------------------
# Tool schema (OpenAI tool-calling format)
# ---------------------------------------------------------------------------


_READ_GUIDELINE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_guideline_section",
        "description": (
            "Fetch the full content of a specific guideline section by "
            "its guideline_id and section_id. Use this when you need "
            "detail beyond what the executive summary provides."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "guideline_id": {
                    "type": "string",
                    "description": (
                        "Stable ID of the guideline, e.g. 'af-2023', "
                        "'brady-2018', 'svt-2015'."
                    ),
                },
                "section_id": {
                    "type": "string",
                    "description": (
                        "Stable ID of the section within the guideline, "
                        "e.g. 'ahre-24h-plus'."
                    ),
                },
            },
            "required": ["guideline_id", "section_id"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


_ENABLED_ENV_FLAG = "LLM_PROACTIVE_ENABLED"
_MAX_TOOL_ITERATIONS = 3


class ProactiveLLMJudge:
    """
    Contextual decision layer using an LLM with access to clinical
    guidelines.

    Safety posture: stays in stub mode (returning no-alert) unless
    ``LLM_PROACTIVE_ENABLED`` is set to a truthy value in the
    environment. This prevents runaway token costs during unattended
    streaming demos.

    When enabled, uses the shared ``LLMService`` with an OpenAI-
    compatible tool-calling loop for ``read_guideline_section``.
    """

    def __init__(
        self,
        llm_service: Any = None,
        profile: Any = None,
        modality: Modality = Modality.ECG,
        library: Any = None,
        force_enabled: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        llm_service, profile:
            Dependency-injected LLM client. If both are None and the
            env flag is true, the judge lazy-constructs defaults via
            ``LLMService()`` and ``get_llm_profile("proactive")``.
        modality:
            ECG or PPG. Affects the system prompt's signal description.
        library:
            GuidelineLibrary instance. Lazy-constructed if None.
        force_enabled:
            If not None, overrides the env flag. Use for tests to
            enable without setting the env var.
        """
        self._modality = modality
        self._explicit_service = llm_service
        self._explicit_profile = profile
        self._explicit_library = library

        self._force_enabled = force_enabled
        self._enabled = self._resolve_enabled()

        # Lazy-initialise everything else on first use, so constructing
        # a judge doesn't hit the filesystem / env unnecessarily.
        self._service_cache = None
        self._profile_cache = None
        self._library_cache = None
        self._system_prompt_cache: str | None = None

        if self._enabled:
            logger.info(
                "ProactiveLLMJudge initialised in LIVE mode "
                "(modality=%s). Will make real LLM calls.",
                modality.value,
            )
        else:
            logger.info(
                "ProactiveLLMJudge initialised in STUB mode "
                "(modality=%s). Set %s=true to enable live calls.",
                modality.value,
                _ENABLED_ENV_FLAG,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        health_state: dict[str, Any],
        patient_context: dict[str, Any] | None = None,
    ) -> JudgeDecision:
        """
        Evaluate the current health state.

        Flow:
          1. Run coarse gating. If it says skip, return no-alert.
          2. If in stub mode, return no-alert after logging what would
             have been sent.
          3. If in live mode, call the LLM with tool-calling loop and
             parse the result.
          4. On any parsing / call error, fall back to no-alert with
             a logged warning. The rule layer is the safety net.
        """
        if not should_invoke_judge(health_state):
            return JudgeDecision.no_alert()

        if not self._enabled:
            logger.debug(
                "[STUB] LLM judge would be invoked. Short-state: %s",
                health_state.get("short"),
            )
            return JudgeDecision.no_alert()

        try:
            return self._call_llm(health_state, patient_context or {})
        except Exception as exc:
            logger.exception("LLM judge call failed; falling back to no-alert: %s", exc)
            return JudgeDecision.no_alert()

    # Backwards-compatible name used by pipeline code that reads
    # `judge._enabled` directly. Kept for compatibility; prefer the
    # public .enabled below in new code.
    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_enabled(self) -> bool:
        if self._force_enabled is not None:
            return bool(self._force_enabled)
        raw = os.getenv(_ENABLED_ENV_FLAG, "")
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    def _get_service(self):
        if self._explicit_service is not None:
            return self._explicit_service
        if self._service_cache is None:
            # Lazy import to keep `from agent.proactive.llm_judge import ...`
            # cheap for callers that never hit live mode.
            from agent.llm import LLMService

            self._service_cache = LLMService()
        return self._service_cache

    def _get_profile(self):
        if self._explicit_profile is not None:
            return self._explicit_profile
        if self._profile_cache is None:
            from agent.config import get_llm_profile

            self._profile_cache = get_llm_profile("proactive")
        return self._profile_cache

    def _get_library(self):
        if self._explicit_library is not None:
            return self._explicit_library
        if self._library_cache is None:
            from agent.proactive.guidelines import GuidelineLibrary

            self._library_cache = GuidelineLibrary()
        return self._library_cache

    def _get_system_prompt(self) -> str:
        if self._system_prompt_cache is None:
            library = self._get_library()
            # Concatenate all guideline summaries; they are short enough
            # (~30-50K tokens total) to fit comfortably in the system
            # prompt of any model with >= 128K context.
            parts: list[str] = []
            for gid in library.list_guideline_ids():
                parts.append(f"### Guideline: {gid}\n\n{library.get_summary_text(gid)}")
            summaries = "\n\n---\n\n".join(parts)
            self._system_prompt_cache = _build_system_prompt(self._modality, summaries)
        return self._system_prompt_cache

    # ------------------------------------------------------------------
    # LLM invocation + tool loop
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        health_state: dict[str, Any],
        patient_context: dict[str, Any],
    ) -> JudgeDecision:
        """
        Run a tool-calling loop against the configured LLM.

        The loop terminates when the model stops requesting tool calls
        or when we hit the iteration cap. Final message content is
        parsed as JudgeDecision.
        """
        system_prompt = self._get_system_prompt()
        user_message = _build_user_message(health_state, patient_context)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        total_usage = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        tool_calls_made = 0
        # Track which guideline sections were actually fetched via tool calls.
        fetched_sections: list[tuple[str, str]] = []

        service = self._get_service()
        profile = self._get_profile()
        client = service._get_client(profile)  # reuse existing cached client

        for iteration in range(_MAX_TOOL_ITERATIONS + 1):
            response = client.chat.completions.create(
                model=profile.model,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                messages=messages,
                tools=[_READ_GUIDELINE_TOOL_SCHEMA],
            )

            # Merge usage
            from agent.llm import usage_from_openai

            iter_usage = usage_from_openai(getattr(response, "usage", None))
            for k in total_usage:
                total_usage[k] += int(iter_usage.get(k, 0) or 0)

            choice = response.choices[0]
            message = choice.message

            # If the model wants tools, execute them and continue the loop.
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls and iteration < _MAX_TOOL_ITERATIONS:
                # Append the assistant message (with tool calls) to history.
                # DeepSeek-R1 and other thinking-mode models require
                # ``reasoning_content`` from the previous turn to be passed
                # back; otherwise the API rejects the next request with
                # "The reasoning_content in the thinking mode must be
                # passed back to the API."
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
                reasoning_content = getattr(message, "reasoning_content", None)
                if reasoning_content:
                    assistant_msg["reasoning_content"] = reasoning_content
                messages.append(assistant_msg)
                # Execute each tool call and append its result.
                for tc in tool_calls:
                    tool_calls_made += 1
                    content = self._execute_tool_call(tc.function.name, tc.function.arguments)
                    # Track sections actually fetched so citations are accurate
                    # even when the LLM omits them from its final JSON output.
                    if tc.function.name == "read_guideline_section":
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                            gid = args.get("guideline_id")
                            sid = args.get("section_id")
                            if gid and sid:
                                fetched_sections.append((str(gid), str(sid)))
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": content,
                        }
                    )
                continue

            # No tool calls, or we've hit the iteration cap. Parse JSON.
            raw = message.content or ""
            return _parse_judge_response(
                raw=raw,
                token_usage=total_usage,
                tool_call_count=tool_calls_made,
                fetched_sections=fetched_sections,
            )

        # Shouldn't reach here, but just in case.
        return JudgeDecision.no_alert()

    def _execute_tool_call(self, name: str, arguments_json: str) -> str:
        """Execute a tool the LLM requested and return the result as a string."""
        if name != "read_guideline_section":
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "Tool arguments were not valid JSON."})

        guideline_id = args.get("guideline_id")
        section_id = args.get("section_id")
        if not guideline_id or not section_id:
            return json.dumps({"error": "Both guideline_id and section_id are required."})

        library = self._get_library()
        section = library.get_section(guideline_id, section_id)
        if section is None:
            return json.dumps(
                {
                    "error": (
                        f"No section found for {guideline_id}.{section_id}. "
                        "Check the summary for valid section IDs."
                    )
                }
            )

        # Prefer full-text expansion if available; otherwise return the
        # summary section's content.
        full_text = library.get_fulltext(guideline_id, section_id)
        return json.dumps(
            {
                "guideline_id": section.guideline_id,
                "section_id": section.section_id,
                "title": section.title,
                "content": full_text or section.content,
                "has_fulltext": full_text is not None,
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_message(
    health_state: dict[str, Any],
    patient_context: dict[str, Any],
) -> str:
    """Format the per-window input as a user message."""
    short = health_state.get("short") or {}
    payload = {
        "short_window": {
            "hr_bpm": short.get("hr_bpm"),
            "rhythm_class": short.get("rhythm_class"),
            "rhythm_regular": short.get("rhythm_regular"),
            "af_episode_duration_s": short.get("af_episode_duration_s"),
            "tachycardia_ratio_5min": short.get("tachycardia_ratio_5min"),
            "tachycardia_sample_count": short.get("tachycardia_sample_count"),
        },
        "patient_context": patient_context,
    }
    return (
        "Evaluate the following window. Reply with the JSON decision object.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _parse_judge_response(
    *,
    raw: str,
    token_usage: dict[str, int],
    tool_call_count: int,
    fetched_sections: list[tuple[str, str]] | None = None,
) -> JudgeDecision:
    """Parse the LLM's final JSON response into a JudgeDecision.

    Citations are the union of:
    - Sections the LLM self-reported in ``cited_sections``
    - Sections actually fetched via ``read_guideline_section`` tool calls
      (tracked in ``fetched_sections``). This covers cases where the LLM
      reads a section but omits it from its JSON output.
    """
    from agent.llm import extract_json_object_text

    try:
        json_text = extract_json_object_text(raw)
        parsed = json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("LLM judge returned unparseable output: %s | raw: %s", exc, raw[:300])
        return JudgeDecision.no_alert()

    try:
        intervene = bool(parsed.get("intervene", False))
        urgency_raw = str(parsed.get("urgency", "none")).lower().strip()
        urgency = Urgency(urgency_raw) if urgency_raw in {u.value for u in Urgency} else Urgency.NONE
        reason = str(parsed.get("reason", ""))[:500]
        advice = str(parsed.get("advice", ""))[:1000]

        # Collect LLM self-reported citations.
        cited_raw = parsed.get("cited_sections") or []
        seen: set[tuple[str, str]] = set()
        cited_sections: list[CitedSection] = []
        for item in cited_raw:
            if not isinstance(item, dict):
                continue
            gid = item.get("guideline_id")
            sid = item.get("section_id")
            if gid and sid:
                key = (str(gid), str(sid))
                if key not in seen:
                    seen.add(key)
                    cited_sections.append(CitedSection(guideline_id=key[0], section_id=key[1]))

        # Union-merge with sections actually fetched via tool calls.
        for gid, sid in (fetched_sections or []):
            key = (gid, sid)
            if key not in seen:
                seen.add(key)
                cited_sections.append(CitedSection(guideline_id=gid, section_id=sid))
    except Exception:
        logger.exception("Failed to normalise LLM judge fields; raw=%s", raw[:300])
        return JudgeDecision.no_alert()

    # Safety: if the model says intervene=False, ensure urgency is NONE.
    if not intervene:
        urgency = Urgency.NONE

    return JudgeDecision(
        intervene=intervene,
        urgency=urgency,
        reason=reason,
        advice=advice,
        cited_sections=cited_sections,
        token_usage=token_usage,
        tool_call_count=tool_call_count,
    )
