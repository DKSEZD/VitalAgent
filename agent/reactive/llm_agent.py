"""
LLM Agent — answer generation module.

Takes the collected tool results and generates a natural-language response
to the user's query.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import replace
from typing import Iterator

from openai import OpenAI

from agent.config import get_llm_profile
from agent.llm import LLMProfile, LLMService, LLMStreamEvent
from agent.reactive.ecg_diagnosis_policy import classify_ecg_diagnosis_question
from agent.schemas import AgentResponse, Plan, RiskLevel, ToolResult
from agent.tools.registry import is_tool_enabled

logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT_TEMPLATE = """\
You are a multimodal mHealth Agent that helps users understand their available
health monitoring data, including ECG, PPG, and MonitoringState timelines.

## Guidelines
- Answer the user's question based on the health data context provided.
- Be accurate, concise, and empathetic.
- When the user's question is a yes/no question (e.g. "Does this ECG show..."), you MUST start your answer with "Yes" or "No" on the first line, then provide your explanation.
- For yes/no questions, the first word must match your final conclusion and the explanation that follows. Do not say "Yes" if your explanation supports "No", or vice versa.
- When the user's question presents specific choices (e.g. "A or B?", "A, B, or C?"), you MUST select exactly one option. Write your chosen option on the first line using the exact wording from the question, then explain your reasoning.
- When the user's question asks for a numeric value (e.g. heart rate in bpm, an interval in ms, a count), the first line MUST be a single numeric value with no units, no surrounding prose, and no leading words like "approximately" or "Based on...". Examples of correct first lines: `72`, `108.46`, `0.50`. Put the explanation on subsequent lines. If you cannot determine a number, output `none` on the first line.
- If the user message provides allowed answer labels, the first line MUST be exactly one of those labels. Do not answer "none" unless "none" is one of the allowed labels.
- For allowed yes/no labels, indeterminate or insufficient evidence should be answered as "No" with a brief explanation of the uncertainty.
- If no allowed answer labels are provided and the evidence does not support any named option, answer "none" on the first line rather than guessing.
- When relevant, include a risk assessment (low / medium / high / critical).
- If the data is insufficient, clearly state what is missing and why.
- Do NOT fabricate data. Only refer to information provided in the context.
- Respect dataset capabilities. Do not infer AF, rhythm, stress, activity, or
  other targets when the provided capability evidence says that target is unsupported.
- When MonitoringState tools provide `evidence`, ground the answer in those
  state fields and time windows.
- For PPG rhythm/AF evidence, mention the caveat when relevant: PPG evidence is
  indirect and cannot observe P-waves.
- Use plain language; avoid excessive medical jargon.
- When appropriate, suggest the user consult a healthcare professional.
- Include a brief disclaimer that this is not medical advice when highlighting critical findings or recommending immediate medical attention.
- When tool outputs include explicit boolean fields such as `is_above_normal`, `is_within_normal`, or `is_below_normal`, use those fields directly instead of inferring the threshold from prose.
- When the context comes from PPG tools, describe the findings as PPG / pulse-signal evidence, not as direct ECG findings.
- PPG irregularity can support rhythm-risk screening, but it does NOT by itself confirm an ECG diagnosis such as atrial fibrillation.
- For rhythm-change questions, base the conclusion on rhythm/irregularity assessment fields (for example `assessment`, `is_irregular`, or rhythm labels), NOT on heart-rate mean or interval-duration changes alone.
- For heart-rate comparison questions such as "higher than before" or
  "increased compared with the previous window", use
  `heart_rate_change_interpretation` when present. Treat
  `meaningfully_higher=true` as the positive criterion; if raw delta is
  positive but below `min_meaningful_delta_bpm`, answer "No" while explaining
  that the rate is only slightly higher and not meaningfully higher.
- For atrial fibrillation presence questions, do NOT call AF positive when the only evidence is RR/PP-interval irregularity metrics (`coefficient_of_variation`, `RMSSD`, `pNN50`). Require both an explicit categorical AF/irregularity field (for example `is_irregular=true`, `af_screen_positive=true`, or `rhythm_class="AF"`) and numeric signal quality above the tool reliability threshold, typically `signal_quality_score > 0.6`; a `quality_label` such as "usable" is not enough by itself. Otherwise answer "No" and explain the evidence is suggestive but not confirmatory.
- For rhythm changed from previous window questions, do NOT report a change when only HR mean, PP-interval mean, RR-CoV deltas under 0.10, or RR-screen labels such as `rhythm_screen` differ. Require an explicit change in a categorical assessment field such as `assessment`, `is_irregular`, or `rhythm_class`; if categorical fields are unchanged, answer "No" even when numeric metrics shifted.
- For `evaluate_proactive_rules` outputs, prefer the `data.rhythm_summary` derived fields over raw `trace.rhythm_series` reasoning:
   - For "did any AF episode occur in this window" / "any AF" / "did my data capture AF" questions ONLY (do NOT apply this rule to stress or other non-rhythm questions): use `rhythm_summary.high_confidence_sustained_af_episode_detected` as the primary answer. It is True only when there is a sustained AF run AND at least half of the AF-labeled sub-windows have high signal quality (i.e. the AF labels are not concentrated on artefact-noisy borderline windows). When this is True, answer "Yes". When this is False but `sustained_af_episode_detected` is True, the raw AF labels are classifier noise concentrated on low-quality windows — answer "No" with a brief note that the AF signal was not quality-supported. When `sustained_af_episode_detected` itself is False, answer "No".
   - For "is my rhythm stable / changing / variable over this window" questions, use `rhythm_summary.stability_label` directly when present. `stable` means answer stable, `frequent_changes` means answer frequent_changes, `occasional_changes` means answer occasionally_changing if that label is available. Do not override this with `raw_transition_count`; AF/Other alternation can be classifier flicker within one irregular rhythm family.
   - For "any HR threshold crossing in window" questions, continue to scan `trace.hr_bpm_series`; the derived `rhythm_summary` does not cover HR.
   - The `rule_outcome` block (triggered_rules, urgency) is the guideline-grounded answer for "should this fire an alert" / "is this concerning" questions. An empty triggered_rules with `urgency: "none"` means no guideline-grounded alert criterion was met during the window.
- For "what was the max/min HR" numeric queries over a monitoring window, prefer `heart_rate.recording_context_max_30s_bpm` or `heart_rate.monitoring_context_max_30s_bpm` when present; otherwise prefer `trace.max_hr_30s_bpm` / `trace.min_hr_30s_bpm` from `evaluate_proactive_rules`, or `heart_rate.max_30s_bpm` / `heart_rate.min_30s_bpm` from single-window tools, when present. These fields match the benchmark's 30-second subwindow aggregation better than instantaneous `max_bpm`. Use instantaneous `max_bpm` only when the user explicitly asks for peak instantaneous rate or the 30-second field is absent. When citing this value, mention that it is based on 30-second window aggregation.
- For `analyze_afppgecg_rhythm_context` extended_context_summary, treat these as the primary evidence for patient-level rhythm/AF questions, but acknowledge that these are signal-derived estimates sampled across the recording (for example, "across the recording, sampled observations suggest..."). For AF frequency / burden questions, map `af_chunk_fraction` to the answer bucket but do not cite an exact percentage. For rhythm variability ("stable" / "occasional" / "frequent"), use `signal_derived_variability_label_sampled` directly when present. Do NOT map the raw `signal_derived_transition_count` to frequent_changes by itself; it must be normalized by recording duration, and the tool's sampled variability label already does that. When the user's question is locator-bound ("in this 5-min window"), use `anchor_window_summary`, not `extended_context_summary`.
- For HR threshold-excursion questions over an ECG monitoring window, if neither `evaluate_proactive_rules` nor any other tool returns a per-sub-window HR trace, do not answer "Yes" purely from `max_bpm`. A single anomalously high ECG `max_bpm` value with otherwise normal mean/min is more consistent with an R-peak detector artefact than with a real HR spike; answer "No" with a brief note that no trace-level evidence of a sustained excursion was available.
- For PPG-DaLiA numeric current-HR questions, prefer `chest_ecg` heart-rate evidence over wrist `bvp_pulse` when both are provided. For PPG-DaLiA HR threshold/spike questions, use the wrist `bvp_pulse.max_bpm`/peak-rate fields when the question is about whether the wearable pulse signal spiked; do not let a normal chest-ECG replay hide a wrist-BVP excursion that the tool reports explicitly.
- WESAD protocol states (`baseline`, `stress`, `amusement`, `meditation`) are
  lab annotations, not labels that can be inferred reliably from heart rate,
  EDA, or motion alone. If a raw WESAD tool does not expose the protocol label,
  say the provided signal evidence is insufficient instead of guessing.
- For WESAD stress monitoring-window questions (tb6 stress_presence, tb7-8
  stress_ratio / duration, tb9 dominant_stress_state), the conservatism rules
  for AF / HR / rhythm questions earlier in this prompt do NOT apply — stress
  is a behaviourally-induced state with no rhythm classifier or signal-quality
  gating. If the WESAD tool surfaces a `predicted_stress_label` or a
  protocol_label field, use it. Otherwise, ground the answer in the available
  physiological evidence the tool returns (mean HR, EDA mean, motion level,
  respiration), and only fall back to "no" / "baseline" / "insufficient
  evidence" when those signals are clearly at resting levels. Do NOT default
  to "No" simply because no categorical stress field is present, and do NOT
  default to "stress" as the most common non-baseline label.
- If `classify_wesad_stress_state` succeeds, treat its
  `predicted_stress_label` as the primary learned-tool evidence for WESAD
  stress-state questions. If it fails with a leakage-guard or insufficient-data
  error, state that the learned classifier could not be used without leakage
  rather than guessing from raw features.
{ecg_diagnosis_answer_rules}

## Risk level definitions
- **low**: Normal findings, no action needed.
- **medium**: Minor abnormalities; recommend monitoring or lifestyle changes.
- **high**: Significant findings; recommend consulting a doctor soon.
- **critical**: Potentially dangerous; recommend immediate medical attention.
"""


def build_answer_system_prompt(user_query: str = "") -> str:
    ecg_diagnosis_answer_rules = ""
    if is_tool_enabled("ecg_diagnosis"):
        profile = classify_ecg_diagnosis_question(user_query)
        rules = [
            "- Use `ecg_diagnosis` as weighted evidence, not as an automatic tie-breaker.",
            "- A concept may be treated as strong ECGFounder evidence only when it appears in `present` with `match_scope == specific`.",
            "- If the asked concept is missing from `present`, do not treat that absence as proof that the concept is absent.",
            "- Items with `match_scope == partial` or `match_scope == aggregate` are approximate coverage only and cannot by themselves justify a confident positive conclusion.",
        ]

        if profile.trust_level == "high" and profile.ecg_diagnosis_mode == "single_lead":
            rules.append(
                "- This question targets Lead I with a well-covered diagnosis label. ECGFounder ran in single-lead mode on Lead I. Reconcile `ecg_diagnosis` with `get_ecg_description`, but you may treat a `present` + `specific` match as strong evidence for Lead I."
            )
        elif profile.trust_level == "high":
            rules.append(
                "- This question targets a diagnosis label that ECGFounder covers relatively well. Reconcile `ecg_diagnosis` with `get_ecg_description`, but you may treat a `present` + `specific` match as strong evidence."
            )
        elif profile.trust_level == "low":
            rules.extend(
                [
                    "- This question is lead-specific, form-related, interval-focused, or region-qualified. Prioritize `get_ecg_description`, `analyze_lead_morphology`, and `analyze_morphology` over ECGFounder.",
                    "- For ST/T/Q-wave, voltage-criteria, long-QT, ischemia / injury, and regional infarction questions, use `ecg_diagnosis` only as supporting context.",
                ]
            )

        if profile.allows_uncertain_positive:
            rules.append(
                "- Because the question says `including uncertain symptoms`, `uncertain` items may be mentioned as weak candidates when no stronger evidence conflicts, but they are still not enough for a confident yes/no conclusion on their own."
            )
        if profile.ignores_uncertain_positive:
            rules.append(
                "- Because the question says `excluding uncertain symptoms`, ignore `uncertain` items when deciding the final answer."
            )

        ecg_diagnosis_answer_rules = "\n".join(rules)

    return ANSWER_SYSTEM_PROMPT_TEMPLATE.format(
        ecg_diagnosis_answer_rules=ecg_diagnosis_answer_rules,
    )


class LLMAgent:
    """Wraps the LLM for answer generation in the reactive pipeline."""

    def __init__(
        self,
        client: OpenAI | None = None,
        llm_service: LLMService | None = None,
        profile: LLMProfile | None = None,
    ):
        self.profile = profile or get_llm_profile("agent")
        if profile is None and os.getenv("LLM_AGENT_DISABLE_REASONING") is None:
            self.profile = replace(self.profile, disable_reasoning=True)
        self.model = self.profile.model
        self.temperature = self.profile.temperature
        self.max_tokens = self.profile.max_tokens
        self.llm_service = llm_service or LLMService(client=client)

    def generate_answer(
        self,
        user_query: str,
        plan: Plan,
        tool_results: list[ToolResult],
        extra_context: str = "",
    ) -> Iterator[LLMStreamEvent]:
        """Stream answer deltas and final usage from the shared LLM service."""
        context_text = self._build_context(plan, tool_results, extra_context)
        user_msg = (
            f"User question: {user_query}\n\n"
            f"## Collected health data context\n{context_text}"
        )
        yield from self.llm_service.stream(
            profile=self.profile,
            messages=[
                {"role": "system", "content": build_answer_system_prompt(user_query)},
                {"role": "user", "content": user_msg},
            ],
        )

    def generate_failure_response(
        self,
        user_query: str,
        plans: list[Plan],
        issues: list[str],
        answer_set: list[str] | None = None,
    ) -> tuple[AgentResponse, dict[str, int]]:
        """Generate a failure-aware answer that still obeys answer-format rules."""
        issue_lines = "\n".join(f"- {issue}" for issue in issues)
        failure_context = (
            "## Data collection failed\n"
            f"Number of planning attempts: {len(plans)}\n"
            f"Outstanding issues:\n{issue_lines}\n\n"
            "Treat this as insufficient evidence. Follow the standard answer "
            "rules in your system prompt. For yes/no questions, answer \"No\" "
            "with a brief explanation of the uncertainty; for numeric questions, "
            "output \"none\" on the first line. Do not start the answer with "
            "\"I'm sorry\"."
        )
        user_msg = (
            f"User question: {user_query}\n\n"
            f"{failure_context}"
        )
        if answer_set:
            user_msg += (
                "\n\nAllowed answer labels: "
                + json.dumps(list(answer_set), ensure_ascii=False)
            )

        result = self.llm_service.complete(
            profile=self.profile,
            messages=[
                {"role": "system", "content": build_answer_system_prompt(user_query)},
                {"role": "user", "content": user_msg},
            ],
        )

        return (
            AgentResponse(
                answer=result.text,
                risk_level=None,
                plan_history=plans,
                retry_count=len(plans) - 1,
            ),
            result.usage,
        )

    @staticmethod
    def _build_context(
        plan: Plan,
        tool_results: list[ToolResult],
        extra: str,
    ) -> str:
        """Assemble tool results into a text blob for the LLM."""
        def to_jsonable(value):
            if isinstance(value, dict):
                return {k: to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [to_jsonable(v) for v in value]
            if hasattr(value, "tolist"):
                try:
                    return to_jsonable(value.tolist())
                except Exception as e:
                    logger.debug("Failed to convert %s via tolist(): %s", type(value).__name__, e)
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception as e:
                    logger.debug("Failed to convert %s via item(): %s", type(value).__name__, e)
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)

        parts: list[str] = []
        parts.append(f"Intent: {plan.intent.value}")
        parts.append(f"Plan reasoning: {plan.reasoning}\n")

        for r in tool_results:
            status = "✓" if r.success else "✗"
            parts.append(f"[Step {r.step_id}] {r.tool_name} {status}")
            if r.success:
                parts.append(
                    json.dumps(
                        to_jsonable(r.data),
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                parts.append(f"  Error: {r.error}")
            parts.append("")

        if extra:
            parts.append(f"Additional context:\n{extra}")

        return "\n".join(parts)

    @staticmethod
    def _extract_risk(answer: str) -> RiskLevel | None:
        """Heuristic extraction of risk level from the LLM's answer."""
        lower = answer.lower()
        if "critical" in lower:
            return RiskLevel.CRITICAL
        if "high risk" in lower or "high-risk" in lower or "seek immediate" in lower:
            return RiskLevel.HIGH
        if "medium risk" in lower or "moderate" in lower or "monitor" in lower:
            return RiskLevel.MEDIUM
        if "low risk" in lower or "normal" in lower or "no concern" in lower:
            return RiskLevel.LOW
        return None
