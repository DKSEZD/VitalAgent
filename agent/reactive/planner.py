"""
Planner — analyses user query, identifies intent, and generates an execution
plan consisting of tool invocations.

Uses OpenAI-compatible LLM for structured plan generation.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.config import get_llm_profile
from agent.llm import LLMProfile, LLMResult, LLMService, extract_json_object_text, merge_usage, zero_usage
from agent.reactive.ecg_diagnosis_policy import classify_ecg_diagnosis_question
from agent.schemas import IntentType, Plan, PlanStep
from agent.tools.registry import get_tools_prompt, is_tool_enabled, list_tools

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt_template(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


PLANNER_SYSTEM_PROMPT_TEMPLATE = _load_prompt_template("planner_system.txt")
REPLAN_SYSTEM_PROMPT_TEMPLATE = _load_prompt_template("planner_replan.txt")


def _infer_signal_modality(record_id: str | None) -> str | None:
    if not record_id:
        return None
    text = str(record_id).strip().lower()
    if text.startswith("ppg:"):
        return "ppg"
    return "ecg"


def _ecg_diagnosis_prompt_sections() -> dict[str, str]:
    if not is_tool_enabled("ecg_diagnosis"):
        return {
            "ecg_diagnosis_tool_guidance": "",
            "ecg_diagnosis_anomaly_guidance": "",
            "ecg_diagnosis_lead_guidance": "",
            "ecg_diagnosis_diagnosis_guidance": "",
        }

    return {
        "ecg_diagnosis_tool_guidance": (
            "- `ecg_diagnosis`: Run ECGFounder, a pretrained ECG diagnosis model that returns\n"
            "  diagnosis-oriented labels and SCP-style mapped diagnostic findings. It is STRONGEST\n"
            "  for exact diagnosis labels such as atrial fibrillation / flutter, sinus rhythm,\n"
            "  sinus bradycardia / tachycardia, bundle branch block, fascicular block,\n"
            "  first-degree AV block, ventricular hypertrophy, atrial enlargement, normal ECG,\n"
            "  and digitalis effect.\n"
            "  It is WEAKER for lead-specific, form-related, or region-qualified findings such\n"
            "  as ST/T/Q-wave abnormalities, Q-wave presence, voltage-criteria questions,\n"
            "  long-QT questions, and regional ischemia / injury / infarction.\n"
            "  EXCEPTION: For Lead I with a well-covered diagnosis label, ECGFounder supports\n"
            "  a `single_lead` mode that runs inference on Lead I alone and remains strong evidence."
        ),
        "ecg_diagnosis_anomaly_guidance": (
            "   → + ecg_diagnosis for exact diagnosis-label questions that ECGFounder covers well\n"
            "     (arrhythmia, bundle branch block, fascicular block, AV block, hypertrophy,\n"
            "     atrial enlargement, normal ECG, digitalis effect)\n"
            "   → For lead-specific, form-related, long-QT, or region-qualified ischemia /\n"
            "     injury / infarction questions, PRIORITIZE `get_ecg_description` and the\n"
            "     morphology tools first; use ecg_diagnosis only as supporting context"
        ),
        "ecg_diagnosis_lead_guidance": (
            "   → Lead-specific or form-related questions should PRIORITIZE\n"
            "     `get_ecg_description` + `analyze_lead_morphology`\n"
            "   → Add ecg_diagnosis only as supporting context; do NOT rely on it as the sole\n"
            "     source for lead-specific ST/T/Q, voltage-criteria, or regional diagnosis labels\n"
            "   → EXCEPTION: If the question targets Lead I with a well-covered diagnosis label\n"
            "     (e.g. 'digitalis effect in lead I'), ecg_diagnosis can use single-lead mode\n"
            "     and serve as strong evidence"
        ),
        "ecg_diagnosis_diagnosis_guidance": (
            "   → For exact diagnosis labels that ECGFounder covers well, start with\n"
            "     ecg_diagnosis + get_ecg_description\n"
            "   → For lead-specific, form-related, long-QT, or region-qualified diagnosis labels,\n"
            "     start with get_ecg_description\n"
            "   → + analyze_lead_morphology for ST/T/Q-wave, voltage-criteria, and regional\n"
            "     pattern questions\n"
            "   → + analyze_morphology for long-QT or interval-focused questions\n"
            "   → ecg_diagnosis may be added as supporting context, but should not be the only\n"
            "     evidence source for weak-coverage label families"
        ),
    }


def _infer_record_id_from_steps(
    steps: list[PlanStep],
    active_record_id: str | None,
) -> str | None:
    if active_record_id:
        return active_record_id

    for step in steps:
        record_id = step.tool_args.get("record_id")
        if isinstance(record_id, str) and record_id:
            return record_id
    return None


def _step_description(tool_name: str, trust_level: str) -> str:
    if tool_name == "ecg_diagnosis":
        if trust_level == "high":
            return "Run ECGFounder for a strongly covered diagnosis label."
        return "Run ECGFounder as supporting context only."
    if tool_name == "get_ecg_description":
        return "Load the clinical report and SCP labels for diagnosis context."
    if tool_name == "analyze_lead_morphology":
        return "Inspect per-lead ST/T/Q-wave morphology for weak-coverage findings."
    if tool_name == "analyze_morphology":
        return "Measure PR/QRS/QT/QTc intervals for interval-focused findings."
    return f"Run {tool_name}."


def _prepare_policy_step(
    tool_name: str,
    existing: PlanStep | None,
    *,
    trust_level: str,
    record_id: str | None,
    extracted_leads: tuple[str, ...],
    ecg_diagnosis_mode: str = "12_lead",
) -> PlanStep:
    tool_args = dict(existing.tool_args) if existing is not None else {}
    description = existing.description if existing is not None else _step_description(
        tool_name,
        trust_level,
    )

    if record_id and not tool_args.get("record_id"):
        tool_args["record_id"] = record_id

    if tool_name == "ecg_diagnosis" and not tool_args.get("mode"):
        tool_args["mode"] = ecg_diagnosis_mode

    if tool_name == "analyze_lead_morphology" and extracted_leads and not tool_args.get(
        "leads"
    ):
        tool_args["leads"] = ",".join(extracted_leads)

    if tool_name == "analyze_morphology" and extracted_leads and not tool_args.get(
        "lead"
    ):
        tool_args["lead"] = extracted_leads[0]

    step_id = existing.step_id if existing is not None else 0
    return PlanStep(
        step_id=step_id,
        tool_name=tool_name,
        tool_args=tool_args,
        description=description,
    )


def _renumber_steps(steps: list[PlanStep]) -> list[PlanStep]:
    return [
        PlanStep(
            step_id=index,
            tool_name=step.tool_name,
            tool_args=dict(step.tool_args),
            description=step.description,
        )
        for index, step in enumerate(steps, start=1)
    ]


def _apply_ecg_diagnosis_policy(
    plan: Plan,
    user_query: str,
    *,
    active_record_id: str | None,
) -> Plan:
    if _infer_signal_modality(active_record_id) == "ppg":
        return plan
    if not is_tool_enabled("ecg_diagnosis"):
        return plan

    profile = classify_ecg_diagnosis_question(user_query)
    if not profile.is_relevant or profile.trust_level == "none":
        return plan

    targeted_names = {
        "ecg_diagnosis",
        "get_ecg_description",
        "analyze_lead_morphology",
        "analyze_morphology",
    }
    targeted_steps: dict[str, PlanStep] = {}
    remaining_steps: list[PlanStep] = []

    for step in plan.steps:
        if step.tool_name in targeted_names and step.tool_name not in targeted_steps:
            targeted_steps[step.tool_name] = step
            continue
        remaining_steps.append(step)

    record_id = _infer_record_id_from_steps(plan.steps, active_record_id)
    diagnosis_mode = profile.ecg_diagnosis_mode
    ordered_steps: list[PlanStep] = []

    if profile.trust_level == "high":
        ordered_steps.append(
            _prepare_policy_step(
                "ecg_diagnosis",
                targeted_steps.get("ecg_diagnosis"),
                trust_level=profile.trust_level,
                record_id=record_id,
                extracted_leads=profile.extracted_leads,
                ecg_diagnosis_mode=diagnosis_mode,
            )
        )
        ordered_steps.append(
            _prepare_policy_step(
                "get_ecg_description",
                targeted_steps.get("get_ecg_description"),
                trust_level=profile.trust_level,
                record_id=record_id,
                extracted_leads=profile.extracted_leads,
                ecg_diagnosis_mode=diagnosis_mode,
            )
        )
        if diagnosis_mode == "single_lead":
            reasoning_note = (
                "ECGFounder single-lead mode (Lead I) is used for this lead-specific "
                "diagnosis-label question."
            )
        else:
            reasoning_note = (
                "ECGFounder is treated as strong evidence only for well-covered diagnosis labels."
            )
    elif profile.trust_level == "low":
        ordered_steps.append(
            _prepare_policy_step(
                "get_ecg_description",
                targeted_steps.get("get_ecg_description"),
                trust_level=profile.trust_level,
                record_id=record_id,
                extracted_leads=profile.extracted_leads,
                ecg_diagnosis_mode=diagnosis_mode,
            )
        )
        if profile.needs_lead_morphology:
            ordered_steps.append(
                _prepare_policy_step(
                    "analyze_lead_morphology",
                    targeted_steps.get("analyze_lead_morphology"),
                    trust_level=profile.trust_level,
                    record_id=record_id,
                    extracted_leads=profile.extracted_leads,
                    ecg_diagnosis_mode=diagnosis_mode,
                )
            )
        if profile.needs_interval_morphology:
            ordered_steps.append(
                _prepare_policy_step(
                    "analyze_morphology",
                    targeted_steps.get("analyze_morphology"),
                    trust_level=profile.trust_level,
                    record_id=record_id,
                    extracted_leads=profile.extracted_leads,
                    ecg_diagnosis_mode=diagnosis_mode,
                )
            )
        if "ecg_diagnosis" in targeted_steps:
            ordered_steps.append(
                _prepare_policy_step(
                    "ecg_diagnosis",
                    targeted_steps.get("ecg_diagnosis"),
                    trust_level=profile.trust_level,
                    record_id=record_id,
                    extracted_leads=profile.extracted_leads,
                    ecg_diagnosis_mode=diagnosis_mode,
                )
            )
        reasoning_note = (
            "This lead/form/regional question is weakly covered by ECGFounder, so the plan "
            "prioritizes report and morphology evidence."
        )
    else:
        return plan

    updated_reasoning = plan.reasoning
    if reasoning_note not in updated_reasoning:
        updated_reasoning = f"{updated_reasoning} {reasoning_note}".strip()

    normalized_steps = _renumber_steps(ordered_steps + remaining_steps)
    return Plan(
        intent=plan.intent,
        reasoning=updated_reasoning,
        steps=normalized_steps,
        original_query=plan.original_query,
    )


def build_planner_system_prompt(*, tools_description: str, tool_names: str) -> str:
    return PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
        tools_description=tools_description,
        tool_names=tool_names,
        **_ecg_diagnosis_prompt_sections(),
    )


def build_replan_system_prompt(
    *,
    previous_plan: str,
    issues: str,
    tools_description: str,
    tool_names: str,
) -> str:
    return REPLAN_SYSTEM_PROMPT_TEMPLATE.format(
        previous_plan=previous_plan,
        issues=issues,
        tools_description=tools_description,
        tool_names=tool_names,
        **_ecg_diagnosis_prompt_sections(),
    )


class Planner:
    """Generates and revises execution plans via LLM structured output."""

    def __init__(
        self,
        client: OpenAI | None = None,
        llm_service: LLMService | None = None,
        profile: LLMProfile | None = None,
    ):
        self.profile = profile or get_llm_profile("planner")
        self.model = self.profile.model
        self.temperature = self.profile.temperature
        self.llm_service = llm_service or LLMService(client=client)

    def create_plan(
        self,
        user_query: str,
        active_record_id: str | None = None,
        extra_context: str = "",
    ) -> tuple[Plan, dict[str, int]]:
        """Generate an initial execution plan for the user query."""
        usage_total = zero_usage()
        tools_desc = get_tools_prompt()
        tool_names = self._allowed_tool_names_text()
        system = build_planner_system_prompt(
            tools_description=tools_desc,
            tool_names=tool_names,
        )

        user_msg = f"User question: {user_query}"
        if active_record_id:
            modality = _infer_signal_modality(active_record_id)
            if modality == "ppg":
                user_msg += f"\nActive PPG record ID: {active_record_id}"
            else:
                user_msg += f"\nActive ECG record ID, if applicable: {active_record_id}"
        if extra_context:
            user_msg += f"\nAdditional context:\n{extra_context}"

        plan, usage = self._generate_plan(
            system=system,
            user_msg=user_msg,
            user_query=user_query,
            plan_kind="plan",
            active_record_id=active_record_id,
        )
        usage_total = merge_usage(usage_total, usage)

        logger.info("Plan created: intent=%s, steps=%d", plan.intent, len(plan.steps))
        return plan, usage_total

    def replan(
        self,
        previous_plan: Plan,
        issues: list[str],
        user_query: str,
    ) -> tuple[Plan, dict[str, int]]:
        """Generate a revised plan after validation failure."""
        usage_total = zero_usage()
        tools_desc = get_tools_prompt()
        tool_names = self._allowed_tool_names_text()
        system = build_replan_system_prompt(
            previous_plan=previous_plan.model_dump_json(indent=2),
            issues=json.dumps(issues, ensure_ascii=False),
            tools_description=tools_desc,
            tool_names=tool_names,
        )
        user_msg = f"Original user question: {user_query}"

        plan, usage = self._generate_plan(
            system=system,
            user_msg=user_msg,
            user_query=user_query,
            plan_kind="replan",
            active_record_id=_infer_record_id_from_steps(previous_plan.steps, None),
        )
        usage_total = merge_usage(usage_total, usage)

        logger.info("Re-plan created: intent=%s, steps=%d", plan.intent, len(plan.steps))
        return plan, usage_total

    @staticmethod
    def _allowed_tool_names() -> set[str]:
        return {tool["name"] for tool in list_tools()}

    def _allowed_tool_names_text(self) -> str:
        return ", ".join(sorted(self._allowed_tool_names()))

    def _find_invalid_tool_names(self, plan_data: dict[str, Any]) -> set[str]:
        """Find invalid tool names before parsing can discard their steps."""
        allowed = self._allowed_tool_names()
        return {
            str(step.get("tool_name", ""))
            for step in plan_data.get("steps", [])
            if isinstance(step, dict)
            and step.get("tool_name")
            and step.get("tool_name") not in allowed
        }

    def _generate_plan(
        self,
        *,
        system: str,
        user_msg: str,
        user_query: str,
        plan_kind: str,
        active_record_id: str | None,
    ) -> tuple[Plan, dict[str, int]]:
        """Generate, validate tool names, retry once, then parse a plan."""
        usage_total = zero_usage()
        plan_data, usage = self._call_llm(system, user_msg)
        usage_total = merge_usage(usage_total, usage)

        invalid = self._find_invalid_tool_names(plan_data)
        if invalid:
            logger.warning(
                "Planner produced invalid tools in %s: %s; retrying once",
                plan_kind,
                invalid,
            )
            retry_user_msg = (
                user_msg
                + f"\n\nIMPORTANT: Your previous {plan_kind} used invalid tool names: "
                + ", ".join(sorted(invalid))
                + "\nPlease regenerate using ONLY allowed tool names."
            )
            plan_data, usage = self._call_llm(system, retry_user_msg)
            usage_total = merge_usage(usage_total, usage)

        plan = self._parse_plan(plan_data, user_query)
        plan = _apply_ecg_diagnosis_policy(
            plan,
            user_query,
            active_record_id=active_record_id,
        )
        return plan, usage_total

    def _dump_planner_raw(self, result: LLMResult, attempt: int, status: str) -> None:
        dump_dir = os.environ.get("PLANNER_RAW_DUMP_DIR")
        if not dump_dir:
            return

        try:
            now = datetime.now(timezone.utc)
            timestamp_utc = now.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            timestamp_file = timestamp_utc.replace("-", "").replace(":", "")
            repo_root = Path(__file__).resolve().parents[2]
            path = Path(dump_dir)
            if not path.is_absolute():
                path = repo_root / path
            path = path.resolve()
            try:
                rel_path = path.relative_to(repo_root)
                if not rel_path.parts or rel_path.parts[0] != "results":
                    path = repo_root / "results" / path.name
            except ValueError:
                pass
            path.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp_utc": timestamp_utc,
                "attempt": attempt,
                "status": status,
                "usage": dict(result.usage),
                "raw_text": result.text,
                "raw_length_chars": len(result.text),
                "reasoning_text": result.reasoning_text,
                "reasoning_length_chars": len(result.reasoning_text),
            }
            (path / f"planner_raw_{timestamp_file}_{status}_attempt{attempt}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to dump raw planner output: %s", exc)

    def _call_llm(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, int]]:
        """Call the LLM and parse the JSON response."""
        usage_total = zero_usage()
        retry_user = user
        raw = ""
        for attempt in range(2):
            result = self.llm_service.complete(
                profile=self.profile,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": retry_user},
                ],
            )
            usage_total = merge_usage(usage_total, result.usage)
            raw = result.text
            usage = dict(result.usage)
            suspicious_output = int(usage.get("output_tokens", 0) or 0) >= 3500

            try:
                data = json.loads(raw)
                status = "parsed"
            except json.JSONDecodeError:
                try:
                    data = json.loads(extract_json_object_text(raw))
                    status = "parsed_after_extract"
                except (ValueError, json.JSONDecodeError):
                    if suspicious_output or attempt == 1:
                        self._dump_planner_raw(result, attempt, "parse_failed")
                    if attempt == 0:
                        retry_user = (
                            user
                            + "\n\nIMPORTANT: Your previous planner output was invalid or truncated JSON. "
                            "Regenerate the plan as one compact JSON object only. Do not include markdown, "
                            "comments, or explanatory text outside the JSON."
                        )
                        continue
                    break

            if suspicious_output:
                self._dump_planner_raw(result, attempt, status)
            return data, usage_total

        logger.error("Planner LLM returned invalid JSON: %s", raw[:500])
        return (
            {
                "intent": "knowledge_qa",
                "reasoning": "[FALLBACK] Failed to parse planner JSON output; falling back to direct QA.",
                "steps": [],
            },
            usage_total,
        )

    def _parse_plan(self, data: dict[str, Any], original_query: str) -> Plan:
        """Convert raw dict into a validated Plan model."""
        intent_raw = data.get("intent", "knowledge_qa")
        try:
            intent = IntentType(intent_raw)
        except ValueError:
            intent = IntentType.KNOWLEDGE_QA

        allowed_tools = self._allowed_tool_names()
        steps: list[PlanStep] = []
        for s in data.get("steps", []):
            tool_name = s.get("tool_name", "")
            if tool_name and tool_name not in allowed_tools:
                logger.warning("Dropping invalid planner tool name: %s", tool_name)
                continue
            steps.append(
                PlanStep(
                    step_id=s.get("step_id", len(steps) + 1),
                    tool_name=tool_name,
                    tool_args=s.get("tool_args", {}),
                    description=s.get("description", ""),
                )
            )

        return Plan(
            intent=intent,
            reasoning=data.get("reasoning", ""),
            steps=steps,
            original_query=original_query,
        )
