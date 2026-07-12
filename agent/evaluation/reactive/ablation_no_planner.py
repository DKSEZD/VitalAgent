"""No-planner Agent ablation for VitalBench evaluation.

This ablation keeps the Agent execution shape, but removes semantic planning:
tools are selected without the planner, and the LLM is only allowed to fill
arguments for the selected tool. It may not choose a different tool.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any

from openai import OpenAI

from agent.config import get_llm_profile
from agent.llm import LLMProfile, LLMService, extract_json_object_text, zero_usage
from agent.reactive import planner as planner_mod
from agent.schemas import IntentType, Plan, PlanStep

logger = logging.getLogger(__name__)


BATCHED_TOOL_ARGS_SYSTEM_PROMPT = """\
You are filling arguments for a pre-selected list of health-data tools.

Rules:
- You MUST NOT add, remove, rename, or duplicate tools.
- Return only a JSON object with this exact shape:
  {"tool_calls": [{"tool_name": "...", "tool_args": {...}}, ...]}.
- Use only argument names from the provided parameter schema.
- Use the provided benchmark locator exactly for dataset, patient_id,
  subject_id, recording_id, segment_id, record_id, window_start_s,
  window_end_s, anchor_window_start_s, and anchor_window_end_s.
- If a parameter is optional and not needed, omit it.
- Include one entry for every pre-selected tool, in the given order.
"""


def _stable_seed(seed: int, key: str) -> int:
    payload = f"{seed}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _compact_tool_info(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name"),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters") or {},
    }


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class NoPlannerPlanner:
    """Planner-compatible object for the agent_no_planner ablation."""

    def __init__(
        self,
        *,
        seed: int = 42,
        sample_key: str = "",
        tool_count: int = 1,
        tool_count_max: int | None = None,
        tool_pool_names: set[str] | None = None,
        selection_mode: str = "random",
        canonical_context: dict[str, Any] | None = None,
        client: OpenAI | None = None,
        llm_service: LLMService | None = None,
        profile: LLMProfile | None = None,
    ):
        self.seed = seed
        self.sample_key = sample_key
        self.tool_pool_names = tool_pool_names
        if selection_mode not in {"random", "none", "all"}:
            raise ValueError(f"Unsupported selection_mode: {selection_mode!r}")
        self.selection_mode = selection_mode
        self.canonical_context = dict(canonical_context or {})
        self.tool_count_min = max(1, int(tool_count))
        self.tool_count_max = max(
            self.tool_count_min,
            int(tool_count_max) if tool_count_max is not None else self.tool_count_min,
        )
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
        """Return a no-planner tool plan with LLM-filled arguments."""

        tools = [
            _compact_tool_info(tool)
            for tool in planner_mod.list_tools()
            if self.tool_pool_names is None or tool["name"] in self.tool_pool_names
        ]
        tools = sorted(
            tools,
            key=lambda item: str(item.get("name")),
        )
        if self.selection_mode == "none":
            return (
                Plan(
                    intent=IntentType.STATUS_QUERY,
                    reasoning="No-tools baseline: answer without tool-computed evidence.",
                    steps=[],
                    original_query=user_query,
                ),
                zero_usage(),
            )

        if not tools:
            logger.warning("NoPlannerPlanner found no allowed tools")
            return (
                Plan(
                    intent=IntentType.STATUS_QUERY,
                    reasoning="agent_no_planner found no allowed tools.",
                    steps=[],
                    original_query=user_query,
                ),
                zero_usage(),
            )

        if self.selection_mode == "all":
            selected = tools
        else:
            rng = random.Random(_stable_seed(self.seed, self.sample_key or user_query))
            selected_count = rng.randint(self.tool_count_min, self.tool_count_max)
            selected = rng.sample(tools, k=min(selected_count, len(tools)))

        args_by_tool, usage_total = self._generate_args_batch(
            user_query=user_query,
            tools=selected,
            active_record_id=active_record_id,
            extra_context=extra_context,
        )
        steps: list[PlanStep] = []
        for step_id, tool in enumerate(selected, start=1):
            steps.append(
                PlanStep(
                    step_id=step_id,
                    tool_name=str(tool["name"]),
                    tool_args=args_by_tool[str(tool["name"])],
                    description=(
                        f"No-planner {self.selection_mode} strategy pre-selected "
                        "this tool; the LLM filled only its arguments."
                    ),
                )
            )

        return (
            Plan(
                intent=IntentType.STATUS_QUERY,
                reasoning=(
                    f"No-planner {self.selection_mode} strategy selected "
                    f"{len(selected)} tool(s), then used one batched LLM call "
                    "only for arguments."
                ),
                steps=steps,
                original_query=user_query,
            ),
            usage_total,
        )

    def replan(
        self,
        previous_plan: Plan,
        issues: list[str],
        user_query: str,
    ) -> tuple[Plan, dict[str, int]]:
        """No-planner ablation does not re-plan; the eval runner disables retries."""

        return previous_plan, zero_usage()

    def _canonical_args(
        self,
        *,
        tool: dict[str, Any],
        active_record_id: str | None,
    ) -> dict[str, Any]:
        allowed = set((tool.get("parameters") or {}).keys())
        context = self.canonical_context
        window_start_s = context.get("window_start_s")
        window_end_s = context.get("window_end_s")
        patient_id = context.get("patient_id") or context.get("subject_id")
        candidates = {
            "dataset": context.get("dataset"),
            "patient_id": patient_id,
            "subject_id": patient_id,
            "recording_id": context.get("recording_id"),
            "segment_id": context.get("segment_id") or context.get("recording_id"),
            "record_id": active_record_id,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "anchor_window_start_s": window_start_s,
            "anchor_window_end_s": window_end_s,
        }
        return {
            key: value
            for key, value in candidates.items()
            if key in allowed and value is not None
        }

    def _generate_args_batch(
        self,
        *,
        user_query: str,
        tools: list[dict[str, Any]],
        active_record_id: str | None,
        extra_context: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
        user_msg = (
            f"User question:\n{user_query}\n\n"
            f"Pre-selected tools:\n{_safe_json(tools)}\n\n"
            f"Active record ID, if applicable: {active_record_id or ''}\n\n"
            f"Benchmark locator / extra context:\n{extra_context or '{}'}"
        )
        result = self.llm_service.complete(
            profile=self.profile,
            messages=[
                {"role": "system", "content": BATCHED_TOOL_ARGS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        parsed_calls: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(extract_json_object_text(result.text))
            calls = payload.get("tool_calls", [])
            if not isinstance(calls, list):
                calls = []
            selected_names = {str(tool["name"]) for tool in tools}
            for item in calls:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("tool_name") or "")
                if name not in selected_names or name in parsed_calls:
                    continue
                args = item.get("tool_args", {})
                parsed_calls[name] = args if isinstance(args, dict) else {}
        except Exception:
            logger.warning(
                "NoPlannerPlanner could not parse batched tool args JSON: %r",
                result.text,
            )

        args_by_tool: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = str(tool["name"])
            allowed = set((tool.get("parameters") or {}).keys())
            generated = {
                key: value
                for key, value in parsed_calls.get(name, {}).items()
                if key in allowed
            }
            generated.update(
                self._canonical_args(tool=tool, active_record_id=active_record_id)
            )
            args_by_tool[name] = generated
        return args_by_tool, result.usage
