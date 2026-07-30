"""Frozen no-planner implementation used for the submitted paper results.

This module intentionally preserves the frozen ``paper_v1`` implementation:
random tools are drawn from the supplied global pool and the LLM fills arguments
with one call per selected tool. Extended-protocol batching, applicable-pool filtering,
and canonical locator overrides must not be added here.
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


RANDOM_TOOL_ARGS_SYSTEM_PROMPT = """\
You are filling arguments for a pre-selected health-data tool.

Rules:
- You MUST NOT change the tool name.
- Return only a JSON object with this exact shape: {"tool_args": {...}}.
- Use only argument names from the provided parameter schema.
- Use the provided benchmark locator exactly for dataset, patient_id,
  subject_id, recording_id, segment_id, window_start_s, and window_end_s.
- If a parameter is optional and not needed, omit it.
- If the schema is empty, return {"tool_args": {}}.
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


class PaperV1NoPlannerPlanner:
    """Planner-compatible frozen paper-v1 random-tool ablation."""

    def __init__(
        self,
        *,
        seed: int = 42,
        sample_key: str = "",
        tool_count: int = 1,
        tool_count_max: int | None = None,
        tool_pool_names: set[str] | None = None,
        client: OpenAI | None = None,
        llm_service: LLMService | None = None,
        profile: LLMProfile | None = None,
    ):
        self.seed = seed
        self.sample_key = sample_key
        self.tool_pool_names = tool_pool_names
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
        """Return the submitted-paper random-tool plan."""

        if self.tool_pool_names is not None:
            import agent.tools.registry as registry_mod

            tools = [
                _compact_tool_info(tool)
                for tool in registry_mod.list_tools()
                if tool["name"] in self.tool_pool_names
            ]
        else:
            tools = [_compact_tool_info(tool) for tool in planner_mod.list_tools()]
        tools = sorted(tools, key=lambda item: str(item.get("name")))
        if not tools:
            logger.warning("PaperV1NoPlannerPlanner found no allowed tools")
            return (
                Plan(
                    intent=IntentType.STATUS_QUERY,
                    reasoning="agent_no_planner found no allowed tools.",
                    steps=[],
                    original_query=user_query,
                ),
                zero_usage(),
            )

        rng = random.Random(_stable_seed(self.seed, self.sample_key or user_query))
        selected_count = rng.randint(self.tool_count_min, self.tool_count_max)
        selected = rng.sample(tools, k=min(selected_count, len(tools)))

        usage_total = zero_usage()
        steps: list[PlanStep] = []
        for step_id, tool in enumerate(selected, start=1):
            args, usage = self._generate_args(
                user_query=user_query,
                tool=tool,
                active_record_id=active_record_id,
                extra_context=extra_context,
            )
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0) or 0)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    tool_name=str(tool["name"]),
                    tool_args=args,
                    description=(
                        "agent_no_planner selected this tool; the LLM filled "
                        "only its arguments."
                    ),
                )
            )

        return (
            Plan(
                intent=IntentType.STATUS_QUERY,
                reasoning=(
                    f"agent_no_planner: selected {len(selected)} tool(s) "
                    "without the semantic planner, then asked "
                    "the LLM only for arguments."
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
        """Return the previous plan; the paper-v1 runner disables retries."""

        return previous_plan, zero_usage()

    def _generate_args(
        self,
        *,
        user_query: str,
        tool: dict[str, Any],
        active_record_id: str | None,
        extra_context: str,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        user_msg = (
            f"User question:\n{user_query}\n\n"
            f"Pre-selected tool:\n{_safe_json(tool)}\n\n"
            f"Active record ID, if applicable: {active_record_id or ''}\n\n"
            f"Benchmark locator / extra context:\n{extra_context or '{}'}"
        )
        result = self.llm_service.complete(
            profile=self.profile,
            messages=[
                {"role": "system", "content": RANDOM_TOOL_ARGS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        try:
            payload = json.loads(extract_json_object_text(result.text))
        except Exception:
            logger.warning(
                "PaperV1NoPlannerPlanner could not parse tool args JSON for %s: %r",
                tool.get("name"),
                result.text,
            )
            return {}, result.usage

        args = payload.get("tool_args", {})
        if not isinstance(args, dict):
            args = {}
        return args, result.usage
