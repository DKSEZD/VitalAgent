from __future__ import annotations

from agent.llm import LLMProfile, LLMResult
from agent.reactive.llm_agent import LLMAgent
from agent.schemas import IntentType, Plan


class FakeLLMService:
    def __init__(self, text: str):
        self.text = text
        self.messages = None

    def complete(self, *, profile, messages):
        self.messages = messages
        return LLMResult(
            text=self.text,
            usage={
                "input_tokens": 1,
                "cached_input_tokens": 0,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        )


def test_generate_failure_response_preserves_allowed_yes_no_first_line() -> None:
    service = FakeLLMService("No\n\nExplanation: insufficient evidence.")
    agent = LLMAgent(
        llm_service=service,
        profile=LLMProfile("agent", "k", "https://example.com/v1", "demo-model", 0.0, 64),
    )
    plan = Plan(intent=IntentType.STATUS_QUERY, reasoning="test", steps=[])

    response, usage = agent.generate_failure_response(
        user_query="At this moment, is there a sign of AF in my data?",
        plans=[plan],
        issues=["The required tool failed validation."],
        answer_set=["yes", "no"],
    )

    assert response.answer.splitlines()[0].lower() in {"yes", "no"}
    assert usage["total_tokens"] == 2
    assert service.messages is not None
    assert "first line MUST be exactly one of those labels" in service.messages[0]["content"]
    assert "Allowed answer labels" in service.messages[1]["content"]
    assert "Do not start the answer with" in service.messages[1]["content"]
