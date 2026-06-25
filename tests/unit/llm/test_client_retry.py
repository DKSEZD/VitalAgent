from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai

from agent.llm import LLMProfile, LLMService
from agent.llm import client as client_module


class FakeCompletions:
    def __init__(self, outcomes: list[object]):
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes: list[object]):
        self.chat = SimpleNamespace(completions=FakeCompletions(outcomes))


class FirstChunkTimeout:
    def __iter__(self):
        return self

    def __next__(self):
        raise openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))


def _profile() -> LLMProfile:
    return LLMProfile("agent", "k", "https://example.com/v1", "demo-model", 0.0, 64)


def _response(text: str = "ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )


def _chunk(delta: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))],
        usage=None,
    )


def test_complete_retries_api_timeout_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))
    client = FakeClient([timeout, _response("recovered")])
    service = LLMService(client=client)

    result = service.complete(profile=_profile(), messages=[{"role": "user", "content": "hi"}])

    assert result.text == "recovered"
    assert len(client.chat.completions.calls) == 2


def test_stream_retries_timeout_before_first_chunk(monkeypatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    client = FakeClient([FirstChunkTimeout(), iter([_chunk("ok")])])
    service = LLMService(client=client)

    events = list(service.stream(profile=_profile(), messages=[{"role": "user", "content": "hi"}]))

    assert [event.delta for event in events] == ["ok"]
    assert len(client.chat.completions.calls) == 2


def test_stream_does_not_retry_after_delta_was_emitted(monkeypatch) -> None:
    monkeypatch.setattr(client_module.time, "sleep", lambda _seconds: None)
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "https://example.com"))

    def stream():
        yield _chunk("partial")
        raise timeout

    client = FakeClient([stream()])
    service = LLMService(client=client)
    iterator = service.stream(profile=_profile(), messages=[{"role": "user", "content": "hi"}])

    assert next(iterator).delta == "partial"
    try:
        next(iterator)
    except openai.APITimeoutError:
        pass
    else:
        raise AssertionError("Expected stream timeout after first delta")
    assert len(client.chat.completions.calls) == 1
