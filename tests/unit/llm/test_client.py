from __future__ import annotations

from types import SimpleNamespace

from agent.config import reset_config
from agent.llm import (
    LLMProfile,
    LLMService,
    extract_json_object_text,
    merge_usage,
    zero_usage,
)
from agent.llm import client as client_module


class FakeCompletions:
    def __init__(
        self,
        response: object = None,
        stream_events: list[object] | None = None,
        errors: list[Exception] | None = None,
    ):
        self.response = response
        self.stream_events = stream_events or []
        self.errors = errors or []
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        if kwargs.get("stream"):
            return iter(self.stream_events)
        return self.response


class FakeClient:
    def __init__(
        self,
        response: object = None,
        stream_events: list[object] | None = None,
        errors: list[Exception] | None = None,
    ):
        self.chat = SimpleNamespace(completions=FakeCompletions(response, stream_events, errors))


class FakeBadRequest(Exception):
    status_code = 400


def _usage(prompt: int = 10, completion: int = 4, total: int = 14, cached: int = 2):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def test_complete_returns_text_and_usage() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=_usage(),
    )
    client = FakeClient(response=response)
    service = LLMService(client=client)
    profile = LLMProfile("judge", "k", "https://example.com/v1", "demo-model", 0.0, 64)

    result = service.complete(
        profile=profile,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.text == "hello"
    assert result.usage == {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    assert client.chat.completions.calls[0]["model"] == "demo-model"


def test_complete_captures_reasoning_content_without_changing_text() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="final answer",
                    reasoning_content="private reasoning",
                )
            )
        ],
        usage=_usage(),
    )
    client = FakeClient(response=response)
    service = LLMService(client=client)
    profile = LLMProfile("planner", "k", "https://example.com/v1", "demo-model", 0.0, 64)

    result = service.complete(
        profile=profile,
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result.text == "final answer"
    assert result.reasoning_text == "private reasoning"


def test_disable_reasoning_adds_extra_body_to_completion_request() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=_usage(),
    )
    client = FakeClient(response=response)
    service = LLMService(client=client)
    profile = LLMProfile(
        "agent",
        "k",
        "https://example.com/v1",
        "demo-model",
        0.0,
        64,
        disable_reasoning=True,
    )

    service.complete(profile=profile, messages=[{"role": "user", "content": "hi"}])

    extra_body = client.chat.completions.calls[0]["extra_body"]
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_disable_reasoning_adds_deepseek_thinking_toggle() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=_usage(),
    )
    client = FakeClient(response=response)
    service = LLMService(client=client)
    profile = LLMProfile(
        "agent",
        "k",
        "https://api.deepseek.com/v1",
        "deepseek-v4-flash",
        0.0,
        64,
        disable_reasoning=True,
    )

    service.complete(profile=profile, messages=[{"role": "user", "content": "hi"}])

    extra_body = client.chat.completions.calls[0]["extra_body"]
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra_body["thinking"] == {"type": "disabled"}


def test_disable_reasoning_false_omits_extra_body_from_completion_request() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=_usage(),
    )
    client = FakeClient(response=response)
    service = LLMService(client=client)
    profile = LLMProfile(
        "planner",
        "k",
        "https://example.com/v1",
        "demo-model",
        0.0,
        64,
        disable_reasoning=False,
    )

    service.complete(profile=profile, messages=[{"role": "user", "content": "hi"}])

    assert "extra_body" not in client.chat.completions.calls[0]


def test_disable_reasoning_bad_request_retries_without_extra_body() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=_usage(),
    )
    client = FakeClient(response=response, errors=[FakeBadRequest("unsupported extra_body")])
    service = LLMService(client=client)
    profile = LLMProfile(
        "agent",
        "k",
        "https://example.com/v1",
        "demo-model",
        0.0,
        64,
        disable_reasoning=True,
    )

    result = service.complete(profile=profile, messages=[{"role": "user", "content": "hi"}])

    assert result.text == "hello"
    assert len(client.chat.completions.calls) == 2
    assert "extra_body" in client.chat.completions.calls[0]
    assert "extra_body" not in client.chat.completions.calls[1]


def test_stream_returns_delta_events_and_final_usage() -> None:
    stream_events = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
            usage=_usage(prompt=8, completion=3, total=11, cached=1),
        ),
    ]
    client = FakeClient(stream_events=stream_events)
    service = LLMService(client=client)
    profile = LLMProfile("agent", "k", "https://example.com/v1", "demo-model", 0.2, 128)

    events = list(
        service.stream(
            profile=profile,
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    assert [event.delta for event in events] == ["Hel", "lo", ""]
    assert events[-1].usage == {
        "input_tokens": 8,
        "cached_input_tokens": 1,
        "output_tokens": 3,
        "total_tokens": 11,
    }


def test_client_cache_reuses_same_profile_connection() -> None:
    created: list[tuple[str, str]] = []

    def factory(api_key: str, base_url: str):
        created.append((api_key, base_url))
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=_usage(),
        )
        return FakeClient(response=response)

    service = LLMService(client_factory=factory)
    profile = LLMProfile("planner", "k", "https://example.com/v1", "demo-model", 0.0, 32)

    service.complete(profile=profile, messages=[{"role": "user", "content": "one"}])
    service.complete(profile=profile, messages=[{"role": "user", "content": "two"}])

    assert created == [("k", "https://example.com/v1")]


def test_explicit_client_bypasses_factory_cache() -> None:
    created: list[tuple[str, str]] = []

    def factory(api_key: str, base_url: str):
        created.append((api_key, base_url))
        return FakeClient()

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=_usage(),
    )
    service = LLMService(client=FakeClient(response=response), client_factory=factory)
    profile = LLMProfile("extractor", "k", "https://example.com/v1", "demo-model", 0.0, 32)

    service.complete(profile=profile, messages=[{"role": "user", "content": "hi"}])

    assert created == []


def test_default_client_factory_uses_transport_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setenv("LLM_CONNECT_TIMEOUT_S", "4")
    monkeypatch.setenv("LLM_READ_TIMEOUT_S", "90")
    monkeypatch.setenv("LLM_WRITE_TIMEOUT_S", "45")
    monkeypatch.setenv("LLM_POOL_TIMEOUT_S", "30")
    monkeypatch.setenv("LLM_SDK_MAX_RETRIES", "1")
    monkeypatch.setattr(client_module, "OpenAI", fake_openai)
    reset_config()

    service = LLMService()
    profile = LLMProfile("agent", "k", "https://example.com/v1", "demo", 0.0, 32)
    service._get_client(profile)

    timeout = captured["timeout"]
    assert timeout.connect == 4.0
    assert timeout.read == 90.0
    assert timeout.write == 45.0
    assert timeout.pool == 30.0
    assert captured["max_retries"] == 1
    reset_config()


def test_json_extract_utility_handles_fenced_payload() -> None:
    text = '```json\n{"verdict":"correct","reason":"ok"}\n```'
    assert extract_json_object_text(text) == '{"verdict":"correct","reason":"ok"}'


def test_merge_usage_adds_counts() -> None:
    assert merge_usage(
        {"input_tokens": 1, "cached_input_tokens": 2, "output_tokens": 3, "total_tokens": 4},
        {"input_tokens": 5, "cached_input_tokens": 6, "output_tokens": 7, "total_tokens": 8},
    ) == {
        "input_tokens": 6,
        "cached_input_tokens": 8,
        "output_tokens": 10,
        "total_tokens": 12,
    }
    assert zero_usage()["total_tokens"] == 0
