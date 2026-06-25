"""Shared OpenAI-compatible LLM client utilities."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from itertools import chain
from typing import Callable, Iterator

import openai
from httpx import Timeout
from openai import OpenAI

logger = logging.getLogger(__name__)

_STREAM_EXHAUSTED = object()


@dataclass(frozen=True)
class LLMProfile:
    """Resolved role-specific model profile."""

    role: str
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    disable_reasoning: bool = False


@dataclass(frozen=True)
class LLMResult:
    """Non-streaming LLM response text and token usage."""

    text: str
    usage: dict[str, int]
    reasoning_text: str = ""


@dataclass(frozen=True)
class LLMStreamEvent:
    """Incremental stream update emitted by the shared LLM service."""

    delta: str = ""
    usage: dict[str, int] | None = None
    reasoning_delta: str = ""
    reasoning_text: str = ""


def zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    merged = zero_usage()
    for key in merged:
        merged[key] = int(left.get(key, 0) or 0) + int(right.get(key, 0) or 0)
    return merged


def usage_from_openai(usage_obj: object | None) -> dict[str, int]:
    usage = zero_usage()
    if usage_obj is None:
        return usage

    prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)

    details = getattr(usage_obj, "prompt_tokens_details", None)
    cached_input_tokens = 0
    if details is not None:
        if isinstance(details, dict):
            cached_input_tokens = int(details.get("cached_tokens", 0) or 0)
        else:
            cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)

    usage["input_tokens"] = prompt_tokens
    usage["cached_input_tokens"] = cached_input_tokens
    usage["output_tokens"] = completion_tokens
    usage["total_tokens"] = int(getattr(usage_obj, "total_tokens", 0) or 0)
    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _response_field(obj: object, name: str) -> str:
    if isinstance(obj, dict):
        return obj.get(name) or ""
    return (
        getattr(obj, name, None)
        or (getattr(obj, "model_extra", {}) or {}).get(name)
        or ""
    )


def extract_json_object_text(text: str) -> str:
    """Extract the most likely JSON object block from model output."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot extract JSON from an empty response.")

    candidates = [stripped]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidates.extend(fenced)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    raise ValueError(f"Could not locate a JSON object in response: {text!r}")


class LLMService:
    """Shared OpenAI-compatible chat wrapper with client reuse."""

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        client_factory: Callable[[str, str], OpenAI] | None = None,
        client_cache: dict[tuple[str, str], OpenAI] | None = None,
    ):
        from agent.config import get_config

        cfg = get_config().llm
        self._explicit_client = client
        self._client_factory = client_factory or (
            lambda api_key, base_url: OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=Timeout(
                    connect=cfg.connect_timeout_s,
                    read=cfg.read_timeout_s,
                    write=cfg.write_timeout_s,
                    pool=cfg.pool_timeout_s,
                ),
                max_retries=cfg.sdk_max_retries,
            )
        )
        self._client_cache = client_cache if client_cache is not None else {}
        self._transient_retry_attempts = max(0, cfg.transient_retry_attempts)
        self._transient_retry_backoff_s = cfg.transient_retry_backoff_s
        self._disable_reasoning_warning_logged = False

    def _get_client(self, profile: LLMProfile) -> OpenAI:
        if self._explicit_client is not None:
            return self._explicit_client

        cache_key = (profile.api_key, profile.base_url)
        client = self._client_cache.get(cache_key)
        if client is None:
            client = self._client_factory(profile.api_key, profile.base_url)
            self._client_cache[cache_key] = client
        return client

    @staticmethod
    def _base_request_kwargs(
        *,
        profile: LLMProfile,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "model": profile.model,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "messages": messages,
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if profile.disable_reasoning:
            extra_body: dict[str, object] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
            if "deepseek" in profile.base_url.lower() or profile.model.lower().startswith("deepseek-"):
                extra_body["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = extra_body
        return kwargs

    @staticmethod
    def _is_extra_body_bad_request(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        return status_code == 400

    @staticmethod
    def _is_transient_error(exc: Exception) -> bool:
        transient_types = (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        )
        if isinstance(exc, transient_types):
            return True

        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True

        message = str(exc).lower()
        return "timed out" in message or "timeout" in message

    def _call_with_transient_retries(
        self,
        *,
        profile: LLMProfile,
        operation: Callable[[], object],
    ) -> object:
        for attempt in range(self._transient_retry_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                if (
                    not self._is_transient_error(exc)
                    or attempt >= self._transient_retry_attempts
                ):
                    raise
                logger.warning(
                    "LLM transient error on role=%s attempt=%d/%d: %s",
                    profile.role,
                    attempt + 1,
                    self._transient_retry_attempts,
                    exc,
                )
                backoff_index = min(attempt, len(self._transient_retry_backoff_s) - 1)
                time.sleep(self._transient_retry_backoff_s[backoff_index])

        raise RuntimeError("unreachable transient retry state")

    def _create_chat_completion(
        self,
        *,
        profile: LLMProfile,
        kwargs: dict[str, object],
    ):
        client = self._get_client(profile)
        try:
            return self._call_with_transient_retries(
                profile=profile,
                operation=lambda: client.chat.completions.create(**kwargs),
            )
        except Exception as exc:
            if profile.disable_reasoning and "extra_body" in kwargs and self._is_extra_body_bad_request(exc):
                if not self._disable_reasoning_warning_logged:
                    logger.warning(
                        "LLM server rejected disable_reasoning extra_body for role=%s model=%s; "
                        "retrying without it.",
                        profile.role,
                        profile.model,
                    )
                    self._disable_reasoning_warning_logged = True
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("extra_body", None)
                return self._call_with_transient_retries(
                    profile=profile,
                    operation=lambda: client.chat.completions.create(**fallback_kwargs),
                )
            raise

    def _create_stream_chat_completion(
        self,
        *,
        profile: LLMProfile,
        kwargs: dict[str, object],
    ) -> tuple[Iterator[object], object]:
        client = self._get_client(profile)

        def start_stream(stream_kwargs: dict[str, object]) -> tuple[Iterator[object], object]:
            stream = client.chat.completions.create(**stream_kwargs)
            iterator = iter(stream)
            try:
                first_chunk = next(iterator)
            except StopIteration:
                first_chunk = _STREAM_EXHAUSTED
            return iterator, first_chunk

        try:
            return self._call_with_transient_retries(
                profile=profile,
                operation=lambda: start_stream(kwargs),
            )
        except Exception as exc:
            if profile.disable_reasoning and "extra_body" in kwargs and self._is_extra_body_bad_request(exc):
                if not self._disable_reasoning_warning_logged:
                    logger.warning(
                        "LLM server rejected disable_reasoning extra_body for role=%s model=%s; "
                        "retrying without it.",
                        profile.role,
                        profile.model,
                    )
                    self._disable_reasoning_warning_logged = True
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("extra_body", None)
                return self._call_with_transient_retries(
                    profile=profile,
                    operation=lambda: start_stream(fallback_kwargs),
                )
            raise

    def complete(
        self,
        *,
        profile: LLMProfile,
        messages: list[dict[str, str]],
    ) -> LLMResult:
        try:
            response = self._create_chat_completion(
                profile=profile,
                kwargs=self._base_request_kwargs(profile=profile, messages=messages, stream=False),
            )
        except Exception:
            logger.exception("LLM completion failed for role=%s model=%s", profile.role, profile.model)
            raise

        msg = response.choices[0].message
        text = _response_field(msg, "content")
        reasoning_text = _response_field(msg, "reasoning_content")
        if reasoning_text:
            logger.debug(
                "LLM reasoning_content captured: %d chars, content_chars=%d",
                len(reasoning_text),
                len(text),
            )

        return LLMResult(
            text=text,
            usage=usage_from_openai(getattr(response, "usage", None)),
            reasoning_text=reasoning_text,
        )

    def stream(
        self,
        *,
        profile: LLMProfile,
        messages: list[dict[str, str]],
    ) -> Iterator[LLMStreamEvent]:
        try:
            stream, first_chunk = self._create_stream_chat_completion(
                profile=profile,
                kwargs=self._base_request_kwargs(profile=profile, messages=messages, stream=True),
            )
        except Exception:
            logger.exception("LLM stream failed for role=%s model=%s", profile.role, profile.model)
            raise

        text = ""
        reasoning_text = ""
        logged_reasoning = False
        chunks: Iterator[object]
        if first_chunk is _STREAM_EXHAUSTED:
            chunks = iter(())
        else:
            chunks = chain((first_chunk,), stream)

        for chunk in chunks:
            usage = None
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = usage_from_openai(chunk_usage)

            delta = ""
            reasoning_delta = ""
            if chunk.choices and chunk.choices[0].delta:
                chunk_delta = chunk.choices[0].delta
                delta = _response_field(chunk_delta, "content")
                reasoning_delta = _response_field(chunk_delta, "reasoning_content")
                text += delta
                reasoning_text += reasoning_delta

            if reasoning_text and usage is not None and not logged_reasoning:
                logger.debug(
                    "LLM reasoning_content captured: %d chars, content_chars=%d",
                    len(reasoning_text),
                    len(text),
                )
                logged_reasoning = True

            if delta or reasoning_delta or usage is not None:
                yield LLMStreamEvent(
                    delta=delta,
                    usage=usage,
                    reasoning_delta=reasoning_delta,
                    reasoning_text=reasoning_text if usage is not None else "",
                )

        if reasoning_text and not logged_reasoning:
            logger.debug(
                "LLM reasoning_content captured: %d chars, content_chars=%d",
                len(reasoning_text),
                len(text),
            )


__all__ = [
    "LLMProfile",
    "LLMResult",
    "LLMService",
    "LLMStreamEvent",
    "extract_json_object_text",
    "merge_usage",
    "usage_from_openai",
    "zero_usage",
]
