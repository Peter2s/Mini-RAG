import asyncio
import json
from typing import Any

import anthropic
import httpx2
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMRefusalError
from app.providers.llm.anthropic_provider import AnthropicLLMProvider


def _provider(handler, **overrides) -> AnthropicLLMProvider:
    provider = AnthropicLLMProvider(Settings(_env_file=None, anthropic_api_key="k", **overrides))
    provider._client = anthropic.AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    return provider


def _message(stop_reason: str, text: str = "Paris [1].") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


def test_sends_expected_request_and_parses_text():
    seen = {}

    def handler(request):
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_message("end_turn"))

    result = asyncio.run(_provider(handler).generate(system="sys", prompt="q"))

    assert result.text == "Paris [1]."
    assert seen["beta"] == "server-side-fallback-2026-07-01"
    assert seen["body"]["model"] == "claude-opus-5-5"
    assert seen["body"]["fallbacks"] == "default"
    assert seen["body"]["output_config"] == {"effort": "medium"}
    assert seen["body"]["system"] == "sys"


def test_fallbacks_can_be_disabled():
    seen = {}

    def handler(request):
        seen["beta"] = request.headers.get("anthropic-beta")
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_message("end_turn"))

    asyncio.run(_provider(handler, llm_enable_fallbacks=False).generate(system="s", prompt="q"))
    assert seen["beta"] is None
    assert "fallbacks" not in seen["body"]


def test_refusal_raises():
    handler = lambda request: httpx2.Response(200, json=_message("refusal", text=""))  # noqa: E731
    with pytest.raises(LLMRefusalError):
        asyncio.run(_provider(handler).generate(system="s", prompt="q"))


def test_api_error_is_wrapped():
    handler = lambda request: httpx2.Response(  # noqa: E731
        500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}}
    )
    with pytest.raises(LLMError, match="500"):
        asyncio.run(_provider(handler).generate(system="s", prompt="q"))
