import logging

import anthropic

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMRefusalError
from app.providers.llm.base import LLMResponse

logger = logging.getLogger(__name__)

# `fallbacks: "default"` re-runs a request that a safety classifier declined on a
# recommended fallback model, server-side, inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMProvider:
    """Generates answers with Claude through the official Anthropic SDK."""

    def __init__(self, settings: Settings) -> None:
        client_kwargs = {}
        if settings.anthropic_api_key:
            client_kwargs["api_key"] = settings.anthropic_api_key.get_secret_value()
        # Without an explicit key the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile.
        self._client = anthropic.AsyncAnthropic(**client_kwargs)
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._effort = settings.llm_effort
        self._enable_fallbacks = settings.llm_enable_fallbacks

    async def generate(self, *, system: str, prompt: str) -> LLMResponse:
        request = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self._effort},
        }
        try:
            if self._enable_fallbacks:
                response = await self._client.beta.messages.create(
                    **request, betas=[_FALLBACK_BETA], fallbacks="default"
                )
            else:
                response = await self._client.messages.create(**request)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic rate limit reached. Retry later.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Anthropic rejected the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.") from exc
        except TypeError as exc:
            # The SDK raises TypeError when no API key, auth token, or profile is configured.
            if "authentication method" not in str(exc):
                raise
            raise LLMError(
                "No Anthropic credentials configured. "
                "Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMRefusalError(f"The model declined to answer (category: {category}).")
        if response.stop_reason == "max_tokens":
            logger.warning("Answer truncated at max_tokens=%s", self._max_tokens)

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            model=response.model,
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def aclose(self) -> None:
        await self._client.close()
