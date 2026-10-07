import logging

import anthropic
from anthropic.types import Message
from anthropic.types.beta import BetaMessage

from app.core.config import Settings
from app.core.exceptions import LLMError, LLMRefusalError
from app.schemas.llm import LLMResponse

logger = logging.getLogger(__name__)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLMProvider:
    """Generates answers with Claude through the official Anthropic SDK."""

    def __init__(self, settings: Settings) -> None:

        self._client = (
            anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
            if settings.anthropic_api_key
            else anthropic.AsyncAnthropic()
        )
        self._model = settings.llm_model
        self._max_tokens = settings.llm_max_tokens
        self._effort = settings.llm_effort
        self._enable_fallbacks = settings.llm_enable_fallbacks

    async def generate(self, *, system: str, prompt: str) -> LLMResponse:
        try:
            response = await self._create(system=system, prompt=prompt)
        except anthropic.APIError as error:
            raise _to_llm_error(error) from error
        except TypeError as error:
            if "authentication method" not in str(error):
                raise
            raise LLMError(
                "No Anthropic credentials configured. "
                "Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from error

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

    async def _create(self, *, system: str, prompt: str) -> Message | BetaMessage:
        if self._enable_fallbacks:
            return await self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": self._effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        return await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self._effort},
        )

    async def aclose(self) -> None:
        await self._client.close()


def _to_llm_error(error: anthropic.APIError) -> LLMError:
    """Map SDK errors to a message the API can show, most specific first."""
    if isinstance(error, anthropic.AuthenticationError):
        return LLMError("Anthropic authentication failed. Check ANTHROPIC_API_KEY.")
    if isinstance(error, anthropic.RateLimitError):
        return LLMError("Anthropic rate limit reached. Retry later.")
    if isinstance(error, anthropic.BadRequestError):
        return LLMError(f"Anthropic rejected the request: {error.message}")
    if isinstance(error, anthropic.APIStatusError):
        return LLMError(f"Anthropic API error ({error.status_code}): {error.message}")
    if isinstance(error, anthropic.APIConnectionError):
        return LLMError("Could not reach the Anthropic API.")
    return LLMError(f"Anthropic API error: {error.message}")
