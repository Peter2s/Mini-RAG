from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    async def generate(self, *, system: str, prompt: str) -> LLMResponse: ...

    async def aclose(self) -> None: ...
