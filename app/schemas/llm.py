from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    stop_reason: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
