from typing import Protocol

from app.schemas.llm import LLMResponse


class LLMProvider(Protocol):
    async def generate(self, *, system: str, prompt: str) -> LLMResponse: ...

    async def aclose(self) -> None: ...
