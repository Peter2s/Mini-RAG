from typing import Literal, Protocol

InputType = Literal["document", "query"]


class EmbeddingProvider(Protocol):
    name: str

    async def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...

    async def aclose(self) -> None: ...
