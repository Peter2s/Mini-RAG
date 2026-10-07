import hashlib
import math
import re
from itertools import pairwise

from app.providers.embeddings.base import EmbeddingProvider

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashEmbeddingProvider(EmbeddingProvider):
    """Offline, dependency-free embeddings via signed feature hashing of words and bigrams.

    Good for local development and tests: deterministic, no API key, captures lexical overlap.
    It has no semantic understanding, so switch to a real model (e.g. Voyage) for production.
    """

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"hash:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        tokens = [token.lower() for token in _TOKEN.findall(text)]
        features = tokens + [f"{first}_{second}" for first, second in pairwise(tokens)]
        vector = [0.0] * self._dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None
