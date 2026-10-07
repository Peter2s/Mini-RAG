"""Every embedding provider implements the EmbeddingProvider interface."""

import asyncio

import pytest

from app.providers.embeddings import EmbeddingProvider
from app.providers.embeddings.fastembed_provider import FastEmbedProvider
from app.providers.embeddings.hash_provider import HashEmbeddingProvider
from app.providers.embeddings.voyage_provider import VoyageEmbeddingProvider


@pytest.mark.parametrize(
    "implementation", [FastEmbedProvider, HashEmbeddingProvider, VoyageEmbeddingProvider]
)
def test_providers_implement_the_interface(implementation):
    assert issubclass(implementation, EmbeddingProvider)
    assert not implementation.__abstractmethods__


def test_interface_cannot_be_instantiated_without_every_method():
    class Incomplete(EmbeddingProvider):
        async def embed_query(self, text: str) -> list[float]:
            return []

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_hash_provider_identity_depends_on_its_dimension():
    small, large = HashEmbeddingProvider(dimension=256), HashEmbeddingProvider(dimension=512)
    assert (small.model_id, small.dimension) == ("hash:256", 256)
    assert small.space != large.space


def test_query_and_document_methods_return_vectors_of_the_declared_size():
    provider = HashEmbeddingProvider(dimension=64)
    documents = asyncio.run(provider.embed_documents(["first text", "second text"]))
    query = asyncio.run(provider.embed_query("first"))
    assert [len(vector) for vector in documents] == [64, 64]
    assert len(query) == 64
