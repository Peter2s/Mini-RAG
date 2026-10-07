"""Contract tests: every VectorRepository implementation must behave the same way."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from bson import ObjectId

from app.core.config import Settings
from app.core.enums import VectorStoreName
from app.db import MongoDatabase
from app.repositories import ChunkRepository, VectorRepository, build_vector_repository
from app.repositories.vector import MongoVectorRepository, QdrantVectorRepository
from app.schemas.chunk import TextChunk
from app.schemas.vector import EmbeddingSpace, VectorPoint

Scenario = Callable[[VectorRepository, "Corpus"], Awaitable[Any]]

SPACE = EmbeddingSpace(model_id="test:3", dimension=3)


class Corpus:
    """Two collections with known chunk ids and hand-made 3-d vectors."""

    def __init__(self) -> None:
        self.collection = str(ObjectId())
        self.other_collection = str(ObjectId())
        self.document_a = str(ObjectId())
        self.document_b = str(ObjectId())
        self.other_document = str(ObjectId())
        self.chunk_ids: dict[str, list[str]] = {}

    async def create_chunks(self, chunks: ChunkRepository) -> None:
        layout = {
            (self.collection, self.document_a): 2,
            (self.collection, self.document_b): 1,
            (self.other_collection, self.other_document): 1,
        }
        for (collection_id, document_id), count in layout.items():
            await chunks.replace_for_document(
                collection_id, document_id, [TextChunk(text=f"chunk {i}") for i in range(count)]
            )
            stored = await chunks.list_for_indexing(collection_id, SPACE.model_id)
            self.chunk_ids[document_id] = [c.id for c in stored if c.document_id == document_id]

    def points(self) -> dict[str, list[VectorPoint]]:
        a, b, other = (
            self.chunk_ids[d] for d in (self.document_a, self.document_b, self.other_document)
        )
        return {
            self.collection: [
                VectorPoint(a[0], self.document_a, [1.0, 0.0, 0.0]),
                VectorPoint(a[1], self.document_a, [0.9, 0.1, 0.0]),
                VectorPoint(b[0], self.document_b, [0.0, 1.0, 0.0]),
            ],
            self.other_collection: [VectorPoint(other[0], self.other_document, [1.0, 0.0, 0.0])],
        }

    async def index_all(self, vectors: VectorRepository) -> None:
        for collection_id, points in self.points().items():
            await vectors.upsert(collection_id, points)


@pytest.fixture(params=[VectorStoreName.MONGODB, VectorStoreName.QDRANT])
def store_settings(request: pytest.FixtureRequest, settings: Settings) -> Settings:
    if request.param == VectorStoreName.QDRANT:
        qdrant_settings: Settings = request.getfixturevalue("qdrant_settings")
        return qdrant_settings
    return settings


def run(settings: Settings, scenario: Scenario, space: EmbeddingSpace = SPACE) -> Any:
    async def main() -> Any:
        mongo = MongoDatabase(settings.mongodb_url.get_secret_value(), settings.mongodb_database)
        await mongo.init()
        vectors = build_vector_repository(settings, mongo, space)
        await vectors.init()
        corpus = Corpus()
        await corpus.create_chunks(ChunkRepository(mongo.db))
        try:
            return await scenario(vectors, corpus)
        finally:
            await vectors.close()
            await mongo.close()

    return asyncio.run(main())


def test_empty_store_finds_nothing(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        assert await vectors.search(corpus.collection, [1.0, 0.0, 0.0], top_k=5) == []
        assert await vectors.count(corpus.collection) == 0
        await vectors.delete_collection(corpus.collection)

    run(store_settings, scenario)


def test_search_ranks_by_similarity_within_one_collection(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        hits = await vectors.search(corpus.collection, [1.0, 0.0, 0.0], top_k=3)

        a, b = corpus.chunk_ids[corpus.document_a], corpus.chunk_ids[corpus.document_b]
        assert [hit.chunk_id for hit in hits] == [a[0], a[1], b[0]]
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)
        assert hits[0].score >= hits[1].score >= hits[2].score
        assert corpus.chunk_ids[corpus.other_document][0] not in {hit.chunk_id for hit in hits}

    run(store_settings, scenario)


def test_top_k_limits_results(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        assert len(await vectors.search(corpus.collection, [1.0, 0.0, 0.0], top_k=1)) == 1

    run(store_settings, scenario)


def test_upserting_the_same_chunk_twice_keeps_one_vector(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        await corpus.index_all(vectors)
        assert await vectors.count(corpus.collection) == 3
        assert await vectors.count(corpus.other_collection) == 1

    run(store_settings, scenario)


def test_delete_document_removes_only_its_vectors(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        await vectors.delete_document(corpus.collection, corpus.document_a)

        hits = await vectors.search(corpus.collection, [1.0, 0.0, 0.0], top_k=5)
        assert [hit.chunk_id for hit in hits] == corpus.chunk_ids[corpus.document_b]
        assert await vectors.count(corpus.other_collection) == 1

    run(store_settings, scenario)


def test_delete_collection_leaves_other_collections_alone(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        await vectors.delete_collection(corpus.collection)

        assert await vectors.count(corpus.collection) == 0
        assert await vectors.search(corpus.collection, [1.0, 0.0, 0.0], top_k=5) == []
        assert await vectors.count(corpus.other_collection) == 1

    run(store_settings, scenario)


def test_query_with_wrong_dimension_is_rejected(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        with pytest.raises(ValueError, match="dimension mismatch"):
            await vectors.search(corpus.collection, [1.0, 0.0], top_k=1)

    run(store_settings, scenario)


def test_vector_with_wrong_dimension_is_rejected(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        chunk_id = corpus.chunk_ids[corpus.document_a][0]
        with pytest.raises(ValueError, match="dimension mismatch"):
            await vectors.upsert(
                corpus.collection, [VectorPoint(chunk_id, corpus.document_a, [1.0, 0.0])]
            )

    run(store_settings, scenario)


def test_vectors_of_another_model_are_invisible(store_settings):
    """Same size, different model: the vectors aren't comparable, so they must never mix."""
    other_model = EmbeddingSpace(model_id="other:3", dimension=3)
    searched: dict[str, Any] = {}

    async def index_with_one_model(vectors: VectorRepository, corpus: Corpus) -> None:
        await corpus.index_all(vectors)
        searched["collection"] = corpus.collection

    async def search_with_another(vectors: VectorRepository, corpus: Corpus) -> None:
        assert await vectors.search(searched["collection"], [1.0, 0.0, 0.0], top_k=5) == []
        assert await vectors.count(searched["collection"]) == 0

    run(store_settings, index_with_one_model)
    run(store_settings, search_with_another, space=other_model)


def test_ping_succeeds(store_settings):
    async def scenario(vectors: VectorRepository, corpus: Corpus) -> None:
        await vectors.ping()

    run(store_settings, scenario)


@pytest.mark.parametrize("implementation", [MongoVectorRepository, QdrantVectorRepository])
def test_stores_implement_the_interface(implementation):
    assert issubclass(implementation, VectorRepository)
    assert not implementation.__abstractmethods__


def test_interface_cannot_be_instantiated_without_every_method():
    class Incomplete(VectorRepository):
        async def ping(self) -> None:
            return None

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]
