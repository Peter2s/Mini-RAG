"""Qdrant-specific behavior: point ids, tenant indexes, vector sizes, and the API end to end."""

import asyncio
import uuid
from contextlib import closing

import pytest
from bson import ObjectId
from qdrant_client import QdrantClient, models

from app.core.exceptions import VectorStoreError
from app.repositories.vector.qdrant import (
    QdrantVectorRepository,
    point_id,
    qdrant_collection_name,
)
from app.schemas.vector import EmbeddingSpace, VectorPoint

SPACE = EmbeddingSpace(model_id="test:4", dimension=4)


def _sync_client(qdrant_connection) -> QdrantClient:
    url, api_key = qdrant_connection
    return QdrantClient(url=url, api_key=api_key)


def _repository(
    qdrant_connection, prefix: str, space: EmbeddingSpace = SPACE
) -> QdrantVectorRepository:
    url, api_key = qdrant_connection
    return QdrantVectorRepository(url=url, api_key=api_key, collection_prefix=prefix, space=space)


def test_point_id_is_a_deterministic_uuid_per_chunk():
    chunk_id, other = str(ObjectId()), str(ObjectId())
    assert uuid.UUID(point_id(chunk_id))
    assert point_id(chunk_id) == point_id(chunk_id)
    assert point_id(chunk_id) != point_id(other)


def test_each_model_gets_its_own_collection_name():
    assert (
        qdrant_collection_name("chunks", "fastembed:google/embeddinggemma-300m")
        == "chunks__fastembed-google-embeddinggemma-300m"
    )
    assert qdrant_collection_name("chunks", "hash:512") != qdrant_collection_name(
        "chunks", "hash:256"
    )


def test_collection_is_created_at_startup_with_tenant_indexes(qdrant_connection, qdrant_collection):
    repository = _repository(qdrant_connection, qdrant_collection)

    async def scenario() -> None:
        await repository.init()
        await repository.init()
        await repository.close()

    asyncio.run(scenario())
    with closing(_sync_client(qdrant_connection)) as client:
        info = client.get_collection(repository.collection_name)
    assert isinstance(info.config.params.vectors, models.VectorParams)
    assert info.config.params.vectors.size == 4
    assert info.config.params.vectors.distance == models.Distance.COSINE
    tenant_index = info.payload_schema["collection_id"]
    assert tenant_index.data_type == models.PayloadSchemaType.KEYWORD
    assert isinstance(tenant_index.params, models.KeywordIndexParams)
    assert tenant_index.params.is_tenant is True
    assert info.payload_schema["document_id"].data_type == models.PayloadSchemaType.KEYWORD


def test_restart_keeps_vectors(qdrant_connection, qdrant_collection):
    chunk_id = str(ObjectId())

    async def write() -> None:
        repository = _repository(qdrant_connection, qdrant_collection)
        await repository.init()
        await repository.upsert("collection-1", [VectorPoint(chunk_id, "d", [1.0, 0, 0, 0])])
        await repository.close()

    async def read_after_restart() -> list[str]:
        repository = _repository(qdrant_connection, qdrant_collection)
        await repository.init()
        hits = await repository.search("collection-1", [1.0, 0, 0, 0], top_k=1)
        await repository.close()
        return [hit.chunk_id for hit in hits]

    asyncio.run(write())
    assert asyncio.run(read_after_restart()) == [chunk_id]


def test_startup_rejects_a_collection_of_another_size(qdrant_connection, qdrant_collection):
    """Same model id, different size (e.g. an edited setting): refuse rather than corrupt."""

    async def scenario() -> None:
        first = _repository(qdrant_connection, qdrant_collection)
        await first.init()
        await first.close()
        resized = _repository(
            qdrant_connection, qdrant_collection, EmbeddingSpace(SPACE.model_id, 2)
        )
        with pytest.raises(VectorStoreError, match="stores 4-dimension vectors"):
            await resized.init()
        await resized.close()

    asyncio.run(scenario())


@pytest.mark.filterwarnings("ignore:Failed to obtain server version:UserWarning")
def test_unreachable_qdrant_raises_vector_store_error(qdrant_collection):
    async def scenario() -> None:
        repository = QdrantVectorRepository("http://127.0.0.1:9", None, qdrant_collection, SPACE)
        with pytest.raises(VectorStoreError, match="Qdrant ping failed"):
            await repository.ping()
        await repository.close()

    asyncio.run(scenario())


DOCS = {
    "france.txt": b"Paris is the capital of France. France is famous for wine and cheese.",
    "japan.txt": b"Tokyo is the capital of Japan. Japan is known for sushi and bullet trains.",
    "brazil.txt": b"Brasilia is the capital of Brazil. Brazil is famous for football and coffee.",
}


def _qdrant_count(qdrant_connection, prefix: str, collection_id: str) -> int:
    collection_name = qdrant_collection_name(prefix, "hash:512")
    with closing(_sync_client(qdrant_connection)) as client:
        if not client.collection_exists(collection_name):
            return 0
        condition = models.FieldCondition(
            key="collection_id", match=models.MatchValue(value=collection_id)
        )
        return client.count(collection_name, count_filter=models.Filter(must=[condition])).count


def _ingest(client, collection_id: str) -> dict[str, str]:
    base = f"/api/v1/collections/{collection_id}"
    ids = {}
    for name, content in DOCS.items():
        response = client.post(f"{base}/documents", files={"file": (name, content, "text/plain")})
        ids[name] = response.json()["id"]
    assert client.post(f"{base}/documents/process").json()["documents_processed"] == 3
    assert client.post(f"{base}/index").json()["chunks_indexed"] == 3
    return ids


@pytest.fixture
def collection_id(qdrant_client_app) -> str:
    created: str = qdrant_client_app.post("/api/v1/collections", json={"name": "q"}).json()["id"]
    return created


def test_health_reports_qdrant(qdrant_client_app):
    assert qdrant_client_app.get("/api/v1/health").json()["vector_store"] == "ok"


def test_index_search_and_answer_through_qdrant(
    qdrant_client_app, collection_id, qdrant_connection, qdrant_collection, fake_llm
):
    _ingest(qdrant_client_app, collection_id)
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 3

    base = f"/api/v1/collections/{collection_id}"
    hits = qdrant_client_app.post(
        f"{base}/search", json={"query": "What is the capital of Japan?", "top_k": 2}
    ).json()["results"]
    assert hits[0]["filename"] == "japan.txt"
    assert hits[0]["text"].startswith("Tokyo")
    assert hits[0]["score"] >= hits[1]["score"]

    answer = qdrant_client_app.post(
        f"{base}/answer", json={"question": "What is the capital of France?", "top_k": 1}
    ).json()
    assert answer["sources"][0]["filename"] == "france.txt"
    assert "Paris is the capital of France" in fake_llm.calls[0]["prompt"]

    assert qdrant_client_app.post(f"{base}/index").json()["chunks_indexed"] == 0
    reset = qdrant_client_app.post(f"{base}/index", json={"do_reset": True})
    assert reset.json()["chunks_indexed"] == 3
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 3


def test_reprocessing_and_deleting_documents_remove_their_vectors(
    qdrant_client_app, collection_id, qdrant_connection, qdrant_collection
):
    ids = _ingest(qdrant_client_app, collection_id)
    base = f"/api/v1/collections/{collection_id}"

    reprocessed = qdrant_client_app.post(f"{base}/documents/{ids['japan.txt']}/process").json()
    assert reprocessed["status"] == "processed"
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 2
    assert qdrant_client_app.post(f"{base}/index").json()["chunks_indexed"] == 1
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 3

    assert qdrant_client_app.delete(f"{base}/documents/{ids['brazil.txt']}").status_code == 204
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 2
    filenames = {
        hit["filename"]
        for hit in qdrant_client_app.post(
            f"{base}/search", json={"query": "capital", "top_k": 10}
        ).json()["results"]
    }
    assert filenames == {"france.txt", "japan.txt"}


def test_deleting_a_collection_removes_only_its_vectors(
    qdrant_client_app, collection_id, qdrant_connection, qdrant_collection
):
    other = qdrant_client_app.post("/api/v1/collections", json={"name": "other"}).json()["id"]
    _ingest(qdrant_client_app, collection_id)
    _ingest(qdrant_client_app, other)

    assert qdrant_client_app.delete(f"/api/v1/collections/{collection_id}").status_code == 204
    assert _qdrant_count(qdrant_connection, qdrant_collection, collection_id) == 0
    assert _qdrant_count(qdrant_connection, qdrant_collection, other) == 3
