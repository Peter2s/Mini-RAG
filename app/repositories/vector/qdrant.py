"""Qdrant vector store.

Each embedding model gets its own Qdrant collection, named `<prefix>__<model id>`, so vectors
of different models never mix and switching models needs no cleanup.

Inside it, every app collection shares the Qdrant collection, partitioned by a `collection_id`
payload field indexed with `is_tenant=True`. That's Qdrant's recommended multitenancy layout:
one HNSW graph per tenant, fast filtered search, and no per-tenant collection overhead.
"""

import uuid
from collections.abc import Awaitable, Sequence

from qdrant_client import AsyncQdrantClient, models

from app.core.exceptions import VectorStoreError
from app.repositories.vector.base import VectorRepository, check_dimension, model_slug
from app.schemas.vector import EmbeddingSpace, VectorHit, VectorPoint

_COLLECTION_FIELD = "collection_id"
_DOCUMENT_FIELD = "document_id"
_CHUNK_FIELD = "chunk_id"


def qdrant_collection_name(prefix: str, model_id: str) -> str:
    """The Qdrant collection for one embedding model: `<prefix>__<model id as a slug>`."""
    return f"{prefix}__{model_slug(model_id)}"


def point_id(chunk_id: str) -> str:
    """Qdrant point ids must be integers or UUIDs; chunk ids are UUIDs already.

    The mapping is deterministic, so re-indexing a chunk overwrites its point.
    """
    return str(uuid.UUID(chunk_id))


async def _qdrant[T](operation: str, call: Awaitable[T]) -> T:
    try:
        return await call
    except Exception as error:
        raise VectorStoreError(f"Qdrant {operation} failed: {error}") from error


class QdrantVectorRepository(VectorRepository):
    def __init__(
        self, url: str, api_key: str | None, collection_prefix: str, space: EmbeddingSpace
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=30)
        self._space = space
        self.collection_name = qdrant_collection_name(collection_prefix, space.model_id)

    async def init(self) -> None:
        if not await self._exists():
            try:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self._space.dimension, distance=models.Distance.COSINE
                    ),
                )
            except Exception as error:
                if not await self._exists():
                    raise VectorStoreError(f"Qdrant create collection failed: {error}") from error
        await self._check_existing_dimension()
        await self._ensure_payload_indexes()

    async def upsert(self, collection_id: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        check_dimension(self._space, [point.vector for point in points])
        await _qdrant(
            "upsert",
            self._client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id(point.chunk_id),
                        vector=point.vector,
                        payload={
                            _COLLECTION_FIELD: collection_id,
                            _DOCUMENT_FIELD: point.document_id,
                            _CHUNK_FIELD: point.chunk_id,
                        },
                    )
                    for point in points
                ],
                wait=True,
            ),
        )

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorHit]:
        check_dimension(self._space, [query_vector])
        response = await _qdrant(
            "search",
            self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=_match(collection_id),
                limit=top_k,
                with_payload=[_CHUNK_FIELD],
            ),
        )
        return [
            VectorHit(chunk_id=str(point.payload[_CHUNK_FIELD]), score=point.score)
            for point in response.points
            if point.payload
        ]

    async def delete_document(self, collection_id: str, document_id: str) -> None:
        await self._delete(_match(collection_id, document_id))

    async def delete_collection(self, collection_id: str) -> None:
        await self._delete(_match(collection_id))

    async def count(self, collection_id: str) -> int:
        result = await _qdrant(
            "count",
            self._client.count(
                collection_name=self.collection_name,
                count_filter=_match(collection_id),
                exact=True,
            ),
        )
        return result.count

    async def ping(self) -> None:
        await _qdrant("ping", self._client.get_collections())

    async def close(self) -> None:
        await self._client.close()

    async def _delete(self, condition: models.Filter) -> None:
        await _qdrant(
            "delete",
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=condition),
                wait=True,
            ),
        )

    async def _exists(self) -> bool:
        return await _qdrant("lookup", self._client.collection_exists(self.collection_name))

    async def _check_existing_dimension(self) -> None:
        info = await _qdrant("describe", self._client.get_collection(self.collection_name))
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise VectorStoreError(
                f"Qdrant collection '{self.collection_name}' must use a single unnamed vector"
            )
        if vectors.size != self._space.dimension:
            raise VectorStoreError(
                f"Qdrant collection '{self.collection_name}' stores {vectors.size}-dimension "
                f"vectors, but '{self._space.model_id}' produces {self._space.dimension}. "
                "Delete that Qdrant collection, then re-index with do_reset=true."
            )

    async def _ensure_payload_indexes(self) -> None:
        await _qdrant(
            "create index",
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=_COLLECTION_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD, is_tenant=True
                ),
            ),
        )
        await _qdrant(
            "create index",
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=_DOCUMENT_FIELD,
                field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
            ),
        )


def _match(collection_id: str, document_id: str | None = None) -> models.Filter:
    conditions = [
        models.FieldCondition(key=_COLLECTION_FIELD, match=models.MatchValue(value=collection_id))
    ]
    if document_id is not None:
        conditions.append(
            models.FieldCondition(key=_DOCUMENT_FIELD, match=models.MatchValue(value=document_id))
        )
    return models.Filter(must=conditions)
