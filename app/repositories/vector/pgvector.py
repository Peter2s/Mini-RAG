"""pgvector store: vectors in PostgreSQL, next to the chunks they belong to.

Each embedding model gets its own table, `chunk_vectors__<model>`, created at startup. That
mirrors Qdrant's collection per model: an HNSW index needs one fixed dimension, and vectors of
different models never mix.

The tables have no foreign key to `chunks`. The store writes on its own connections, so a
foreign key would make it wait on chunk rows locked by the request's open transaction.
"""

import hashlib
import math
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Index, MetaData, Table, Uuid, delete, func, inspect, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import Executable

from app.core.exceptions import VectorStoreError
from app.repositories.vector.base import VectorRepository, check_dimension, model_slug
from app.schemas.vector import EmbeddingSpace, VectorHit, VectorPoint

_TABLE_PREFIX = "chunk_vectors__"
_MAX_TABLE_NAME = 50
_HNSW_MAX_DIMENSION = 2000


def pgvector_table_name(model_id: str) -> str:
    """The table for one embedding model, short enough for its index names (63 bytes max)."""
    name = _TABLE_PREFIX + model_slug(model_id, separator="_")
    if len(name) <= _MAX_TABLE_NAME:
        return name
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:8]
    return f"{name[: _MAX_TABLE_NAME - len(digest) - 1]}_{digest}"


async def _pg[T](operation: str, call: Awaitable[T]) -> T:
    try:
        return await call
    except Exception as error:
        raise VectorStoreError(f"pgvector {operation} failed: {error}") from error


def _uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


class PgVectorRepository(VectorRepository):
    def __init__(self, engine: AsyncEngine, space: EmbeddingSpace) -> None:
        self._engine = engine
        self._space = space
        self.table_name = pgvector_table_name(space.model_id)
        self._metadata = MetaData()
        self._table = Table(
            self.table_name,
            self._metadata,
            Column("chunk_id", Uuid, primary_key=True),
            Column("collection_id", Uuid, nullable=False, index=True),
            Column("document_id", Uuid, nullable=False, index=True),
            Column("embedding", Vector(space.dimension), nullable=False),
        )
        if space.dimension <= _HNSW_MAX_DIMENSION:
            Index(
                f"{self.table_name}_hnsw",
                self._table.c.embedding,
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "vector_cosine_ops"},
            )

    async def init(self) -> None:
        await _pg("startup check", self._check_extension())
        try:
            await self._create_table()
        except Exception as error:
            if not await _pg("lookup", self._table_exists()):
                raise VectorStoreError(f"pgvector create table failed: {error}") from error
        stored = await _pg("describe", self._stored_dimension())
        if stored != self._space.dimension:
            raise VectorStoreError(
                f"pgvector table '{self.table_name}' stores {stored}-dimension vectors, but "
                f"'{self._space.model_id}' produces {self._space.dimension}. Drop that table, "
                "then re-index with do_reset=true."
            )

    async def upsert(self, collection_id: str, points: Sequence[VectorPoint]) -> None:
        if not points:
            return
        check_dimension(self._space, [point.vector for point in points])
        collection = uuid.UUID(collection_id)
        statement = insert(self._table)
        statement = statement.on_conflict_do_update(
            index_elements=[self._table.c.chunk_id],
            set_={
                "collection_id": statement.excluded.collection_id,
                "document_id": statement.excluded.document_id,
                "embedding": statement.excluded.embedding,
            },
        )
        rows = [
            {
                "chunk_id": uuid.UUID(point.chunk_id),
                "collection_id": collection,
                "document_id": uuid.UUID(point.document_id),
                "embedding": point.vector,
            }
            for point in points
        ]
        await _pg("upsert", self._execute(statement, rows))

    async def search(
        self, collection_id: str, query_vector: list[float], top_k: int
    ) -> list[VectorHit]:
        check_dimension(self._space, [query_vector])
        collection = _uuid(collection_id)
        if collection is None:
            return []
        distance = self._table.c.embedding.cosine_distance(query_vector)
        statement = (
            select(self._table.c.chunk_id, distance.label("distance"))
            .where(self._table.c.collection_id == collection)
            .order_by(distance)
            .limit(top_k)
        )
        rows = await _pg("search", self._search(statement))
        return [VectorHit(chunk_id=str(chunk_id), score=_similarity(d)) for chunk_id, d in rows]

    async def delete_document(self, collection_id: str, document_id: str) -> None:
        collection, document = _uuid(collection_id), _uuid(document_id)
        if collection is not None and document is not None:
            await _pg(
                "delete",
                self._execute(
                    delete(self._table).where(
                        self._table.c.collection_id == collection,
                        self._table.c.document_id == document,
                    )
                ),
            )

    async def delete_collection(self, collection_id: str) -> None:
        collection = _uuid(collection_id)
        if collection is not None:
            await _pg(
                "delete",
                self._execute(delete(self._table).where(self._table.c.collection_id == collection)),
            )

    async def count(self, collection_id: str) -> int:
        collection = _uuid(collection_id)
        if collection is None:
            return 0
        statement = (
            select(func.count())
            .select_from(self._table)
            .where(self._table.c.collection_id == collection)
        )
        return await _pg("count", self._scalar(statement))

    async def ping(self) -> None:
        await _pg("ping", self._scalar(select(1)))

    async def close(self) -> None:
        return None

    async def _execute(
        self, statement: Executable, rows: Sequence[Mapping[str, Any]] | None = None
    ) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(statement, rows)

    async def _scalar(self, statement: Executable) -> int:
        async with self._engine.connect() as connection:
            value = await connection.scalar(statement)
        return int(value or 0)

    async def _search(self, statement: Executable) -> list[tuple[uuid.UUID, float | None]]:
        async with self._engine.begin() as connection:
            await connection.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
            result = await connection.execute(statement)
            return [(row[0], row[1]) for row in result]

    async def _check_extension(self) -> None:
        async with self._engine.connect() as connection:
            installed = await connection.scalar(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
        if installed is None:
            raise RuntimeError("the vector extension is missing. Run: uv run alembic upgrade head")

    async def _create_table(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    async def _table_exists(self) -> bool:
        async with self._engine.connect() as connection:
            return await connection.run_sync(
                lambda sync_connection: inspect(sync_connection).has_table(self.table_name)
            )

    async def _stored_dimension(self) -> int:
        async with self._engine.connect() as connection:
            dimension = await connection.scalar(
                text(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
                ),
                {"table": self.table_name},
            )
        return int(dimension)


def _similarity(distance: float | None) -> float:
    """Cosine similarity from pgvector's cosine distance. A zero vector gives NaN: score 0."""
    if distance is None or math.isnan(distance):
        return 0.0
    return 1.0 - distance
