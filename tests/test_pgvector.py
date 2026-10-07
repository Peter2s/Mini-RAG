"""pgvector-specific behavior: per-model tables, HNSW index, sizes, and restarts."""

import asyncio
import uuid

import pytest
from sqlalchemy import Connection, inspect

from app.core.exceptions import VectorStoreError
from app.db import Database
from app.repositories.vector.pgvector import PgVectorRepository, pgvector_table_name
from app.schemas.vector import EmbeddingSpace, VectorPoint

SPACE = EmbeddingSpace(model_id="test:4", dimension=4)


def _run(database_url: str, space: EmbeddingSpace, scenario):
    async def main():
        database = Database(database_url)
        repository = PgVectorRepository(database.engine, space)
        try:
            return await scenario(repository)
        finally:
            await repository.close()
            await database.close()

    return asyncio.run(main())


def test_each_model_gets_its_own_short_table_name():
    assert pgvector_table_name("hash:512") == "chunk_vectors__hash_512"
    assert pgvector_table_name("hash:512") != pgvector_table_name("hash:256")
    long_name = pgvector_table_name("fastembed:" + "a-very-long-organization/model-name" * 3)
    assert len(long_name) <= 50
    assert long_name != pgvector_table_name("fastembed:" + "a-very-long-organization/x" * 3)


def test_table_and_hnsw_index_are_created_at_startup(database_url, db: Connection):
    async def scenario(repository: PgVectorRepository) -> str:
        await repository.init()
        await repository.init()
        return repository.table_name

    table = _run(database_url, SPACE, scenario)

    inspector = inspect(db)
    assert {column["name"] for column in inspector.get_columns(table)} == {
        "chunk_id",
        "collection_id",
        "document_id",
        "embedding",
    }
    indexes = {index["name"]: index for index in inspector.get_indexes(table)}
    assert indexes[f"{table}_hnsw"]["dialect_options"]["postgresql_using"] == "hnsw"


def test_vectors_survive_a_restart(database_url, db):
    collection, chunk = str(uuid.uuid4()), str(uuid.uuid4())

    async def write(repository: PgVectorRepository) -> None:
        await repository.init()
        await repository.upsert(collection, [VectorPoint(chunk, str(uuid.uuid4()), [1, 0, 0, 0])])

    async def read(repository: PgVectorRepository) -> list[str]:
        await repository.init()
        return [hit.chunk_id for hit in await repository.search(collection, [1, 0, 0, 0], 1)]

    _run(database_url, SPACE, write)
    assert _run(database_url, SPACE, read) == [chunk]


def test_startup_rejects_a_table_of_another_size(database_url, db):
    async def create(repository: PgVectorRepository) -> None:
        await repository.init()

    async def resized(repository: PgVectorRepository) -> None:
        with pytest.raises(VectorStoreError, match="stores 4-dimension vectors"):
            await repository.init()

    _run(database_url, SPACE, create)
    _run(database_url, EmbeddingSpace(SPACE.model_id, 2), resized)


def test_ids_that_are_not_uuids_find_nothing(database_url, db):
    async def scenario(repository: PgVectorRepository) -> None:
        await repository.init()
        assert await repository.search("not-a-uuid", [1, 0, 0, 0], 5) == []
        assert await repository.count("not-a-uuid") == 0
        await repository.delete_collection("not-a-uuid")
        await repository.delete_document("not-a-uuid", "nor-this")

    _run(database_url, SPACE, scenario)


def test_unreachable_database_raises_vector_store_error():
    async def scenario(repository: PgVectorRepository) -> None:
        with pytest.raises(VectorStoreError, match="pgvector ping failed"):
            await repository.ping()

    _run("postgresql+asyncpg://nobody@127.0.0.1:9/none", SPACE, scenario)
