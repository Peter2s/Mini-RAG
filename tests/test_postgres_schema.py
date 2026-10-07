"""The PostgreSQL schema: constraints, indexes the queries rely on, cascades, stored values."""

import uuid
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, inspect, text
from sqlalchemy.exc import DataError, IntegrityError

DOCS = {
    "a.txt": b"Paris is the capital of France. " * 20,
    "b.txt": b"Tokyo is the capital of Japan. " * 20,
}
SHA256 = "0" * 64


def _ingest(client, collection_id: str) -> None:
    base = f"/api/v1/collections/{collection_id}"
    for name, content in DOCS.items():
        assert client.post(f"{base}/documents", files={"file": (name, content)}).status_code == 201
    assert client.post(f"{base}/documents/process").json()["documents_failed"] == 0
    assert client.post(f"{base}/index").json()["chunks_indexed"] > 0


COUNT_BY_COLLECTION = {
    "collections": "SELECT count(*) FROM collections WHERE id = :id",
    "documents": "SELECT count(*) FROM documents WHERE collection_id = :id",
    "chunks": "SELECT count(*) FROM chunks WHERE collection_id = :id",
}


def _count(db: Connection, table: str, collection_id: str) -> int:
    return db.execute(text(COUNT_BY_COLLECTION[table]), {"id": collection_id}).scalar_one()


def test_tables_have_the_named_constraints_and_indexes(client, db):
    inspector = inspect(db)
    unique = {
        table: {c["name"] for c in inspector.get_unique_constraints(table)}
        for table in ("collections", "documents", "chunks")
    }
    assert unique == {
        "collections": {"uq_collections_name"},
        "documents": {"uq_documents_collection_id_content_hash"},
        "chunks": {"uq_chunks_document_id_chunk_index"},
    }
    indexes = {
        table: {
            index["name"]
            for index in inspector.get_indexes(table)
            if not index.get("duplicates_constraint")
        }
        for table in ("documents", "chunks")
    }
    assert indexes == {
        "documents": {"ix_documents_collection_id_id"},
        "chunks": {"ix_chunks_collection_id_id"},
    }
    foreign_keys = inspector.get_foreign_keys("chunks") + inspector.get_foreign_keys("documents")
    assert {fk["options"].get("ondelete") for fk in foreign_keys} == {"CASCADE"}


def _insert_collection(db: Connection, name: str = "c") -> str:
    statement = text("INSERT INTO collections (name) VALUES (:name) RETURNING id")
    return str(db.execute(statement, {"name": name}).scalar_one())


def _insert_document(db: Connection, **overrides: Any) -> str:
    values = {
        "filename": "a.txt",
        "stored_path": "/tmp/a.txt",
        "size_bytes": 1,
        "content_hash": SHA256,
        "status": "uploaded",
        **overrides,
    }
    statement = text(
        "INSERT INTO documents "
        "(collection_id, filename, stored_path, size_bytes, content_hash, status) "
        "VALUES (:collection_id, :filename, :stored_path, :size_bytes, :content_hash, :status) "
        "RETURNING id"
    )
    return str(db.execute(statement, values).scalar_one())


def _insert_chunk(db: Connection, collection_id: str, document_id: str, index: int) -> None:
    db.execute(
        text(
            "INSERT INTO chunks (collection_id, document_id, chunk_index, text, metadata) "
            "VALUES (:collection_id, :document_id, :chunk_index, 'hi', '{}')"
        ),
        {"collection_id": collection_id, "document_id": document_id, "chunk_index": index},
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "done"},
        {"content_hash": "not-a-sha256"},
        {"content_hash": "A" * 64},
        {"size_bytes": -1},
        {"collection_id": str(uuid.uuid4())},
    ],
    ids=["status", "hash", "uppercase-hash", "size", "missing-collection"],
)
def test_invalid_documents_are_rejected(client, db, overrides):
    values = {"collection_id": _insert_collection(db), **overrides}
    with pytest.raises(IntegrityError):
        _insert_document(db, **values)


@pytest.mark.parametrize("name", ["", "x" * 101], ids=["empty", "too-long"])
def test_invalid_collection_names_are_rejected(client, db, name):
    with pytest.raises((IntegrityError, DataError)):
        _insert_collection(db, name)


def test_duplicate_or_negative_chunk_index_is_rejected(client, db):
    collection_id = _insert_collection(db)
    document_id = _insert_document(db, collection_id=collection_id)
    _insert_chunk(db, collection_id, document_id, 0)
    with pytest.raises(IntegrityError):
        _insert_chunk(db, collection_id, document_id, 0)
    with pytest.raises(IntegrityError):
        _insert_chunk(db, collection_id, document_id, -1)


QUERIES = {
    "documents of a collection, in order": (
        "EXPLAIN SELECT * FROM documents WHERE collection_id = :id ORDER BY id LIMIT 50",
        "ix_documents_collection_id_id",
    ),
    "chunks to index, in order": (
        "EXPLAIN SELECT * FROM chunks WHERE collection_id = :id "
        "AND embedding_model IS DISTINCT FROM 'hash:512' ORDER BY id",
        "ix_chunks_collection_id_id",
    ),
    "duplicate check by content hash": (
        "EXPLAIN SELECT * FROM documents "
        "WHERE collection_id = :id AND content_hash = repeat('0', 64)",
        ("uq_documents_collection_id_content_hash", "ix_documents_collection_id_id"),
    ),
    "chunks of a document": (
        "EXPLAIN SELECT * FROM chunks WHERE document_id = :id",
        "uq_chunks_document_id_chunk_index",
    ),
    "collection by name": (
        "EXPLAIN SELECT * FROM collections WHERE name = 'demo'",
        "uq_collections_name",
    ),
}
PLANNER_OFF = ("enable_seqscan", "enable_bitmapscan", "enable_sort")


@pytest.mark.parametrize(("query", "indexes"), QUERIES.values(), ids=QUERIES.keys())
def test_queries_use_an_index_without_sorting(client, sync_engine: Engine, query, indexes):
    """With sequential scans and sorts discouraged, the plan must use an expected index.

    On empty tables either of two fitting indexes may win, so some queries accept both.
    """
    transactional = sync_engine.connect().execution_options(isolation_level="READ COMMITTED")
    with transactional as connection, connection.begin():
        for setting in PLANNER_OFF:
            connection.execute(text("SELECT set_config(:name, 'off', true)"), {"name": setting})
        plan = "\n".join(connection.execute(text(query), {"id": str(uuid.uuid4())}).scalars())
    expected = (indexes,) if isinstance(indexes, str) else indexes
    assert any(index in plan for index in expected), plan
    assert "Sort" not in plan


def test_deleting_a_collection_cascades_to_its_rows_only(client, db, collection):
    other = client.post("/api/v1/collections", json={"name": "other"}).json()["id"]
    _ingest(client, collection["id"])
    _ingest(client, other)

    assert client.delete(f"/api/v1/collections/{collection['id']}").status_code == 204

    for table in ("collections", "documents", "chunks"):
        assert _count(db, table, collection["id"]) == 0
        assert _count(db, table, other) > 0
    assert client.post(f"/api/v1/collections/{other}/search", json={"query": "Paris"}).json()[
        "results"
    ]


def test_deleting_a_document_cascades_to_its_chunks(client, db, collection):
    _ingest(client, collection["id"])
    base = f"/api/v1/collections/{collection['id']}/documents"
    first, second = client.get(base).json()["items"]

    assert client.delete(f"{base}/{first['id']}").status_code == 204

    document_ids = set(
        db.execute(
            text("SELECT DISTINCT document_id::text FROM chunks WHERE collection_id = :id"),
            {"id": collection["id"]},
        ).scalars()
    )
    assert document_ids == {second["id"]}


def test_stored_rows_use_uuid7_ids_and_record_indexing(client, db, collection):
    _ingest(client, collection["id"])
    document = db.execute(text("SELECT id, status FROM documents LIMIT 1")).one()
    chunk = db.execute(text("SELECT id, embedding_model, indexed_at FROM chunks LIMIT 1")).one()

    assert uuid.UUID(collection["id"]).version == 7
    assert document.id.version == chunk.id.version == 7
    assert document.status == "processed"
    assert chunk.embedding_model == "hash:512"
    assert chunk.indexed_at is not None


def test_listing_follows_creation_order(client, collection):
    base = f"/api/v1/collections/{collection['id']}/documents"
    names = [f"{index}.txt" for index in range(5)]
    for index, name in enumerate(names):
        client.post(base, files={"file": (name, f"content {index}".encode())})
    assert [item["filename"] for item in client.get(base).json()["items"]] == names


def test_reprocessing_replaces_chunks_despite_the_unique_chunk_index(client, db, collection):
    _ingest(client, collection["id"])
    base = f"/api/v1/collections/{collection['id']}/documents"
    document_id = client.get(base).json()["items"][0]["id"]

    for _ in range(2):
        result = client.post(f"{base}/{document_id}/process").json()
        assert result["status"] == "processed"

    statement = text("SELECT count(*) FROM chunks WHERE document_id = :id")
    assert db.execute(statement, {"id": document_id}).scalar_one() == result["chunks_created"]


def test_text_with_nul_characters_is_stored_without_them(client, db, collection):
    base = f"/api/v1/collections/{collection['id']}/documents"
    content = b"before\x00after " * 20
    document = client.post(base, files={"file": ("nul.txt", content)}).json()

    result = client.post(f"{base}/{document['id']}/process").json()

    assert result["status"] == "processed"
    texts = db.execute(text("SELECT text FROM chunks")).scalars().all()
    assert texts
    assert all("\x00" not in chunk_text and "beforeafter" in chunk_text for chunk_text in texts)
