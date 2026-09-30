"""Check that every query the repositories run is served by an index.

`explain()` shows the plan MongoDB picked. A good plan uses an IXSCAN on the expected index,
never a COLLSCAN (full collection scan) and never a blocking in-memory SORT stage.
"""

from typing import Any

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo import ASCENDING
from pymongo.errors import DuplicateKeyError

from app.core.enums import MongoCollection
from app.db.indexes import INDEXES
from app.main import create_app


def _plan_summary(explain: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Collect every stage name and index name in the winning plan (classic or SBE layout)."""
    stages: set[str] = set()
    index_names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "stage" in node:
                stages.add(node["stage"])
            if "indexName" in node:
                index_names.add(node["indexName"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(explain["queryPlanner"]["winningPlan"])
    return stages, index_names


def _assert_uses_index(cursor, index_name: str) -> None:
    stages, index_names = _plan_summary(cursor.explain())
    assert "COLLSCAN" not in stages, stages
    assert "SORT" not in stages, f"in-memory sort: {stages}"
    assert index_names == {index_name}, index_names


@pytest.fixture
def ingested(client, collection) -> dict[str, ObjectId]:
    base = f"/api/v1/collections/{collection['id']}"
    for name in ("a.txt", "b.txt"):
        content = f"{name} has some text. ".encode() * 20  # distinct content per file
        client.post(f"{base}/documents", files={"file": (name, content, "text/plain")})
    client.post(f"{base}/documents/process")
    documents = client.get(f"{base}/documents").json()["items"]
    return {
        "collection": ObjectId(collection["id"]),
        "document": ObjectId(documents[0]["id"]),
    }


def test_duplicate_check_by_content_hash(mongo_db, ingested):
    content_hash = mongo_db[MongoCollection.DOCUMENTS].find_one()["content_hash"]
    cursor = mongo_db[MongoCollection.DOCUMENTS].find(
        {"collection_id": ingested["collection"], "content_hash": content_hash}
    )
    _assert_uses_index(cursor, "uniq_collection_content_hash")


def test_collection_lookup_by_name(mongo_db, ingested):
    cursor = mongo_db[MongoCollection.COLLECTIONS].find({"name": "demo"})
    _assert_uses_index(cursor, "uniq_name")


def test_documents_listed_in_order_without_in_memory_sort(mongo_db, ingested):
    cursor = (
        mongo_db[MongoCollection.DOCUMENTS]
        .find({"collection_id": ingested["collection"]})
        .sort("_id", 1)
    )
    _assert_uses_index(cursor, "collection_id_order")


def test_chunks_to_index_listed_in_order_without_in_memory_sort(mongo_db, ingested):
    cursor = (
        mongo_db[MongoCollection.CHUNKS]
        .find({"collection_id": ingested["collection"], "embedding": None}, {"embedding": 0})
        .sort("_id", 1)
    )
    _assert_uses_index(cursor, "collection_id_order")


def test_vector_search_candidates(mongo_db, ingested):
    cursor = mongo_db[MongoCollection.CHUNKS].find(
        {"collection_id": ingested["collection"], "embedding": {"$ne": None}}
    )
    _assert_uses_index(cursor, "collection_id_order")


def test_chunks_of_a_document(mongo_db, ingested):
    cursor = mongo_db[MongoCollection.CHUNKS].find({"document_id": ingested["document"]})
    _assert_uses_index(cursor, "uniq_document_chunk_index")


def test_chunk_index_is_unique_per_document(mongo_db, ingested):
    chunks = mongo_db[MongoCollection.CHUNKS]
    existing = chunks.find_one({"document_id": ingested["document"]}, {"_id": 0})
    with pytest.raises(DuplicateKeyError):
        chunks.insert_one(existing)


def _index_names(mongo_db, name: MongoCollection) -> set[str]:
    return set(mongo_db[name].index_information())


def test_every_collection_has_exactly_the_declared_indexes(client, mongo_db):
    for name, indexes in INDEXES.items():
        assert _index_names(mongo_db, name) == {"_id_"} | {i.document["name"] for i in indexes}


def test_restart_drops_undeclared_and_rebuilds_changed_indexes(client, settings, mongo_db):
    chunks = mongo_db[MongoCollection.CHUNKS]
    # The indexes the previous version of the app created.
    chunks.create_index([("collection_id", ASCENDING)], name="by_collection")
    chunks.create_index(
        [("document_id", ASCENDING), ("chunk_index", ASCENDING)], name="by_document_order"
    )
    # A declared name whose definition drifted (no longer unique).
    chunks.drop_index("uniq_document_chunk_index")
    chunks.create_index([("document_id", ASCENDING)], name="uniq_document_chunk_index")

    with TestClient(create_app(settings)):
        pass

    info = chunks.index_information()
    assert set(info) == {"_id_", "collection_id_order", "uniq_document_chunk_index"}
    rebuilt = info["uniq_document_chunk_index"]
    assert rebuilt["unique"] is True
    assert [field for field, _ in rebuilt["key"]] == ["document_id", "chunk_index"]


def test_reprocessing_a_document_still_works_with_unique_chunk_index(client, collection, mongo_db):
    base = f"/api/v1/collections/{collection['id']}/documents"
    document = client.post(base, files={"file": ("a.txt", b"some text " * 50, "text/plain")}).json()
    for chunk_size in (200, 100):
        result = client.post(
            f"{base}/{document['id']}/process", json={"chunk_size": chunk_size, "chunk_overlap": 20}
        ).json()
        assert result["status"] == "processed", result
        stored = mongo_db[MongoCollection.CHUNKS].count_documents(
            {"document_id": ObjectId(document["id"])}
        )
        assert stored == result["chunks_created"]
