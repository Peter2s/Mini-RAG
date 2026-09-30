from datetime import UTC, datetime

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import WriteError

from app.core.enums import MongoCollection
from app.db.schema import INDEXES, VALIDATORS
from app.main import create_app

NOW = datetime.now(UTC)


def test_every_mongo_collection_has_a_schema_and_indexes():
    assert set(VALIDATORS) == set(MongoCollection)
    assert set(INDEXES) == set(MongoCollection)


def test_startup_creates_collections_validators_and_indexes(client, mongo_db):
    collections_by_name = {
        collection_info["name"]: collection_info for collection_info in mongo_db.list_collections()
    }
    for name, validator in VALIDATORS.items():
        assert collections_by_name[name]["options"]["validator"] == validator
        assert collections_by_name[name]["options"]["validationAction"] == "error"
        index_names = set(mongo_db[name].index_information())
        assert {index.document["name"] for index in INDEXES[name]} <= index_names
    assert mongo_db[MongoCollection.COLLECTIONS].index_information()["uniq_name"]["unique"] is True


def test_schema_can_be_applied_again_on_restart(client, settings):
    with TestClient(create_app(settings)) as second:
        assert second.get("/api/v1/health").json()["database"] == "ok"


@pytest.mark.parametrize(
    ("collection", "document"),
    [
        (
            MongoCollection.COLLECTIONS,
            {"name": 123, "description": None, "created_at": NOW, "updated_at": NOW},
        ),
        (
            MongoCollection.COLLECTIONS,
            {"name": "x", "description": None, "created_at": NOW},
        ),  # missing field
        (
            MongoCollection.COLLECTIONS,
            {"name": "x", "description": None, "created_at": NOW, "updated_at": NOW, "extra": 1},
        ),
        (
            MongoCollection.DOCUMENTS,
            {
                "collection_id": ObjectId(),
                "filename": "a.txt",
                "stored_path": "/tmp/a.txt",
                "content_type": None,
                "size_bytes": 1,
                "status": "bogus",
                "error": None,
                "created_at": NOW,
                "updated_at": NOW,
            },
        ),
        (
            MongoCollection.CHUNKS,
            {
                "collection_id": ObjectId(),
                "document_id": "not-an-object-id",
                "chunk_index": 0,
                "text": "hi",
                "metadata": {},
                "embedding": None,
                "created_at": NOW,
            },
        ),
        (
            MongoCollection.CHUNKS,
            {
                "collection_id": ObjectId(),
                "document_id": ObjectId(),
                "chunk_index": 0,
                "text": "hi",
                "metadata": {},
                "embedding": ["not", "floats"],
                "created_at": NOW,
            },
        ),
    ],
)
def test_validator_rejects_invalid_documents(client, mongo_db, collection, document):
    with pytest.raises(WriteError):
        mongo_db[collection].insert_one(document)


def _ingest(client, collection_id: str) -> None:
    base = f"/api/v1/collections/{collection_id}"
    for name in ("a.txt", "b.txt"):
        client.post(f"{base}/documents", files={"file": (name, b"some text " * 50, "text/plain")})
    client.post(f"{base}/documents/process")
    client.post(f"{base}/index")


def test_stored_documents_use_object_ids_and_embeddings(client, collection, mongo_db):
    _ingest(client, collection["id"])
    chunk = mongo_db[MongoCollection.CHUNKS].find_one()
    assert isinstance(chunk["collection_id"], ObjectId)
    assert isinstance(chunk["document_id"], ObjectId)
    assert len(chunk["embedding"]) == 512
    document = mongo_db[MongoCollection.DOCUMENTS].find_one()
    assert document["status"] == "processed"


def test_deleting_document_deletes_its_chunks(client, collection, mongo_db):
    _ingest(client, collection["id"])
    documents = client.get(f"/api/v1/collections/{collection['id']}/documents").json()["items"]
    first = documents[0]["id"]

    response = client.delete(f"/api/v1/collections/{collection['id']}/documents/{first}")
    assert response.status_code == 204
    assert mongo_db[MongoCollection.CHUNKS].count_documents({"document_id": ObjectId(first)}) == 0
    assert (
        mongo_db[MongoCollection.CHUNKS].count_documents({}) > 0
    )  # the other document's chunks remain


def test_deleting_collection_deletes_documents_and_chunks(client, collection, mongo_db):
    _ingest(client, collection["id"])
    other = client.post("/api/v1/collections", json={"name": "other"}).json()
    _ingest(client, other["id"])

    assert client.delete(f"/api/v1/collections/{collection['id']}").status_code == 204

    collection_object_id = ObjectId(collection["id"])
    assert (
        mongo_db[MongoCollection.DOCUMENTS].count_documents({"collection_id": collection_object_id})
        == 0
    )
    assert (
        mongo_db[MongoCollection.CHUNKS].count_documents({"collection_id": collection_object_id})
        == 0
    )
    other_collection_object_id = ObjectId(other["id"])
    assert (
        mongo_db[MongoCollection.DOCUMENTS].count_documents(
            {"collection_id": other_collection_object_id}
        )
        == 2
    )
    assert (
        mongo_db[MongoCollection.CHUNKS].count_documents(
            {"collection_id": other_collection_object_id}
        )
        > 0
    )
