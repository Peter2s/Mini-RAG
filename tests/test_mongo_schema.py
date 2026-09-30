from datetime import UTC, datetime

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import WriteError

from app.db.schema import INDEXES, VALIDATORS
from app.main import create_app

NOW = datetime.now(UTC)


def test_startup_creates_collections_validators_and_indexes(client, mongo_db):
    info = {c["name"]: c for c in mongo_db.list_collections()}
    for name, validator in VALIDATORS.items():
        assert info[name]["options"]["validator"] == validator
        assert info[name]["options"]["validationAction"] == "error"
        index_names = set(mongo_db[name].index_information())
        assert {index.document["name"] for index in INDEXES[name]} <= index_names
    assert mongo_db["collections"].index_information()["uniq_name"]["unique"] is True


def test_schema_can_be_applied_again_on_restart(client, settings):
    with TestClient(create_app(settings)) as second:
        assert second.get("/api/v1/health").json()["database"] == "ok"


@pytest.mark.parametrize(
    ("collection", "document"),
    [
        ("collections", {"name": 123, "description": None, "created_at": NOW, "updated_at": NOW}),
        ("collections", {"name": "x", "description": None, "created_at": NOW}),  # missing field
        (
            "collections",
            {"name": "x", "description": None, "created_at": NOW, "updated_at": NOW, "extra": 1},
        ),
        (
            "documents",
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
            "chunks",
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
            "chunks",
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
    chunk = mongo_db["chunks"].find_one()
    assert isinstance(chunk["collection_id"], ObjectId)
    assert isinstance(chunk["document_id"], ObjectId)
    assert len(chunk["embedding"]) == 512
    document = mongo_db["documents"].find_one()
    assert document["status"] == "processed"


def test_deleting_document_deletes_its_chunks(client, collection, mongo_db):
    _ingest(client, collection["id"])
    documents = client.get(f"/api/v1/collections/{collection['id']}/documents").json()["items"]
    first = documents[0]["id"]

    response = client.delete(f"/api/v1/collections/{collection['id']}/documents/{first}")
    assert response.status_code == 204
    assert mongo_db["chunks"].count_documents({"document_id": ObjectId(first)}) == 0
    assert mongo_db["chunks"].count_documents({}) > 0  # the other document's chunks remain


def test_deleting_collection_deletes_documents_and_chunks(client, collection, mongo_db):
    _ingest(client, collection["id"])
    other = client.post("/api/v1/collections", json={"name": "other"}).json()
    _ingest(client, other["id"])

    assert client.delete(f"/api/v1/collections/{collection['id']}").status_code == 204

    cid = ObjectId(collection["id"])
    assert mongo_db["documents"].count_documents({"collection_id": cid}) == 0
    assert mongo_db["chunks"].count_documents({"collection_id": cid}) == 0
    oid = ObjectId(other["id"])
    assert mongo_db["documents"].count_documents({"collection_id": oid}) == 2
    assert mongo_db["chunks"].count_documents({"collection_id": oid}) > 0
