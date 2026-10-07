from datetime import UTC, datetime

import pytest
from bson import ObjectId
from fastapi.testclient import TestClient
from pymongo.errors import WriteError

from app.core.enums import MongoCollection
from app.db.indexes import INDEXES
from app.db.schema import VALIDATORS
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
        assert index_names == {"_id_"} | {index.document["name"] for index in INDEXES[name]}
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
        ),
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
                "content_hash": "0" * 64,
                "status": "bogus",
                "error": None,
                "created_at": NOW,
                "updated_at": NOW,
            },
        ),
        (
            MongoCollection.DOCUMENTS,
            {
                "collection_id": ObjectId(),
                "filename": "a.txt",
                "stored_path": "/tmp/a.txt",
                "content_type": None,
                "size_bytes": 1,
                "content_hash": "not-a-sha256",
                "status": "uploaded",
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
        content = f"{name} has some text. ".encode() * 20
        client.post(f"{base}/documents", files={"file": (name, content, "text/plain")})
    client.post(f"{base}/documents/process")
    client.post(f"{base}/index")


def test_stored_documents_use_object_ids_and_embeddings(client, collection, mongo_db):
    _ingest(client, collection["id"])
    chunk = mongo_db[MongoCollection.CHUNKS].find_one()
    assert isinstance(chunk["collection_id"], ObjectId)
    assert isinstance(chunk["document_id"], ObjectId)
    assert len(chunk["embedding"]) == 512
    assert chunk["embedding_model"] == "hash:512"
    assert isinstance(chunk["indexed_at"], datetime)
    document = mongo_db[MongoCollection.DOCUMENTS].find_one()
    assert document["status"] == "processed"


def test_deleting_document_deletes_its_chunks(client, collection, mongo_db):
    _ingest(client, collection["id"])
    documents = client.get(f"/api/v1/collections/{collection['id']}/documents").json()["items"]
    first = documents[0]["id"]

    response = client.delete(f"/api/v1/collections/{collection['id']}/documents/{first}")
    assert response.status_code == 204
    assert mongo_db[MongoCollection.CHUNKS].count_documents({"document_id": ObjectId(first)}) == 0
    assert mongo_db[MongoCollection.CHUNKS].count_documents({}) > 0


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


def test_startup_backfills_fields_added_to_existing_chunks(client, collection, settings, mongo_db):
    legacy = {
        "collection_id": ObjectId(collection["id"]),
        "document_id": ObjectId(),
        "chunk_index": 0,
        "text": "stored before embedding_model existed",
        "metadata": {},
        "embedding": None,
        "created_at": NOW,
    }
    chunks = mongo_db[MongoCollection.CHUNKS]
    chunks.insert_one(legacy, bypass_document_validation=True)
    indexed_at = datetime(2026, 1, 1, tzinfo=UTC)
    chunks.insert_one(
        {**legacy, "_id": ObjectId(), "chunk_index": 1, "indexed_at": indexed_at},
        bypass_document_validation=True,
    )

    with TestClient(create_app(settings)):
        pass

    first, second = chunks.find().sort("chunk_index", 1)
    assert (first["embedding_model"], first["indexed_at"]) == (None, None)
    assert (second["embedding_model"], second["indexed_at"]) == (None, indexed_at)
