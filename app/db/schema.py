"""MongoDB collection schemas: JSON Schema validators and indexes.

`apply_schema` runs at startup. It creates missing collections and updates the validator of
existing ones, so editing a schema here takes effect on the next restart.
"""

from typing import Any

from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid

from app.core.enums import DocumentStatus, MongoCollection

_OBJECT_ID = {"bsonType": "objectId"}
_DATE = {"bsonType": "date"}
_STRING = {"bsonType": "string"}
_NULLABLE_STRING = {"bsonType": ["string", "null"]}
_NON_NEGATIVE_INT = {"bsonType": ["int", "long"], "minimum": 0}


def _schema(title: str, properties: dict[str, Any]) -> dict[str, Any]:
    """Every field is required and no unknown fields are allowed."""
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "title": title,
            "required": [name for name in properties if name != "_id"],
            "additionalProperties": False,
            "properties": {"_id": _OBJECT_ID, **properties},
        }
    }


VALIDATORS: dict[MongoCollection, dict[str, Any]] = {
    MongoCollection.COLLECTIONS: _schema(
        "Collection",
        {
            "name": {"bsonType": "string", "minLength": 1, "maxLength": 100},
            "description": _NULLABLE_STRING,
            "created_at": _DATE,
            "updated_at": _DATE,
        },
    ),
    MongoCollection.DOCUMENTS: _schema(
        "Document",
        {
            "collection_id": _OBJECT_ID,
            "filename": {"bsonType": "string", "minLength": 1, "maxLength": 255},
            "stored_path": _STRING,
            "content_type": _NULLABLE_STRING,
            "size_bytes": _NON_NEGATIVE_INT,
            "status": {"enum": [status.value for status in DocumentStatus]},
            "error": _NULLABLE_STRING,
            "created_at": _DATE,
            "updated_at": _DATE,
        },
    ),
    MongoCollection.CHUNKS: _schema(
        "Chunk",
        {
            "collection_id": _OBJECT_ID,
            "document_id": _OBJECT_ID,
            "chunk_index": _NON_NEGATIVE_INT,
            "text": _STRING,
            "metadata": {"bsonType": "object"},
            # null until the chunk is indexed
            "embedding": {"bsonType": ["array", "null"], "items": {"bsonType": "double"}},
            "created_at": _DATE,
        },
    ),
}

INDEXES: dict[MongoCollection, list[IndexModel]] = {
    MongoCollection.COLLECTIONS: [
        IndexModel([("name", ASCENDING)], unique=True, name="uniq_name"),
    ],
    MongoCollection.DOCUMENTS: [
        IndexModel([("collection_id", ASCENDING)], name="by_collection"),
    ],
    MongoCollection.CHUNKS: [
        IndexModel([("collection_id", ASCENDING)], name="by_collection"),
        IndexModel(
            [("document_id", ASCENDING), ("chunk_index", ASCENDING)], name="by_document_order"
        ),
    ],
}


async def apply_schema(db: AsyncDatabase) -> None:
    existing = set(await db.list_collection_names())
    for name, validator in VALIDATORS.items():
        options = {"validator": validator, "validationLevel": "strict", "validationAction": "error"}
        if name not in existing:
            try:
                await db.create_collection(name, **options)
            except CollectionInvalid:  # created concurrently by another worker
                await db.command("collMod", name, **options)
        else:
            await db.command("collMod", name, **options)
        await db[name].create_indexes(INDEXES[name])
