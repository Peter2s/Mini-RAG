"""MongoDB collection schemas: JSON Schema validators, plus the indexes from indexes.py.

`apply_schema` runs at startup. It creates missing collections and updates the validator of
existing ones, so editing a schema here takes effect on the next restart.
"""

from typing import Any

from pymongo.asynchronous.database import AsyncDatabase
from pymongo.errors import CollectionInvalid

from app.core.enums import DocumentStatus, MongoCollection
from app.db.indexes import INDEXES, sync_indexes
from app.db.types import MongoDocument

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
            "content_hash": {"bsonType": "string", "pattern": "^[0-9a-f]{64}$"},
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
            "embedding": {"bsonType": ["array", "null"], "items": {"bsonType": "double"}},
            "embedding_model": _NULLABLE_STRING,
            "indexed_at": {"bsonType": ["date", "null"]},
            "created_at": _DATE,
        },
    ),
}


async def apply_schema(db: AsyncDatabase[MongoDocument]) -> None:
    existing = set(await db.list_collection_names())
    for name, validator in VALIDATORS.items():
        options: dict[str, Any] = {
            "validator": validator,
            "validationLevel": "strict",
            "validationAction": "error",
        }
        if name not in existing:
            try:
                await db.create_collection(name, **options)
            except CollectionInvalid:
                await db.command("collMod", name, **options)
        else:
            await db.command("collMod", name, **options)
        await _backfill(db, name)
        await sync_indexes(db[name], INDEXES[name])


_ADDED_FIELDS: dict[MongoCollection, dict[str, Any]] = {
    MongoCollection.CHUNKS: {"embedding_model": None, "indexed_at": None},
}


async def _backfill(db: AsyncDatabase[MongoDocument], name: MongoCollection) -> None:
    fields = _ADDED_FIELDS.get(name)
    if fields:
        fill_missing = {field: {"$ifNull": [f"${field}", value]} for field, value in fields.items()}
        await db[name].update_many(
            {"$or": [{field: {"$exists": False}} for field in fields]}, [{"$set": fill_missing}]
        )
