"""MongoDB indexes, designed from the queries the repositories run.

Guidelines followed:
- Every query is served by an index; tests/test_mongo_indexes.py checks the plans with explain().
- Compound keys follow the ESR rule: Equality fields first, then the Sort field, then Ranges.
  Queries filter on `collection_id` and sort by `_id`, so the key is (collection_id, _id).
- Unique indexes enforce invariants the app relies on, instead of trusting application code.
- Every index is named explicitly, and no index is a prefix of another (redundant) one.

`sync_indexes` makes the database match these definitions on startup: it drops indexes that
are no longer declared or whose definition changed, then creates whatever is missing.
MongoDB 4.2+ builds indexes without blocking reads and writes, so no `background` flag is needed.
"""

import logging
from collections.abc import Mapping
from typing import Any

from pymongo import ASCENDING, IndexModel
from pymongo.asynchronous.collection import AsyncCollection

from app.core.enums import MongoCollection

logger = logging.getLogger(__name__)

INDEXES: dict[MongoCollection, list[IndexModel]] = {
    MongoCollection.COLLECTIONS: [
        # get_by_name(), and collection names must be unique.
        # Listing all collections sorts by _id, which the built-in _id index serves.
        IndexModel([("name", ASCENDING)], name="uniq_name", unique=True),
    ],
    MongoCollection.DOCUMENTS: [
        # list/count/delete a collection's documents; listing is sorted by _id (paginated).
        IndexModel([("collection_id", ASCENDING), ("_id", ASCENDING)], name="collection_id_order"),
    ],
    MongoCollection.CHUNKS: [
        # Chunks to embed (sorted by _id), vector search candidates, count/delete by collection.
        IndexModel([("collection_id", ASCENDING), ("_id", ASCENDING)], name="collection_id_order"),
        # Delete a document's chunks; a document can't have two chunks at the same position.
        IndexModel(
            [("document_id", ASCENDING), ("chunk_index", ASCENDING)],
            name="uniq_document_chunk_index",
            unique=True,
        ),
    ],
}


_DEFINING_OPTIONS = (
    "unique",
    "sparse",
    "partialFilterExpression",
    "collation",
    "expireAfterSeconds",
)


def _definition(spec: Mapping[str, Any]) -> tuple[Any, ...]:
    key = spec["key"]
    fields = key.items() if isinstance(key, Mapping) else key
    normalized_key = tuple((field, int(direction)) for field, direction in fields)
    options = tuple(
        (option, bool(spec.get(option)) if option in ("unique", "sparse") else spec.get(option))
        for option in _DEFINING_OPTIONS
    )
    return normalized_key, options


async def sync_indexes(collection: AsyncCollection, indexes: list[IndexModel]) -> None:
    declared = {index.document["name"]: index.document for index in indexes}
    existing = await collection.index_information()

    for name, spec in existing.items():
        if name == "_id_":
            continue
        wanted = declared.get(name)
        if wanted is None or _definition(spec) != _definition(wanted):
            reason = "no longer declared" if wanted is None else "definition changed"
            logger.warning("Dropping index %s.%s (%s)", collection.name, name, reason)
            await collection.drop_index(name)

    # Creating an index that already exists with the same definition is a no-op.
    await collection.create_indexes(indexes)
