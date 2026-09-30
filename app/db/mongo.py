from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.db.schema import apply_schema
from app.db.types import MongoDocument


class MongoDatabase:
    """Owns the MongoDB client for the app's lifetime."""

    def __init__(self, url: str, name: str) -> None:
        self.client: AsyncMongoClient[MongoDocument] = AsyncMongoClient(
            url, tz_aware=True, serverSelectionTimeoutMS=5000
        )
        self.db: AsyncDatabase[MongoDocument] = self.client[name]

    async def init(self) -> None:
        """Create collections, validators, and indexes. Safe to run on every startup."""
        await apply_schema(self.db)

    async def ping(self) -> bool:
        await self.client.admin.command("ping")
        return True

    async def close(self) -> None:
        await self.client.close()
