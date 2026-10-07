from app.core.config import Settings
from app.core.enums import VectorStoreName
from app.db import MongoDatabase
from app.repositories.vector.base import VectorRepository
from app.repositories.vector.mongo import MongoVectorRepository
from app.repositories.vector.qdrant import QdrantVectorRepository
from app.schemas.vector import EmbeddingSpace


def build_vector_repository(
    settings: Settings, mongo: MongoDatabase, space: EmbeddingSpace
) -> VectorRepository:
    match settings.vector_store:
        case VectorStoreName.QDRANT:
            api_key = (
                settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
            )
            return QdrantVectorRepository(
                url=settings.qdrant_url,
                api_key=api_key,
                collection_prefix=settings.qdrant_collection,
                space=space,
            )
        case VectorStoreName.MONGODB:
            return MongoVectorRepository(mongo.db, space)
    raise ValueError(f"Unknown vector store: {settings.vector_store}")
