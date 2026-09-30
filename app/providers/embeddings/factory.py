from app.core.config import Settings
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.embeddings.hash_provider import HashEmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    match settings.embedding_provider:
        case "hash":
            return HashEmbeddingProvider(dimension=settings.embedding_dimension)
        case "voyage":
            from app.providers.embeddings.voyage_provider import VoyageEmbeddingProvider

            key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else None
            return VoyageEmbeddingProvider(api_key=key, model=settings.voyage_model)
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
