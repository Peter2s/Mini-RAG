from app.core.config import Settings
from app.core.enums import EmbeddingProviderName
from app.providers.embeddings.base import EmbeddingProvider
from app.providers.embeddings.hash_provider import HashEmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    match settings.embedding_provider:
        case EmbeddingProviderName.HASH:
            return HashEmbeddingProvider(dimension=settings.embedding_dimension)
        case EmbeddingProviderName.FASTEMBED:
            from app.providers.embeddings.fastembed_provider import FastEmbedProvider

            return FastEmbedProvider(
                model_name=settings.fastembed_model,
                models_dir=settings.models_dir,
                threads=settings.fastembed_threads,
            )
        case EmbeddingProviderName.VOYAGE:
            from app.providers.embeddings.voyage_provider import VoyageEmbeddingProvider

            key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else None
            return VoyageEmbeddingProvider(
                api_key=key, model=settings.voyage_model, dimension=settings.voyage_dimension
            )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")
