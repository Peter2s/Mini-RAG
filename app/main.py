from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.db import MongoDatabase
from app.providers.embeddings import build_embedding_provider
from app.providers.llm import AnthropicLLMProvider
from app.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    mongo = MongoDatabase(settings.mongodb_url.get_secret_value(), settings.mongodb_database)
    await mongo.init()
    app.state.mongo = mongo
    app.state.embedder = build_embedding_provider(settings)
    app.state.llm = AnthropicLLMProvider(settings)
    try:
        yield
    finally:
        await app.state.llm.aclose()
        await app.state.embedder.aclose()
        await mongo.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.debug)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A minimal Retrieval-Augmented Generation API.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
