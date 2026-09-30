"""FastAPI dependency wiring: request -> session -> repositories -> services."""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db import Database
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    FileRepository,
    SQLVectorRepository,
)
from app.services import CollectionService, DocumentService, RagService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    return request.app.state.db


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_llm_provider(request: Request) -> LLMProvider:
    return request.app.state.llm


async def get_session(
    db: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    async with db.session() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_file_repository(settings: SettingsDep) -> FileRepository:
    return FileRepository(settings.upload_dir)


def get_collection_service(
    session: SessionDep, files: Annotated[FileRepository, Depends(get_file_repository)]
) -> CollectionService:
    return CollectionService(CollectionRepository(session), files)


def get_document_service(
    settings: SettingsDep,
    session: SessionDep,
    files: Annotated[FileRepository, Depends(get_file_repository)],
) -> DocumentService:
    return DocumentService(
        settings=settings,
        collections=CollectionRepository(session),
        documents=DocumentRepository(session),
        chunks=ChunkRepository(session),
        files=files,
    )


def get_rag_service(
    settings: SettingsDep,
    session: SessionDep,
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> RagService:
    return RagService(
        settings=settings,
        collections=CollectionRepository(session),
        chunks=ChunkRepository(session),
        vectors=SQLVectorRepository(session),
        embedder=embedder,
        llm=llm,
    )


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
