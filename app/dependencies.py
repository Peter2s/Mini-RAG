"""FastAPI dependency wiring: request -> database session -> repositories -> services."""

from collections.abc import AsyncIterator
from typing import Annotated, cast

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
    VectorRepository,
)
from app.services import CollectionService, DocumentService, RagService


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return cast(EmbeddingProvider, request.app.state.embedder)


def get_vector_repository(request: Request) -> VectorRepository:
    return cast(VectorRepository, request.app.state.vectors)


def get_llm_provider(request: Request) -> LLMProvider:
    return cast(LLMProvider, request.app.state.llm)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]


async def get_session(database: DatabaseDep) -> AsyncIterator[AsyncSession]:
    """One session per request, shared by every repository in it.

    Commits when the endpoint succeeds and rolls back when it raises. The "function" scope
    runs this before the response is sent, so a failed commit is an error, not a lost write.
    """
    async with database.sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        await session.commit()


SessionDep = Annotated[AsyncSession, Depends(get_session, scope="function")]


def get_file_repository(settings: SettingsDep) -> FileRepository:
    return FileRepository(settings.upload_dir)


FilesDep = Annotated[FileRepository, Depends(get_file_repository)]
VectorsDep = Annotated[VectorRepository, Depends(get_vector_repository)]


def get_collection_service(
    session: SessionDep, files: FilesDep, vectors: VectorsDep
) -> CollectionService:
    return CollectionService(
        session=session,
        collections=CollectionRepository(session),
        files=files,
        vectors=vectors,
    )


def get_document_service(
    settings: SettingsDep, session: SessionDep, files: FilesDep, vectors: VectorsDep
) -> DocumentService:
    return DocumentService(
        session=session,
        settings=settings,
        collections=CollectionRepository(session),
        documents=DocumentRepository(session),
        chunks=ChunkRepository(session),
        files=files,
        vectors=vectors,
    )


def get_rag_service(
    settings: SettingsDep,
    session: SessionDep,
    vectors: VectorsDep,
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> RagService:
    return RagService(
        session=session,
        settings=settings,
        collections=CollectionRepository(session),
        chunks=ChunkRepository(session),
        documents=DocumentRepository(session),
        vectors=vectors,
        embedder=embedder,
        llm=llm,
    )


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
