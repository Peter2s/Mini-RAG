"""FastAPI dependency wiring: request -> database -> repositories -> services."""

from typing import Annotated, cast

from fastapi import Depends, Request
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.db import MongoDatabase
from app.db.types import MongoDocument
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


def get_mongo(request: Request) -> MongoDatabase:
    return cast(MongoDatabase, request.app.state.mongo)


def get_db(mongo: Annotated[MongoDatabase, Depends(get_mongo)]) -> AsyncDatabase[MongoDocument]:
    return mongo.db


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return cast(EmbeddingProvider, request.app.state.embedder)


def get_vector_repository(request: Request) -> VectorRepository:
    return cast(VectorRepository, request.app.state.vectors)


def get_llm_provider(request: Request) -> LLMProvider:
    return cast(LLMProvider, request.app.state.llm)


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[AsyncDatabase[MongoDocument], Depends(get_db)]


def get_file_repository(settings: SettingsDep) -> FileRepository:
    return FileRepository(settings.upload_dir)


FilesDep = Annotated[FileRepository, Depends(get_file_repository)]
VectorsDep = Annotated[VectorRepository, Depends(get_vector_repository)]


def get_collection_service(db: DbDep, files: FilesDep, vectors: VectorsDep) -> CollectionService:
    return CollectionService(
        collections=CollectionRepository(db),
        documents=DocumentRepository(db),
        chunks=ChunkRepository(db),
        files=files,
        vectors=vectors,
    )


def get_document_service(
    settings: SettingsDep, db: DbDep, files: FilesDep, vectors: VectorsDep
) -> DocumentService:
    return DocumentService(
        settings=settings,
        collections=CollectionRepository(db),
        documents=DocumentRepository(db),
        chunks=ChunkRepository(db),
        files=files,
        vectors=vectors,
    )


def get_rag_service(
    settings: SettingsDep,
    db: DbDep,
    vectors: VectorsDep,
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> RagService:
    return RagService(
        settings=settings,
        collections=CollectionRepository(db),
        chunks=ChunkRepository(db),
        documents=DocumentRepository(db),
        vectors=vectors,
        embedder=embedder,
        llm=llm,
    )


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
