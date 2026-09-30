"""FastAPI dependency wiring: request -> database -> repositories -> services."""

from typing import Annotated

from fastapi import Depends, Request
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import Settings
from app.db import MongoDatabase
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    FileRepository,
    MongoVectorRepository,
)
from app.services import CollectionService, DocumentService, RagService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_mongo(request: Request) -> MongoDatabase:
    return request.app.state.mongo


def get_db(mongo: Annotated[MongoDatabase, Depends(get_mongo)]) -> AsyncDatabase:
    return mongo.db


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_llm_provider(request: Request) -> LLMProvider:
    return request.app.state.llm


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[AsyncDatabase, Depends(get_db)]


def get_file_repository(settings: SettingsDep) -> FileRepository:
    return FileRepository(settings.upload_dir)


FilesDep = Annotated[FileRepository, Depends(get_file_repository)]


def get_collection_service(db: DbDep, files: FilesDep) -> CollectionService:
    return CollectionService(
        collections=CollectionRepository(db),
        documents=DocumentRepository(db),
        chunks=ChunkRepository(db),
        files=files,
    )


def get_document_service(settings: SettingsDep, db: DbDep, files: FilesDep) -> DocumentService:
    return DocumentService(
        settings=settings,
        collections=CollectionRepository(db),
        documents=DocumentRepository(db),
        chunks=ChunkRepository(db),
        files=files,
    )


def get_rag_service(
    settings: SettingsDep,
    db: DbDep,
    embedder: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> RagService:
    return RagService(
        settings=settings,
        collections=CollectionRepository(db),
        chunks=ChunkRepository(db),
        vectors=MongoVectorRepository(db),
        embedder=embedder,
        llm=llm,
    )


CollectionServiceDep = Annotated[CollectionService, Depends(get_collection_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
