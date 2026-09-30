from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status

from app.dependencies import DocumentServiceDep
from app.models import Document
from app.schemas.common import Page
from app.schemas.document import (
    DocumentRead,
    ProcessCollectionResult,
    ProcessRequest,
    ProcessResult,
)

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    collection_id: str, service: DocumentServiceDep, file: Annotated[UploadFile, File()]
) -> Document:
    return await service.upload(collection_id, file)


@router.get("")
async def list_documents(
    collection_id: str,
    service: DocumentServiceDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> Page[DocumentRead]:
    return await service.list(collection_id, offset=offset, limit=limit)


@router.post("/process")
async def process_all_documents(
    collection_id: str, service: DocumentServiceDep, options: ProcessRequest | None = None
) -> ProcessCollectionResult:
    """Extract text from every document in the collection and split it into chunks."""
    return await service.process_collection(collection_id, options or ProcessRequest())


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    collection_id: str, document_id: str, service: DocumentServiceDep
) -> Document:
    return await service.get(collection_id, document_id)


@router.post("/{document_id}/process")
async def process_document(
    collection_id: str,
    document_id: str,
    service: DocumentServiceDep,
    options: ProcessRequest | None = None,
) -> ProcessResult:
    """Extract text from one document and split it into chunks, replacing old chunks."""
    return await service.process(collection_id, document_id, options or ProcessRequest())


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    collection_id: str, document_id: str, service: DocumentServiceDep
) -> None:
    await service.delete(collection_id, document_id)
