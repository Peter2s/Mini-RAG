from fastapi import APIRouter, File, Query, UploadFile, status

from app.dependencies import DocumentServiceDep
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
    collection_id: int, service: DocumentServiceDep, file: UploadFile = File(...)
):
    return await service.upload(collection_id, file)


@router.get("", response_model=Page[DocumentRead])
async def list_documents(
    collection_id: int,
    service: DocumentServiceDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    return await service.list(collection_id, offset=offset, limit=limit)


@router.post("/process", response_model=ProcessCollectionResult)
async def process_all_documents(
    collection_id: int, service: DocumentServiceDep, options: ProcessRequest | None = None
):
    """Extract text from every document in the collection and split it into chunks."""
    return await service.process_collection(collection_id, options or ProcessRequest())


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(collection_id: int, document_id: int, service: DocumentServiceDep):
    return await service.get(collection_id, document_id)


@router.post("/{document_id}/process", response_model=ProcessResult)
async def process_document(
    collection_id: int,
    document_id: int,
    service: DocumentServiceDep,
    options: ProcessRequest | None = None,
):
    """Extract text from one document and split it into chunks, replacing old chunks."""
    return await service.process(collection_id, document_id, options or ProcessRequest())


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    collection_id: int, document_id: int, service: DocumentServiceDep
) -> None:
    await service.delete(collection_id, document_id)
