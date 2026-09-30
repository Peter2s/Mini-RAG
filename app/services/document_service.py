import logging
from pathlib import Path

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.enums import FileExtension
from app.core.exceptions import BadRequestError, NotFoundError, PayloadTooLargeError
from app.models import Document, DocumentStatus
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    FileRepository,
)
from app.schemas.common import Page
from app.schemas.document import (
    DocumentRead,
    ProcessCollectionResult,
    ProcessRequest,
    ProcessResult,
)
from app.utils.file_loader import load_pages
from app.utils.text_splitter import split_pages

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        settings: Settings,
        collections: CollectionRepository,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        files: FileRepository,
    ) -> None:
        self.settings = settings
        self.collections = collections
        self.documents = documents
        self.chunks = chunks
        self.files = files

    async def _ensure_collection(self, collection_id: str) -> None:
        if await self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} not found")

    async def upload(self, collection_id: str, upload: UploadFile) -> Document:
        await self._ensure_collection(collection_id)

        filename = Path(upload.filename or "").name
        if not filename:
            raise BadRequestError("Uploaded file has no filename")
        extension = FileExtension.from_path(filename)
        allowed = self.settings.allowed_extensions
        if extension is None or extension not in allowed:
            suffix = Path(filename).suffix.lower() or "none"
            raise BadRequestError(
                f"File type '{suffix}' is not allowed. "
                f"Allowed: {', '.join(sorted(extension.value for extension in allowed))}"
            )

        limit = self.settings.max_upload_size_bytes
        data = await upload.read(limit + 1)
        if len(data) > limit:
            raise PayloadTooLargeError(
                f"File exceeds the {self.settings.max_upload_size_mb} MB upload limit"
            )
        if not data:
            raise BadRequestError("Uploaded file is empty")

        path = await self.files.save(collection_id, filename, data)
        try:
            return await self.documents.add(
                Document(
                    collection_id=collection_id,
                    filename=filename,
                    stored_path=str(path),
                    content_type=upload.content_type,
                    size_bytes=len(data),
                )
            )
        except Exception:
            await self.files.delete(path)
            raise

    async def get(self, collection_id: str, document_id: str) -> Document:
        document = await self.documents.get_for_collection(collection_id, document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found in collection {collection_id}")
        return document

    async def list(self, collection_id: str, offset: int, limit: int) -> Page[DocumentRead]:
        await self._ensure_collection(collection_id)
        items = await self.documents.list_by_collection(collection_id, offset=offset, limit=limit)
        return Page[DocumentRead](
            items=[DocumentRead.model_validate(document) for document in items],
            total=await self.documents.count_by_collection(collection_id),
            offset=offset,
            limit=limit,
        )

    async def delete(self, collection_id: str, document_id: str) -> None:
        document = await self.get(collection_id, document_id)
        stored_path = document.stored_path
        await self.chunks.delete_by_document(document_id)
        await self.documents.delete(document)
        await self.files.delete(stored_path)

    async def process(
        self, collection_id: str, document_id: str, options: ProcessRequest
    ) -> ProcessResult:
        document = await self.get(collection_id, document_id)
        return await self._process_document(document, *self._chunk_params(options))

    async def process_collection(
        self, collection_id: str, options: ProcessRequest
    ) -> ProcessCollectionResult:
        await self._ensure_collection(collection_id)
        chunk_size, chunk_overlap = self._chunk_params(options)
        results = [
            await self._process_document(document, chunk_size, chunk_overlap)
            for document in await self.documents.list_all_by_collection(collection_id)
        ]
        failed = sum(result.status == DocumentStatus.FAILED for result in results)
        return ProcessCollectionResult(
            collection_id=collection_id,
            documents_processed=len(results) - failed,
            documents_failed=failed,
            chunks_created=sum(result.chunks_created for result in results),
            results=results,
        )

    def _chunk_params(self, options: ProcessRequest) -> tuple[int, int]:
        size = options.chunk_size or self.settings.default_chunk_size
        overlap = (
            options.chunk_overlap
            if options.chunk_overlap is not None
            else self.settings.default_chunk_overlap
        )
        if overlap >= size:
            raise BadRequestError("chunk_overlap must be smaller than chunk_size")
        return size, overlap

    async def _process_document(
        self, document: Document, chunk_size: int, chunk_overlap: int
    ) -> ProcessResult:
        try:
            pages = await run_in_threadpool(load_pages, Path(document.stored_path))
            # Pages are split separately, so every chunk keeps the metadata of its page.
            chunks = split_pages(pages, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No text could be extracted from the document")
            created = await self.chunks.replace_for_document(
                document.collection_id, document.id, chunks
            )
        except Exception as error:
            logger.warning("Processing document %s failed: %s", document.id, error)
            document.status = DocumentStatus.FAILED
            document.error = str(error)
            await self.documents.save(document)
            return ProcessResult(
                document_id=document.id, status=document.status, chunks_created=0, error=str(error)
            )

        document.status = DocumentStatus.PROCESSED
        document.error = None
        await self.documents.save(document)
        return ProcessResult(
            document_id=document.id,
            status=document.status,
            pages_loaded=len(pages),
            chunks_created=created,
        )
