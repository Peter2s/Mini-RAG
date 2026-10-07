import logging
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.enums import DocumentStatus, FileExtension
from app.core.exceptions import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
)
from app.models import Collection, Document
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    FileRepository,
    VectorRepository,
)
from app.schemas.common import Page
from app.schemas.document import (
    DocumentRead,
    ProcessCollectionResult,
    ProcessRequest,
    ProcessResult,
)
from app.utils.file_loader import load_pages
from app.utils.hashing import sha256_hex
from app.utils.text_splitter import split_pages

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        collections: CollectionRepository,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        files: FileRepository,
        vectors: VectorRepository,
    ) -> None:
        self.session = session
        self.settings = settings
        self.collections = collections
        self.documents = documents
        self.chunks = chunks
        self.files = files
        self.vectors = vectors

    async def _get_collection(self, collection_id: str) -> Collection:
        collection = await self.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"Collection {collection_id} not found")
        return collection

    async def upload(self, collection_id: str, upload: UploadFile) -> Document:
        collection = await self._get_collection(collection_id)

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

        content_hash = sha256_hex(data)
        existing = await self.documents.get_by_content_hash(collection.id, content_hash)
        if existing is not None:
            raise ConflictError(
                f"A file with the same content is already in this collection: "
                f"'{existing.filename}' (document {existing.id})"
            )

        path = await self.files.save(str(collection.id), f"{content_hash}{extension}", data)
        try:
            document = await self.documents.add(
                Document(
                    collection_id=collection.id,
                    filename=filename,
                    stored_path=str(path),
                    content_type=upload.content_type,
                    size_bytes=len(data),
                    content_hash=content_hash,
                )
            )
            await self.session.commit()
        except ConflictError:
            raise
        except Exception:
            await self.files.delete(path)
            raise
        return document

    async def get(self, collection_id: str, document_id: str) -> Document:
        document = await self.documents.get_for_collection(collection_id, document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found in collection {collection_id}")
        return document

    async def list(self, collection_id: str, offset: int, limit: int) -> Page[DocumentRead]:
        collection = await self._get_collection(collection_id)
        items = await self.documents.list_by_collection(collection.id, offset=offset, limit=limit)
        return Page[DocumentRead](
            items=[DocumentRead.model_validate(document) for document in items],
            total=await self.documents.count_by_collection(collection.id),
            offset=offset,
            limit=limit,
        )

    async def delete(self, collection_id: str, document_id: str) -> None:
        """Delete a document. Its chunks go with it (ON DELETE CASCADE)."""
        document = await self.get(collection_id, document_id)
        await self.vectors.delete_document(str(document.collection_id), str(document.id))
        await self.documents.delete(document)
        await self.session.commit()
        await self.files.delete(document.stored_path)

    async def process(
        self, collection_id: str, document_id: str, options: ProcessRequest
    ) -> ProcessResult:
        document = await self.get(collection_id, document_id)
        return await self._process_document(document, *self._chunk_params(options))

    async def process_collection(
        self, collection_id: str, options: ProcessRequest
    ) -> ProcessCollectionResult:
        collection = await self._get_collection(collection_id)
        chunk_size, chunk_overlap = self._chunk_params(options)
        results = [
            await self._process_document(document, chunk_size, chunk_overlap)
            for document in await self.documents.list_all_by_collection(collection.id)
        ]
        failed = sum(result.status == DocumentStatus.FAILED for result in results)
        return ProcessCollectionResult(
            collection_id=str(collection.id),
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
        """Chunk one document and commit the outcome, so each document's result is kept."""
        try:
            pages = await run_in_threadpool(load_pages, Path(document.stored_path))
            chunks = split_pages(pages, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError("No text could be extracted from the document")
            await self.vectors.delete_document(str(document.collection_id), str(document.id))
            created = await self.chunks.replace_for_document(
                document.collection_id, document.id, chunks
            )
        except Exception as error:
            logger.warning("Processing document %s failed: %s", document.id, error)
            document.status = DocumentStatus.FAILED
            document.error = str(error)
            await self.documents.save(document)
            await self.session.commit()
            return ProcessResult(
                document_id=document.id, status=document.status, chunks_created=0, error=str(error)
            )

        document.status = DocumentStatus.PROCESSED
        document.error = None
        await self.documents.save(document)
        await self.session.commit()
        return ProcessResult(
            document_id=document.id,
            status=document.status,
            pages_loaded=len(pages),
            chunks_created=created,
        )
