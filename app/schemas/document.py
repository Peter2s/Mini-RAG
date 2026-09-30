from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.enums import DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    collection_id: str
    filename: str
    content_type: str | None
    size_bytes: int
    status: DocumentStatus
    error: str | None
    created_at: datetime
    updated_at: datetime


class ProcessRequest(BaseModel):
    chunk_size: int | None = Field(default=100, ge=100, le=8000)
    chunk_overlap: int | None = Field(default=20, ge=0, le=4000)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "ProcessRequest":
        if (
            self.chunk_size is not None
            and self.chunk_overlap is not None
            and self.chunk_overlap >= self.chunk_size
        ):
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ProcessResult(BaseModel):
    document_id: str
    status: DocumentStatus
    pages_loaded: int = 0
    chunks_created: int
    error: str | None = None


class ProcessCollectionResult(BaseModel):
    collection_id: str
    documents_processed: int
    documents_failed: int
    chunks_created: int
    results: list[ProcessResult]
