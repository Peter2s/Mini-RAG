from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """A chunk returned by vector search, before it is shaped into an API response."""

    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any]


class IndexRequest(BaseModel):
    reset: bool = Field(default=False, description="Re-embed every chunk, not only new ones.")


class IndexResult(BaseModel):
    collection_id: int
    chunks_indexed: int
    total_chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    text: str
    score: float
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Loader metadata, e.g. page and page_label for PDFs."
    )


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerResponse(BaseModel):
    question: str
    answer: str
    model: str
    sources: list[SearchHit]
