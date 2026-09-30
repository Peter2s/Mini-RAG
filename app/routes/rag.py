from fastapi import APIRouter

from app.dependencies import RagServiceDep
from app.schemas.rag import (
    AnswerRequest,
    AnswerResponse,
    IndexRequest,
    IndexResult,
    SearchRequest,
    SearchResponse,
)

router = APIRouter(prefix="/collections/{collection_id}", tags=["rag"])


@router.post("/index")
async def index_collection(
    collection_id: str, service: RagServiceDep, options: IndexRequest | None = None
) -> IndexResult:
    """Embed the collection's processed chunks so they become searchable."""
    return await service.index(collection_id, reset=(options or IndexRequest()).reset)


@router.post("/search")
async def search(collection_id: str, data: SearchRequest, service: RagServiceDep) -> SearchResponse:
    """Return the chunks most similar to the query."""
    return await service.search(collection_id, data.query, data.top_k)


@router.post("/answer")
async def answer(collection_id: str, data: AnswerRequest, service: RagServiceDep) -> AnswerResponse:
    """Retrieve relevant chunks and have Claude answer the question from them."""
    return await service.answer(collection_id, data.question, data.top_k)
