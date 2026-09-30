from html import escape

from app.core.config import Settings
from app.core.exceptions import BadRequestError, NotFoundError
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.repositories import ChunkRepository, CollectionRepository, VectorRepository
from app.schemas.rag import AnswerResponse, IndexResult, SearchHit, SearchResponse, VectorMatch

SYSTEM_PROMPT = """\
You answer questions using only the documents provided from the user's knowledge base.

- Ground every statement in the documents. If they do not contain the answer, say so plainly \
instead of guessing or using outside knowledge.
- Cite the documents you rely on inline by id, like [1] or [2][3].
- Answer directly and concisely, in the language of the question."""


class RagService:
    def __init__(
        self,
        settings: Settings,
        collections: CollectionRepository,
        chunks: ChunkRepository,
        vectors: VectorRepository,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.collections = collections
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm

    async def _ensure_collection(self, collection_id: int) -> None:
        if await self.collections.get(collection_id) is None:
            raise NotFoundError(f"Collection {collection_id} not found")

    async def index(self, collection_id: int, reset: bool = False) -> IndexResult:
        """Embed chunks that have no embedding yet (or all chunks when `reset` is set)."""
        await self._ensure_collection(collection_id)
        pending = await self.chunks.list_for_indexing(collection_id, only_missing=not reset)

        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = await self.embedder.embed([c.text for c in batch], input_type="document")
            await self.chunks.set_embeddings(
                {chunk.id: vector for chunk, vector in zip(batch, vectors, strict=True)}
            )

        return IndexResult(
            collection_id=collection_id,
            chunks_indexed=len(pending),
            total_chunks=await self.chunks.count_by_collection(collection_id),
        )

    async def search(self, collection_id: int, query: str, top_k: int) -> SearchResponse:
        await self._ensure_collection(collection_id)
        matches = await self._retrieve(collection_id, query, top_k)
        return SearchResponse(query=query, results=[_to_hit(m) for m in matches])

    async def answer(self, collection_id: int, question: str, top_k: int) -> AnswerResponse:
        await self._ensure_collection(collection_id)
        matches = await self._retrieve(collection_id, question, top_k)
        if not matches:
            raise BadRequestError(
                "This collection has no indexed content. "
                "Upload, process, and index documents first."
            )

        response = await self.llm.generate(
            system=SYSTEM_PROMPT, prompt=build_prompt(question, matches)
        )
        return AnswerResponse(
            question=question,
            answer=response.text,
            model=response.model,
            sources=[_to_hit(m) for m in matches],
        )

    async def _retrieve(self, collection_id: int, query: str, top_k: int) -> list[VectorMatch]:
        [query_vector] = await self.embedder.embed([query], input_type="query")
        try:
            return await self.vectors.search(collection_id, query_vector, top_k)
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc


def build_prompt(question: str, matches: list[VectorMatch]) -> str:
    documents = "\n".join(_render_document(i, m) for i, m in enumerate(matches, start=1))
    return f"<documents>\n{documents}\n</documents>\n\nQuestion: {question}"


def _render_document(doc_id: int, match: VectorMatch) -> str:
    attributes = {"id": doc_id, "source": match.filename, "chunk": match.chunk_index}
    if (page := match.metadata.get("page_label")) is not None:
        attributes["page"] = page
    rendered = " ".join(f'{key}="{escape(str(value))}"' for key, value in attributes.items())
    return f"<document {rendered}>\n{match.text}\n</document>"


def _to_hit(match: VectorMatch) -> SearchHit:
    return SearchHit(
        chunk_id=match.chunk_id,
        document_id=match.document_id,
        filename=match.filename,
        chunk_index=match.chunk_index,
        text=match.text,
        score=round(match.score, 4),
        metadata=match.metadata,
    )
