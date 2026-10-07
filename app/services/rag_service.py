from html import escape

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import BadRequestError, EmbeddingError, NotFoundError
from app.models import Collection
from app.providers.embeddings import EmbeddingProvider
from app.providers.llm import LLMProvider
from app.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    VectorRepository,
)
from app.schemas.rag import AnswerResponse, IndexResult, SearchHit, SearchResponse, VectorMatch
from app.schemas.vector import VectorHit, VectorPoint

SYSTEM_PROMPT = """\
You answer questions using only the documents provided from the user's knowledge base.

- Ground every statement in the documents. If they do not contain the answer, say so plainly \
instead of guessing or using outside knowledge.
- Cite the documents you rely on inline by id, like [1] or [2][3].
- Answer directly and concisely, in the language of the question."""


class RagService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        collections: CollectionRepository,
        chunks: ChunkRepository,
        documents: DocumentRepository,
        vectors: VectorRepository,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        self.session = session
        self.settings = settings
        self.collections = collections
        self.chunks = chunks
        self.documents = documents
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm

    async def _get_collection(self, collection_id: str) -> Collection:
        collection = await self.collections.get(collection_id)
        if collection is None:
            raise NotFoundError(f"Collection {collection_id} not found")
        return collection

    async def index(self, collection_id: str, do_reset: bool = False) -> IndexResult:
        """Embed chunks not yet indexed by the current model (every chunk with `do_reset`)."""
        collection = await self._get_collection(collection_id)
        key = str(collection.id)
        if do_reset:
            await self.chunks.mark_unindexed(collection.id)
            await self.session.commit()
            await self.vectors.delete_collection(key)
        model_id = self.embedder.model_id
        pending = await self.chunks.list_for_indexing(collection.id, embedding_model=model_id)

        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = await self.embedder.embed_documents([chunk.text for chunk in batch])
            points = [
                VectorPoint(
                    chunk_id=str(chunk.id), document_id=str(chunk.document_id), vector=vector
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            try:
                await self.vectors.upsert(key, points)
            except ValueError as error:
                raise EmbeddingError(str(error)) from error
            await self.chunks.mark_indexed([chunk.id for chunk in batch], model_id)
            await self.session.commit()

        return IndexResult(
            collection_id=key,
            chunks_indexed=len(pending),
            total_chunks=await self.chunks.count_by_collection(collection.id),
        )

    async def search(self, collection_id: str, query: str, top_k: int) -> SearchResponse:
        collection = await self._get_collection(collection_id)
        matches = await self._retrieve(str(collection.id), query, top_k)
        return SearchResponse(query=query, results=[_to_hit(match) for match in matches])

    async def answer(self, collection_id: str, question: str, top_k: int) -> AnswerResponse:
        collection = await self._get_collection(collection_id)
        matches = await self._retrieve(str(collection.id), question, top_k)
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
            sources=[_to_hit(match) for match in matches],
        )

    async def _retrieve(self, collection_id: str, query: str, top_k: int) -> list[VectorMatch]:
        query_vector = await self.embedder.embed_query(query)
        try:
            hits = await self.vectors.search(collection_id, query_vector, top_k)
        except ValueError as error:
            raise EmbeddingError(str(error)) from error
        return await self._to_matches(hits)

    async def _to_matches(self, hits: list[VectorHit]) -> list[VectorMatch]:
        """Attach text, filename, and metadata from PostgreSQL to vector hits, in hit order."""
        chunks = {
            str(chunk.id): chunk
            for chunk in await self.chunks.get_many(hit.chunk_id for hit in hits)
        }
        filenames = await self.documents.filenames({chunk.document_id for chunk in chunks.values()})
        return [
            VectorMatch(
                chunk_id=str(chunk.id),
                document_id=str(chunk.document_id),
                filename=filenames.get(chunk.document_id, "unknown"),
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=hit.score,
                metadata=chunk.meta,
            )
            for hit in hits
            if (chunk := chunks.get(hit.chunk_id)) is not None
        ]


def build_prompt(question: str, matches: list[VectorMatch]) -> str:
    documents = "\n".join(
        _render_document(position, match) for position, match in enumerate(matches, start=1)
    )
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
