# Mini-RAG

A minimal Retrieval-Augmented Generation API built with FastAPI. Upload documents into a
collection, split them into chunks, embed them, search them, and ask Claude questions answered
from your own documents.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
cp .env.example .env        # then set ANTHROPIC_API_KEY
uv run uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

## Workflow

```bash
# 1. Create a collection
curl -X POST localhost:8000/api/v1/collections -H 'Content-Type: application/json' \
  -d '{"name": "handbook"}'

# 2. Upload documents (.txt, .md, .pdf)
curl -X POST localhost:8000/api/v1/collections/1/documents -F file=@handbook.pdf

# 3. Extract text and split into chunks
curl -X POST localhost:8000/api/v1/collections/1/documents/process

# 4. Embed the chunks
curl -X POST localhost:8000/api/v1/collections/1/index

# 5. Search, or ask a question
curl -X POST localhost:8000/api/v1/collections/1/search -H 'Content-Type: application/json' \
  -d '{"query": "vacation policy", "top_k": 5}'
curl -X POST localhost:8000/api/v1/collections/1/answer -H 'Content-Type: application/json' \
  -d '{"question": "How many vacation days do I get?"}'
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness and database check |
| POST / GET | `/api/v1/collections` | Create / list collections |
| GET / DELETE | `/api/v1/collections/{id}` | Get / delete a collection and everything in it |
| POST / GET | `/api/v1/collections/{id}/documents` | Upload / list documents |
| POST | `/api/v1/collections/{id}/documents/process` | Chunk every document |
| GET / DELETE | `/api/v1/collections/{id}/documents/{doc_id}` | Get / delete a document |
| POST | `/api/v1/collections/{id}/documents/{doc_id}/process` | Chunk one document |
| POST | `/api/v1/collections/{id}/index` | Embed new chunks (`{"reset": true}` re-embeds all) |
| POST | `/api/v1/collections/{id}/search` | Top-k similar chunks |
| POST | `/api/v1/collections/{id}/answer` | Retrieve chunks and answer with Claude |

Errors share one shape: `{"error": {"code": "not_found", "message": "..."}}`.

## Architecture

Requests flow **routes → services → repositories**, wired together in `app/dependencies.py`.

```
app/
├── main.py              # app factory + lifespan (DB, embedder, LLM)
├── dependencies.py      # FastAPI Depends wiring
├── core/                # settings, domain exceptions, logging
├── db/                  # SQLAlchemy base + async engine/session
├── models/              # ORM tables: Collection, Document, Chunk
├── schemas/             # Pydantic request/response models
├── routes/              # HTTP layer only: validation and status codes
├── services/            # business logic: CollectionService, DocumentService, RagService
├── repositories/        # persistence: SQL repos, vector search, file storage
├── providers/           # external models: embeddings (hash, Voyage), LLM (Claude)
└── utils/               # text splitter, LangChain document loaders
```

- **Storage** is SQLite through async SQLAlchemy. Change `DATABASE_URL` to use Postgres
  (install `asyncpg`). Tables are created on startup; add Alembic when the schema starts moving.
- **Vector search** is brute-force cosine similarity over embeddings stored in the `chunks`
  table. That is fine for thousands of chunks per collection. For more, implement the
  `VectorRepository` protocol against Qdrant or pgvector.
- **Embeddings** default to `hash`, an offline lexical embedder that needs no API key. It is
  meant for development and tests. For real semantic search, run `uv sync --extra voyage` and set
  `EMBEDDING_PROVIDER=voyage` plus `VOYAGE_API_KEY`, then re-index with `{"reset": true}`.
- **Answers** come from Claude (`claude-opus-5-5` by default) via the official `anthropic` SDK.
  Server-side refusal fallbacks are on by default; set `LLM_ENABLE_FALLBACKS=false` on
  Bedrock, Vertex, or Foundry.

All settings are listed in `.env.example`.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Tests replace Claude with a fake provider, so they run offline.
