# Mini-RAG

A minimal Retrieval-Augmented Generation API built with FastAPI. Upload documents into a
collection, split them into chunks, embed them, search them, and ask Claude questions answered
from your own documents.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Python 3.12+, and Docker.

```bash
uv sync
cp .env.example .env        # set MONGO_ROOT_PASSWORD (and in MONGODB_URL), QDRANT_API_KEY, ANTHROPIC_API_KEY
docker compose up -d        # MongoDB on localhost:27017, Qdrant on localhost:6333
uv run uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

## Workflow

IDs are MongoDB ObjectIds. Use the `id` from the create response in the later calls.

```bash
# 1. Create a collection and keep its id
COLLECTION=$(curl -s -X POST localhost:8000/api/v1/collections \
  -H 'Content-Type: application/json' -d '{"name": "handbook"}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')

# 2. Upload documents (.txt, .md, .pdf)
curl -X POST localhost:8000/api/v1/collections/$COLLECTION/documents -F file=@handbook.pdf

# 3. Extract text and split into chunks
curl -X POST localhost:8000/api/v1/collections/$COLLECTION/documents/process

# 4. Embed the chunks
curl -X POST localhost:8000/api/v1/collections/$COLLECTION/index

# 5. Search, or ask a question
curl -X POST localhost:8000/api/v1/collections/$COLLECTION/search \
  -H 'Content-Type: application/json' -d '{"query": "vacation policy", "top_k": 5}'
curl -X POST localhost:8000/api/v1/collections/$COLLECTION/answer \
  -H 'Content-Type: application/json' -d '{"question": "How many vacation days do I get?"}'
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
| POST | `/api/v1/collections/{id}/index` | Embed new chunks (`{"do_reset": true}` re-embeds all) |
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
├── db/                  # MongoDB client + collection schemas (validators, indexes)
├── models/              # MongoDB documents: Collection, Document, Chunk
├── schemas/             # Pydantic request/response models
├── routes/              # HTTP layer only: validation and status codes
├── services/            # business logic: CollectionService, DocumentService, RagService
├── repositories/        # persistence: MongoDB repos, vector search, file storage
├── providers/           # external models: embeddings (hash, Voyage), LLM (Claude)
└── utils/               # LangChain loaders and RecursiveCharacterTextSplitter
```

- **Storage** is MongoDB through PyMongo's async client, run locally with `docker compose`.
  The `collections`, `documents`, and `chunks` collections each have a strict JSON Schema
  validator and indexes, defined in `app/db/schema.py` and applied on every startup.
  Deleting a collection or document also deletes its documents and chunks.
- **Vector search** goes through one `VectorRepository` interface (`app/repositories/vector/`)
  with two implementations, picked by `VECTOR_STORE`:
  - `qdrant` (default): Qdrant, started by `docker compose`. Each embedding model gets its own
    Qdrant collection (`chunks__<model>`), created at startup. Inside it, all collections are
    partitioned by a `collection_id` field indexed as a tenant key, which is Qdrant's
    recommended multitenancy layout. Its dashboard is at http://localhost:6333/dashboard.
  - `mongodb`: brute-force cosine similarity over vectors stored on MongoDB chunks. No extra
    service, fine for thousands of chunks per collection.

  Chunk text and metadata always stay in MongoDB, and the vector store holds only vectors and
  ids. Indexing records on each chunk which embedding model indexed it (`embedding_model`)
  and when (`indexed_at`), and deleting or reprocessing documents also deletes their vectors.
  After switching vector stores, re-index each collection with `{"do_reset": true}`.
- **Embeddings** go through one `EmbeddingProvider` interface (`app/providers/embeddings/`):
  `embed_documents`, `embed_query`, plus the model's `model_id` and `dimension`. One instance
  is built at startup and shared by every request. Vectors of different models are never
  mixed: the vector store keeps each model apart, so switching models can't corrupt search.

  They default to FastEmbed, which runs free models locally with no API key.
  The default model, `google/embeddinggemma-300m`, handles Arabic and 100+ other languages,
  and matches English questions to Arabic passages. It downloads once (1.2 GB) into
  `data/models` in the background at startup, then works offline. For a smaller, faster model
  set `FASTEMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  `EMBEDDING_PROVIDER=voyage` uses the hosted Voyage API instead (`uv sync --extra voyage`),
  and `hash` is an offline word matcher used by the tests. After changing the provider or
  model, call `/index` on each collection: chunks not embedded by the new model are re-embedded
  automatically, no reset needed.
- **Answers** come from Claude (`claude-opus-5-5` by default) via the official `anthropic` SDK.
  Server-side refusal fallbacks are on by default; set `LLM_ENABLE_FALLBACKS=false` on
  Bedrock, Vertex, or Foundry.

All settings are listed in `.env.example`.

## Development

```bash
uv sync
uv run pre-commit install   # once per clone: installs the git hooks below
```

| When | Hook | What it checks |
|---|---|---|
| Every commit | pre-commit-hooks | Trailing whitespace, final newlines, LF line endings, valid YAML and TOML, merge markers, large files, private keys |
| Every commit | uv-lock | `uv.lock` matches `pyproject.toml` |
| Every commit | ruff | Lint with autofix, then format. Rules are in `pyproject.toml` |
| Every commit | typos | Spelling in code, comments, and docs. Reports only |
| Every commit | gitleaks | API keys, passwords, and other secrets |
| Every commit | mypy | Strict type checking of `app/` and `tests/` |
| Commit message | conventional-pre-commit | Messages like `feat(scope): description` or `fix: description` |
| Before push | pytest | The full test suite |

Run everything by hand with `uv run pre-commit run --all-files`, or the tools directly:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy
uv run pytest
```

Tests replace Claude with a fake provider and use the `hash` embedder. The test that
runs a real embedding model on Arabic documents is opt-in: `RUN_MODEL_TESTS=1 uv run pytest`. Database tests need MongoDB running
and Qdrant (`docker compose up -d`). Each test uses its own throwaway MongoDB database and
Qdrant collection. Tests connect with `MONGODB_TEST_URL`, `QDRANT_TEST_URL`, and
`QDRANT_TEST_API_KEY` if set, otherwise the app settings, and are skipped when a service is
unreachable. The vector-store contract tests run against both implementations.
