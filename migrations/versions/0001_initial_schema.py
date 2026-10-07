"""Initial schema: collections, documents, chunks, and the pgvector extension.

Revision ID: 0001
Revises:
Create Date: 2026-10-07 15:46:30.494444
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "collections",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("char_length(name) >= 1", name=op.f("ck_collections_name_not_empty")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_collections")),
        sa.UniqueConstraint("name", name=op.f("uq_collections_name")),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("stored_path", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="uploaded",
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'", name=op.f("ck_documents_content_hash_is_sha256")
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'processed', 'failed')", name=op.f("ck_documents_status")
        ),
        sa.CheckConstraint("size_bytes >= 0", name=op.f("ck_documents_size_bytes_not_negative")),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            name=op.f("fk_documents_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "collection_id", "content_hash", name=op.f("uq_documents_collection_id_content_hash")
        ),
    )
    op.create_index(
        op.f("ix_documents_collection_id_id"), "documents", ["collection_id", "id"], unique=False
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), server_default=sa.text("uuidv7()"), nullable=False),
        sa.Column("collection_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("chunk_index >= 0", name=op.f("ck_chunks_chunk_index_not_negative")),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["collections.id"],
            name=op.f("fk_chunks_collection_id_collections"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_chunks_document_id_documents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "document_id", "chunk_index", name=op.f("uq_chunks_document_id_chunk_index")
        ),
    )
    op.create_index(
        op.f("ix_chunks_collection_id_id"), "chunks", ["collection_id", "id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_chunks_collection_id_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_documents_collection_id_id"), table_name="documents")
    op.drop_table("documents")
    op.drop_table("collections")
