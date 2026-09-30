"""Chunking built on LangChain's RecursiveCharacterTextSplitter."""

import re
from collections.abc import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas.chunk import LoadedPage, TextChunk

_MANY_NEWLINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\f\v]+")


def normalize_text(text: str) -> str:
    """Collapse runs of spaces and blank lines, which PDF extraction often produces."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Split on paragraphs, then lines, then words, then characters, as needed to fit."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )


def split_pages(
    pages: Sequence[LoadedPage], chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    """Split each page separately. Every chunk gets a copy of its page's metadata."""
    splitter = build_splitter(chunk_size, chunk_overlap)
    documents = splitter.create_documents(
        [normalize_text(page.content) for page in pages],
        metadatas=[page.metadata for page in pages],
    )
    return [TextChunk(text=doc.page_content, metadata=doc.metadata) for doc in documents]
