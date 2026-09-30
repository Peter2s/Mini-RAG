import pytest

from app.schemas.chunk import LoadedPage
from app.utils.text_splitter import normalize_text, split_pages


def _texts(chunks):
    return [chunk.text for chunk in chunks]


def test_short_page_is_single_chunk_with_metadata():
    [chunk] = split_pages([LoadedPage("Hello world.", {"page": 0})], 100, 10)
    assert chunk.text == "Hello world."
    assert chunk.metadata == {"page": 0, "start_index": 0}


def test_empty_pages_produce_no_chunks():
    assert split_pages([LoadedPage("   \n\n  "), LoadedPage("")], 100, 10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{number}" for number in range(500))
    chunks = _texts(split_pages([LoadedPage(text)], 200, 40))
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)
    # consecutive chunks share words thanks to the overlap
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:10])
    # words are never cut in half
    assert all(word.startswith("word") for chunk in chunks for word in chunk.split())


def test_prefers_paragraph_boundaries():
    text = "A" * 80 + "\n\n" + "B" * 80
    assert _texts(split_pages([LoadedPage(text)], 120, 0)) == ["A" * 80, "B" * 80]


def test_each_chunk_keeps_its_own_page_metadata_and_offset():
    pages = [
        LoadedPage("alpha " * 60, {"page": 0, "page_label": "1"}),
        LoadedPage("beta " * 60, {"page": 1, "page_label": "2"}),
    ]
    chunks = split_pages(pages, 100, 20)

    for chunk in chunks:
        page = pages[chunk.metadata["page"]]
        normalized = normalize_text(page.content)
        start = chunk.metadata["start_index"]
        assert normalized[start : start + len(chunk.text)] == chunk.text
        assert chunk.text.split()[0] == ("alpha" if chunk.metadata["page"] == 0 else "beta")

    # metadata is copied per chunk, not shared
    chunks[0].metadata["page_label"] = "changed"
    assert chunks[1].metadata["page_label"] == "1"
    assert pages[0].metadata["page_label"] == "1"


def test_normalizes_whitespace_before_splitting():
    [chunk] = split_pages([LoadedPage("a   b\t\tc\n\n\n\nd")], 100, 0)
    assert chunk.text == "a b c\n\nd"


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_pages([LoadedPage("text")], size, overlap)
