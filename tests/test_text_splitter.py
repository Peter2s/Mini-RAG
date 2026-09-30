import pytest

from app.utils.text_splitter import split_text


def test_short_text_is_single_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_empty_text_has_no_chunks():
    assert split_text("   \n\n  ", 100, 10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # consecutive chunks share some words thanks to the overlap
    assert set(chunks[0].split()[-3:]) & set(chunks[1].split()[:10])
    # no chunk starts mid-word
    assert all(c.split()[0].startswith("word") for c in chunks)


def test_prefers_paragraph_boundaries():
    text = "A" * 80 + "\n\n" + "B" * 80
    assert split_text(text, 120, 0) == ["A" * 80, "B" * 80]


def test_long_unbroken_text_still_progresses():
    chunks = split_text("x" * 1000, 100, 20)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c[20:] if i else c for i, c in enumerate(chunks)) == "x" * 1000


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)
