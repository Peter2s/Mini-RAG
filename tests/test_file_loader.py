import pytest

from app.utils.file_loader import DocumentLoadError, UnsupportedFileTypeError, load_pages
from tests.pdf_factory import make_pdf


def test_markdown_is_one_page_without_server_path(tmp_path):
    path = tmp_path / "notes.MD"
    path.write_text("# Title\nbody")
    [page] = load_pages(path)
    assert page.content == "# Title\nbody"
    assert page.metadata == {}  # `source` (the server path) is stripped


def test_txt_keeps_unicode(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("héllo wörld", encoding="utf-8")
    assert [p.content for p in load_pages(path)] == ["héllo wörld"]


def test_pdf_returns_page_content_and_metadata(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(make_pdf("First page text", "Second page text"))

    pages = load_pages(path)

    assert [p.content for p in pages] == ["First page text", "Second page text"]
    assert [p.metadata["page"] for p in pages] == [0, 1]
    assert [p.metadata["page_label"] for p in pages] == ["1", "2"]
    assert all(p.metadata["total_pages"] == 2 for p in pages)
    assert all("source" not in p.metadata for p in pages)


def test_rejects_unknown_type(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(UnsupportedFileTypeError):
        load_pages(path)


def test_load_errors_do_not_leak_server_paths(tmp_path):
    bad_txt = tmp_path / "latin1.txt"
    bad_txt.write_bytes("caf\xe9".encode("latin-1"))
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a pdf")
    for path in (bad_txt, bad_pdf):
        with pytest.raises(DocumentLoadError) as info:
            load_pages(path)
        assert str(tmp_path) not in str(info.value)
