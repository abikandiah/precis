import zipfile

import pytest

from precis import book_file
from precis.book_file import BookFileError, read_book_file

CHAPTER = "Being heard releases a person from loneliness, Rogers writes, again and again. "


def _epub(path, chapters: dict[str, str], spine: list[str]):
    with zipfile.ZipFile(path, "w") as epub:
        epub.writestr("mimetype", "application/epub+zip")
        epub.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        items = "".join(f'<item id="{n}" href="text/{n}.xhtml" media-type="application/xhtml+xml"/>' for n in chapters)
        refs = "".join(f'<itemref idref="{n}"/>' for n in spine)
        epub.writestr(
            "OEBPS/content.opf",
            f'<package xmlns="http://www.idpf.org/2007/opf"><manifest>{items}</manifest><spine>{refs}</spine></package>',
        )
        for name, body in chapters.items():
            epub.writestr(f"OEBPS/text/{name}.xhtml", f"<html><head><style>p {{}}</style></head><body>{body}</body></html>")


def test_an_epub_is_read_in_spine_order_without_markup(tmp_path):
    path = tmp_path / "book.epub"
    long = CHAPTER * 300
    _epub(
        path,
        {"two": f"<h1>Two</h1><p>{long}</p>", "one": f"<h1>One</h1><p>{long}</p><script>evil()</script>"},
        spine=["one", "two"],
    )
    text = read_book_file(path)
    assert text.index("One") < text.index("Two")
    assert "evil" not in text and "<p>" not in text and "p {}" not in text


def test_a_text_file_is_read_with_whitespace_tidied(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(("Being   heard\n\n\n" + CHAPTER + "\n") * 300)
    text = read_book_file(path)
    assert "Being heard\n" in text and "\n\n" not in text


def test_a_pdf_text_layer_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr(book_file, "MIN_BOOK_CHARS", 10)
    path = tmp_path / "book.pdf"
    path.write_bytes(_pdf("Being heard releases a person"))
    assert "Being heard releases a person" in read_book_file(path)


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("missing.epub", None, "no such file"),
        ("book.docx", b"x", "book files are .epub, .pdf or .txt"),
        ("book.epub", b"not a zip", "couldn't read book.epub"),
        ("book.pdf", b"not a pdf", "couldn't read book.pdf"),
        ("broken.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 9 0 R >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n", "couldn't read broken.pdf"),
        ("book.txt", b"Too short to be a book.", "a scanned PDF needs OCR first"),
    ],
)
def test_what_isnt_a_readable_book_is_a_clean_error(tmp_path, name, content, message):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(BookFileError, match=message):
        read_book_file(path)


def _pdf(text: str) -> bytes:
    """A one-page PDF whose text layer holds `text`."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return out
