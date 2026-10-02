"""The reader's own copy of the book: an EPUB, PDF or plain-text file named
by the known-file's `book_file` (docs/blueprint.md, Input). Its text joins
the research as a source of its own, and digest.py reads it whole like any
long page — so a book is noted from all of it whether or not a copy turns
up in search.

Text only: EPUB chapters in reading order, a PDF's text layer, or a text
file as it is. A scanned PDF without a text layer has nothing to read, and
fails rather than adding an empty source.

DRM-free copies only: a DRM-locked EPUB's chapters are ciphertext, which
would otherwise read as tens of thousands of characters of garbage and pass
the length check, and a locked PDF has no text to read without its key. Both
fail with an error saying so; precis never removes DRM.
"""

from __future__ import annotations

import posixpath
import re
import traceback
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

# Less text than this isn't a book: a scanned PDF's empty text layer, or a
# file that isn't the book at all.
MIN_BOOK_CHARS = 20_000

# Files a DRM scheme adds to an EPUB: Adobe ADEPT's rights.xml, Apple
# FairPlay's sinf.xml.
_EPUB_DRM_FILES = ("META-INF/rights.xml", "META-INF/sinf.xml")
# encryption.xml algorithms that only obfuscate embedded fonts (IDPF and
# Adobe's), which DRM-free EPUBs use too; anything else encrypts content.
_FONT_OBFUSCATION = frozenset(["http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC"])

_BLOCKS = frozenset(["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote", "section"])


class BookFileError(ValueError):
    """The book file can't be read as a book."""


def _locked(name: str) -> BookFileError:
    return BookFileError(f"{name} is DRM-locked or password-protected — precis reads DRM-free copies only")


class _Text(HTMLParser):
    """An XHTML chapter's text, a line per block, without scripts or styles."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_text(html: str) -> str:
    parser = _Text()
    parser.feed(html)
    return "".join(parser.parts)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _epub_text(path: Path) -> str:
    """The chapters in the order the book's spine lists them."""
    with zipfile.ZipFile(path) as epub:
        names = set(epub.namelist())
        if any(f in names for f in _EPUB_DRM_FILES):
            raise _locked(path.name)
        if "META-INF/encryption.xml" in names:
            encryption = ElementTree.fromstring(epub.read("META-INF/encryption.xml"))
            methods = {e.get("Algorithm") for e in encryption.iter() if _local(e.tag) == "EncryptionMethod"}
            if methods - _FONT_OBFUSCATION:
                raise _locked(path.name)
        container = ElementTree.fromstring(epub.read("META-INF/container.xml"))
        rootfile = next(e.get("full-path") for e in container.iter() if _local(e.tag) == "rootfile")
        if not rootfile:
            raise BookFileError("the EPUB's container names no package file")
        package = ElementTree.fromstring(epub.read(rootfile))
        base = posixpath.dirname(rootfile)
        manifest = {e.get("id"): e.get("href") for e in package.iter() if _local(e.tag) == "item"}
        spine = [e.get("idref") for e in package.iter() if _local(e.tag) == "itemref"]
        chapters = []
        for idref in spine:
            href = manifest.get(idref)
            if href:
                name = posixpath.normpath(posixpath.join(base, href.split("#", 1)[0]))
                chapters.append(_html_text(epub.read(name).decode("utf-8", errors="replace")))
    return "\n".join(chapters)


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # imported here: only PDFs need it
    from pypdf.errors import FileNotDecryptedError

    try:
        reader = PdfReader(path)
        # pypdf opens a PDF encrypted only against editing or printing (an
        # empty user password) itself; one that needs a key raises.
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except FileNotDecryptedError as exc:
        raise _locked(path.name) from exc
    except NotImplementedError as exc:
        # A DRM scheme's own encryption handler (Adobe's EBX_HANDLER), which
        # pypdf's encryption module refuses.
        if traceback.extract_tb(exc.__traceback__)[-1].filename.endswith("_encryption.py"):
            raise _locked(path.name) from exc
        raise BookFileError(f"couldn't read {path.name}: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # a malformed PDF raises all kinds, not only PyPdfError
        raise BookFileError(f"couldn't read {path.name}: {type(exc).__name__}: {exc}") from exc


def read_book_file(path: Path) -> str:
    """The book's text, its whitespace tidied. Raises BookFileError for a
    file that's missing, of an unknown type, unreadable, or too short to be
    the book.
    """
    if not path.is_file():
        raise BookFileError(f"no such file: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".epub":
            text = _epub_text(path)
        elif suffix == ".pdf":
            text = _pdf_text(path)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise BookFileError(f"{path.name}: book files are .epub, .pdf or .txt")
    except BookFileError:
        raise
    except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise BookFileError(f"couldn't read {path.name}: {type(exc).__name__}: {exc}") from exc
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    text = "\n".join(line for line in lines if line)
    if len(text) < MIN_BOOK_CHARS:
        raise BookFileError(
            f"{path.name} holds only {len(text):,} characters of text — a scanned PDF needs OCR first, "
            "and a book is longer than that"
        )
    return text
