"""Code checks on written notes, before the review call. No model: each
finding is a specific issue handed to the review as something to fix or
defend, so the review spends its attention where problems are likely.

- **Meta-descriptions**: an idea, its evidence or an answer describing the
  text ("the book examines…", "Storr traces…") instead of stating the idea
  or giving the book's example. The Integrity of the Personality's notes,
  written from thin research, were full of them; a general statement of
  what the book argues is a fine fallback, a description of the text isn't.
- **Near-duplicate ideas**: two ideas sharing most of their vocabulary.
- **Quotes not in the research**: a quoted passage that no source's page
  holds, or — when the research holds the whole book's own text — one found
  only in pages about the book. The Undiscovered Self's notes quoted Jung from
  another essay, which a summary site had attributed to this book. Quotes
  are the one fact checked against the research: a paraphrase of an
  unconfirmed quote loses nothing, while a wrong quote misleads.
- **Long quotes**: a quote over a couple of sentences, or more quoted text
  in all than notes need. The notes are published, so they state a book's
  ideas in their own words and quote only a line where the exact words
  matter — never enough of the book to stand in for it. Checked with or
  without research.

Nothing else checks facts against the research: "not in the research's
excerpts" isn't "invented", and flagging it led the review to strip correct
details. Accuracy is the review's job, and the evals' judge measures it.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import combinations

from precis.research import Research
from precis.schema import Book
from precis.search import author_surnames, normalize_text, overlap

# Two ideas sharing this share of their content words say the same thing.
DUPLICATE_OVERLAP = 0.5

# Shorter quoted spans are terms and phrases ("the real world"), not quotes.
QUOTE_MIN_WORDS = 4
# A quote is found when this share of its letters, in pieces this long,
# appears in a page: OCR'd books break words ("destroy- ing") and garble
# letters, and a model changes a word ("himself" for "itself").
QUOTE_PIECE = 20
QUOTE_FOUND = 0.7
# Longer than a sentence or two, a quote is a passage of the book; more than
# this in all is the book's text standing in for the notes.
QUOTE_MAX_CHARS = 300
QUOTED_MAX_CHARS = 2_000
# Double and single, curly and straight: Haiku writes quotes in single ones
# inside its JSON. A quote only opens where a word can't end (not the inch
# mark of `5" scroll`) and only closes where one can't start, so an
# apostrophe inside a word ("man's", "man’s") neither opens nor closes one;
# straight quotes don't span lines.
_QUOTE = re.compile(
    r'(?<!\w)“(.+?)”(?!\w)|(?<!\w)"([^"\n]+)"(?!\w)|(?<!\w)‘(.+?)’(?!\w)|(?<!\w)\'([^\n]+?)\'(?!\w)'
)
# Words a title leaves lowercase ("Is the Active Self a Limited Resource?").
_TITLE_SMALL = frozenset(
    ["a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of", "on", "or", "the", "to", "vs", "with"]
)
# The quote check only treats a book-text page as the whole book — so a
# quote missing from it is suspect — when it's at least this long: shorter,
# it may be a preview of a few chapters, and a correct quote from a later
# one is only on pages about the book.
WHOLE_BOOK_CHARS = 100_000

# Verbs that describe what a text does rather than what it argues: "the
# book traces X" says nothing about X, "the book argues X" does.
_DESCRIBING = (
    r"(examines|explores|discusses|describes|covers|looks at|delves into|talks about|addresses|traces|"
    r"investigates|includes|presents|outlines)"
)
_META = re.compile(
    rf"\b(the|this) (book|author|chapter|text|novel|work) {_DESCRIBING}\b"
    r"|\bthe author (argues|explains|shows) (that )?this\b",
    re.IGNORECASE,
)
# In evidence, the author describing something is the example itself, and
# so is "the book includes X" partway through ("…Krakauer's sister. The book
# includes critical letters he received"). Only evidence that opens by
# describing the text ("The book examines ten firms.") has no example.
_META_EVIDENCE = re.compile(rf"^\s*(the|this) (book|chapter|text|novel|work) {_DESCRIBING}\b", re.IGNORECASE)


def _describes_text(text: str, surnames: list[str] = [], *, evidence: bool = False) -> bool:  # noqa: B006 — never mutated
    """Whether `text` describes the book rather than stating its idea — with
    the author's surname as a subject too when `surnames` are given
    ("Storr traces…", but not "Storr argues…"). Evidence is checked with
    neither the author nor surnames as a subject: there "the author (or
    Krakauer) describes his 1977 climb of Devils Thumb" is the example itself.
    """
    if (_META_EVIDENCE if evidence else _META).search(text):
        return True
    return any(re.search(rf"\b{re.escape(name)}\s+{_DESCRIBING}\b", text, re.IGNORECASE) for name in surnames)

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "they", "this", "to", "was", "were", "what", "when", "which", "who", "why", "will", "with"]
)


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in normalize_text(text).split() if len(w) > 2 and w not in _STOPWORDS)


def _letters(text: str) -> str:
    """`text` as normalize_text's lowercase, accent-free letters and digits
    alone, so punctuation, spacing, quote style and hyphenation don't matter.
    """
    return normalize_text(text).replace(" ", "")


# Pages only, so a book's quotes can't evict them: check_notes runs before
# and after the review, and each page is folded once.
_page_letters = lru_cache(maxsize=32)(_letters)


def _is_title(span: str) -> bool:
    """Whether a quoted span is a title — of a paper, essay or chapter —
    rather than a quote: every word but the small ones capitalized.
    """
    words = [w for w in re.findall(r"[^\W\d_][\w'’-]*", span) if w.lower() not in _TITLE_SMALL]
    return bool(words) and all(w[0].isupper() for w in words)


def quotes(text: str) -> list[str]:
    """The quoted passages in `text` long enough to be quotes and not titles."""
    spans = [next(g for g in groups if g).strip() for groups in _QUOTE.findall(text)]
    return [q for q in spans if len(q.split()) >= QUOTE_MIN_WORDS and not _is_title(q)]


def _found(quote: str, page: str) -> bool:
    """Whether `quote` (as _letters) is in `page` (as _letters), allowing for
    OCR damage and a changed word.
    """
    if quote in page:
        return True
    pieces = [quote[i : i + QUOTE_PIECE] for i in range(0, len(quote) - QUOTE_PIECE + 1, QUOTE_PIECE)]
    if len(quote) % QUOTE_PIECE and len(quote) > QUOTE_PIECE:
        pieces.append(quote[-QUOTE_PIECE:])  # the end too, not just whole pieces
    return bool(pieces) and sum(p in page for p in pieces) >= QUOTE_FOUND * len(pieces)


def _short(quote: str) -> str:
    return quote if len(quote) <= 80 else quote[:77] + "…"


def _quoted(book: Book) -> list[tuple[str, str]]:
    """Each quote in the notes, with the field it's in."""
    fields = [("the takeaway", book.one_line_takeaway), ("the synopsis", book.synopsis)]
    for n, i in enumerate(book.ideas, 1):  # apart, so a quote mark in one can't pair with one in the other
        fields += [(f'idea {n} "{i.title}"', i.summary), (f'idea {n} "{i.title}"', i.evidence)]
    fields += [(f"key claim {n}", c.answer) for n, c in enumerate(book.key_claims_for_review or [], 1)]
    return [(where, quote) for where, text in fields for quote in quotes(text)]


def _length_issues(book: Book) -> list[str]:
    found = _quoted(book)
    issues = [
        f'{where}: the quote "{_short(quote)}" runs {len(quote):,} characters — quote a sentence or two at most '
        "and paraphrase the rest"
        for where, quote in found
        if len(quote) > QUOTE_MAX_CHARS
    ]
    total = sum(len(quote) for _, quote in found)
    if total > QUOTED_MAX_CHARS:
        issues.append(
            f"the notes quote {total:,} characters of the book in all — keep the quotes whose exact words matter "
            "and paraphrase the rest"
        )
    return issues


def _quote_issues(book: Book, research: Research) -> list[str]:
    pages = {s.id: _page_letters(s.page_text) for s in research.sources}
    book_text = [s.id for s in research.sources if s.is_book_text and len(s.page_text) >= WHOLE_BOOK_CHARS]
    issues = []
    for where, quote in _quoted(book):
        letters = _letters(quote)
        found = [sid for sid, page in pages.items() if _found(letters, page)]
        short = _short(quote)
        if not found:
            issues.append(
                f'{where}: the quote "{short}" isn\'t in any source — check it, or give it as a paraphrase '
                "without quote marks"
            )
        elif book_text and not set(found) & set(book_text):
            issues.append(
                f'{where}: the quote "{short}" is in {", ".join(found)} but not in the book\'s own text '
                f"({', '.join(book_text)}) — it may be from another of the author's works; check it"
            )
    return issues


def check_notes(book: Book, research: Research | None = None) -> list[str]:
    """One line per finding, naming the idea (1-based) or field it's about.
    Checking quotes against the research needs the research the notes were
    written from; their length is checked either way.
    """
    issues: list[str] = []
    ideas = book.ideas

    surnames = author_surnames(book.author)
    for n, idea in enumerate(ideas, 1):
        if _describes_text(f"{idea.title}. {idea.summary}", surnames):
            issues.append(f'idea {n} "{idea.title}": describes the text instead of stating the idea')
        if _describes_text(idea.evidence, evidence=True):
            issues.append(
                f'idea {n} "{idea.title}": its evidence describes the text instead of giving the book\'s example'
            )
    for n, claim in enumerate(book.key_claims_for_review or [], 1):
        if _describes_text(claim.answer, surnames):
            issues.append(f"key claim {n}: the answer describes the text instead of stating the idea")

    words = [content_words(f"{i.title} {i.summary}") for i in ideas]
    for a, b in combinations(range(len(ideas)), 2):
        if overlap(words[a], words[b]) >= DUPLICATE_OVERLAP:
            issues.append(f'ideas {a + 1} and {b + 1} ("{ideas[a].title}", "{ideas[b].title}") say much the same thing')
    issues += _length_issues(book)
    if research is not None:
        issues += _quote_issues(book, research)
    return issues
