"""Code checks on written notes, before the review call. No model: each
finding is a specific issue handed to the review as something to fix or
defend, so the review spends its attention where problems are likely.

- **Meta-descriptions**: an idea, its evidence or an answer describing the
  text ("the book examines…", "Storr traces…") instead of stating the idea
  or giving the book's example. The Integrity of the Personality's notes,
  written from thin research, were full of them; a general statement of
  what the book argues is a fine fallback, a description of the text isn't.
- **Near-duplicate ideas**: two ideas sharing most of their vocabulary.

Nothing here checks facts against the research: "not in the research's
excerpts" isn't "invented", and flagging it led the review to strip correct
details. Accuracy is the review's job, and the evals' judge measures it.
"""

from __future__ import annotations

import re
from itertools import combinations

from precis.schema import Book
from precis.search import author_surnames, normalize_text, overlap

# Two ideas sharing this share of their content words say the same thing.
DUPLICATE_OVERLAP = 0.5

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
# In evidence, the author describing something is the example itself.
_META_EVIDENCE = re.compile(rf"\b(the|this) (book|chapter|text|novel|work) {_DESCRIBING}\b", re.IGNORECASE)


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


def check_notes(book: Book) -> list[str]:
    """One line per finding, naming the idea (1-based) or field it's about."""
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
    return issues
