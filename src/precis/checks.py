"""Code checks on written notes, before the review call. No model: each
finding is a specific issue handed to the review as something to fix or
defend, so the review spends its attention where problems are likely.

- **Meta-descriptions**: an idea or answer describing the text ("the author
  discusses…") instead of stating the idea.
- **Near-duplicate ideas**: two ideas sharing most of their vocabulary.

Nothing here checks facts against the research: "not in the research's
excerpts" isn't "invented", and flagging it led the review to strip correct
details. Accuracy is the review's job, and the evals' judge measures it.
"""

from __future__ import annotations

import re
from itertools import combinations

from precis.schema import Book
from precis.search import normalize_text, overlap

# Two ideas sharing this share of their content words say the same thing.
DUPLICATE_OVERLAP = 0.5

_META = re.compile(
    r"\b(the|this) (book|author|chapter|text|novel|work) (examines|explores|discusses|describes|covers|"
    r"looks at|delves into|talks about|addresses)\b|\bthe author (argues|explains|shows) (that )?this\b",
    re.IGNORECASE,
)

_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "they", "this", "to", "was", "were", "what", "when", "which", "who", "why", "will", "with"]
)


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in normalize_text(text).split() if len(w) > 2 and w not in _STOPWORDS)


def check_notes(book: Book) -> list[str]:
    """One line per finding, naming the idea (1-based) or field it's about."""
    issues: list[str] = []
    ideas = book.ideas

    for n, idea in enumerate(ideas, 1):
        if _META.search(f"{idea.title} {idea.summary}"):
            issues.append(f'idea {n} "{idea.title}": describes the text instead of stating the idea')
    for n, claim in enumerate(book.key_claims_for_review or [], 1):
        if _META.search(claim.answer):
            issues.append(f"key claim {n}: the answer describes the text instead of stating the idea")

    words = [content_words(f"{i.title} {i.summary}") for i in ideas]
    for a, b in combinations(range(len(ideas)), 2):
        if overlap(words[a], words[b]) >= DUPLICATE_OVERLAP:
            issues.append(f'ideas {a + 1} and {b + 1} ("{ideas[a].title}", "{ideas[b].title}") say much the same thing')
    return issues
