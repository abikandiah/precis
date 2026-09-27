"""Code checks on written notes, before the review call. No model: each
finding is a specific issue handed to the review as something to fix or
defend, so the review spends its attention where problems are likely.

- **Meta-descriptions**: an idea or answer describing the text ("the author
  discusses…") instead of stating the idea.
- **Near-duplicate ideas**: two ideas sharing most of their vocabulary.
- **Evidence specifics**: numbers and proper nouns in an idea's evidence —
  study names, figures, people, dates, where invention hides — that don't
  appear in the sources it cites (or, for an uncited idea, anywhere in the
  research). Not proof of invention, since the research can't hold
  everything: a reason to check.
"""

from __future__ import annotations

import re
from itertools import combinations

from precis.research import Research
from precis.schema import Book
from precis.search import normalize_text

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

# Capitalized words that aren't names: sentence openers and titles. A
# name's position doesn't matter — "Henderson's 1994 study" opening the
# evidence is exactly where an invented researcher would sit.
_NOT_A_NAME = _STOPWORDS | frozenset(
    ["after", "all", "also", "although", "among", "another", "any", "because", "before", "between", "both", "during", "each", "early", "even", "every", "first", "he", "her", "here", "his", "i", "if", "later", "many", "most", "much", "no", "not", "one", "only", "other", "our", "over", "second", "she", "since", "so", "some", "such", "then", "there", "these", "third", "those", "though", "three", "through", "two", "under", "until", "we", "when", "where", "while", "you", "dr", "mr", "mrs", "ms", "prof", "professor", "sir", "dame", "lord", "lady"]
)

# A number, with any unit or suffix attached ("1970s", "20th", "10x",
# "$2.5m", "42%") — the whole token is what has to match.
_NUMBER = re.compile(r"(?<![\w.])\d(?:[\d,.]*\d)?(?:st|nd|rd|th|s|x|k|m|bn|%)?(?!\w)")
# A word, accents and hyphens included, split on hyphens afterwards.
_WORD = re.compile(r"[^\W\d_][\w'’-]*")


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in normalize_text(text).split() if len(w) > 2 and w not in _STOPWORDS)


def overlap(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _names(text: str) -> list[str]:
    """Capitalized words, one at a time — "Zimbardo's Stanford Prison
    Experiment" and a source's "Philip Zimbardo ran the Stanford prison
    experiment" should agree — with possessives stripped and a hyphenated
    compound split, so "Harvard-trained" is checked as "Harvard".
    """
    names = []
    for match in _WORD.finditer(text):
        for part in re.split(r"[-]", match.group()):
            part = re.sub(r"['’]s?$", "", part)
            if len(part) > 1 and part[0].isupper() and part.lower() not in _NOT_A_NAME:
                names.append(part)
    return names


def specifics(text: str) -> list[str]:
    """The numbers and proper nouns in `text`, without repeats."""
    return list(dict.fromkeys([m.group() for m in _NUMBER.finditer(text)] + _names(text)))


def _fold(text: str) -> str:
    """Matching form: normalize_text, with thousands separators dropped
    first so "1,435" and "1435" agree (and "Kahneman,Tversky" stays two
    words).
    """
    return normalize_text(re.sub(r"(?<=\d),(?=\d)", "", text))


def _in(haystack: str, needle: str) -> bool:
    key = _fold(needle)
    return bool(key) and f" {key} " in f" {haystack} "


def check_notes(book: Book, research: Research) -> list[str]:
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

    if research.sources:
        texts = {s.id: _fold(s.text) for s in research.sources}
        everything = " ".join(texts.values())
        # The book's own title and author are named everywhere; not specifics.
        known = _fold(f"{book.title} {book.author}")
        for n, idea in enumerate(ideas, 1):
            cited = " ".join(texts.get(s, "") for s in idea.sources) if idea.sources else everything
            missing = [
                s for s in specifics(idea.evidence) if not _in(cited, s) and not _in(known, s)
            ]
            if missing:
                where = f"its cited sources ({', '.join(idea.sources)})" if idea.sources else "the research"
                issues.append(f'idea {n} "{idea.title}": its evidence names {missing!r}, which aren\'t in {where}')
    return issues
