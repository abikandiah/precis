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
from precis.search import contains, normalize_text, overlap

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

# Capitalized words that are never names: common sentence openers and
# titles. Other sentence openers are judged against the research (see
# _names).
_NOT_A_NAME = _STOPWORDS | frozenset(
    ["ceo", "after", "all", "also", "although", "among", "another", "any", "because", "before", "between", "both", "during", "each", "early", "even", "every", "first", "he", "her", "here", "his", "i", "if", "later", "many", "most", "much", "no", "not", "one", "only", "other", "our", "over", "second", "she", "since", "so", "some", "such", "then", "there", "these", "third", "those", "though", "three", "through", "two", "under", "until", "we", "when", "where", "while", "you", "dr", "mr", "mrs", "ms", "prof", "professor", "sir", "dame", "lord", "lady"]
)

# A number, with any unit or suffix attached ("1970s", "20th", "10x",
# "$2.5m", "42km", "3GHz", "42%") — the whole token is what has to match.
_NUMBER = re.compile(r"(?<![\w.])\d(?:[\d,.]*\d)?(?:[^\W\d_]{1,3}|%)?(?!\w)")
# A word, accents and hyphens included, split on hyphens afterwards.
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
# A word starting a sentence: the text's first, or one after . ! or ?
# (and any opening quote or bracket).
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)[\"'“‘(\[]*$")
# Lowercase words, for telling "However" or "Participants" from a name.
_LOWERCASE_WORD = re.compile(r"(?<![\w'’-])[a-z][a-z'’-]*")

# A number's unit, split off so "42km" matches "42 km", with magnitudes
# spelled one way so "$2.5m" matches "$2.5 million" (and "42%" "42 percent").
_UNIT = re.compile(r"(?<=\d)(?=[^\W\d_])")
_MAGNITUDE = re.compile(r"(?<=\d) (million|mn|billion|thousand|per ?cent)\b", re.IGNORECASE)
_MAGNITUDES = {"million": "m", "mn": "m", "billion": "bn", "thousand": "k"}


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in normalize_text(text).split() if len(w) > 2 and w not in _STOPWORDS)


def lowercase_words(text: str) -> frozenset[str]:
    """Every word `text` uses in lowercase — ordinary words, not names."""
    return frozenset(_LOWERCASE_WORD.findall(text))


def _names(text: str, common: frozenset[str]) -> list[str]:
    """Capitalized words, one at a time — "Zimbardo's Stanford Prison
    Experiment" and a source's "Philip Zimbardo ran the Stanford prison
    experiment" should agree — with possessives stripped and a hyphenated
    compound split, so "Harvard-trained" is checked as "Harvard".

    A word capitalized only because it starts a sentence ("However",
    "Participants") is a name only when it isn't in `common`, the words
    seen in lowercase. "Henderson's 1994 study" opening the evidence is
    exactly where an invented researcher would sit, so it still counts.
    """
    names = []
    for match in _WORD.finditer(text):
        opens_sentence = bool(_SENTENCE_START.search(text[: match.start()]))
        for i, part in enumerate(re.split(r"[-]", match.group())):
            part = re.sub(r"['’]s?$", "", part)
            if len(part) < 2 or not part[0].isupper() or part.lower() in _NOT_A_NAME:
                continue
            if opens_sentence and i == 0 and part.lower() in common:
                continue
            names.append(part)
    return names


def specifics(text: str, common: frozenset[str] = frozenset()) -> list[str]:
    """The numbers and proper nouns in `text`, without repeats. `common`
    is the words seen in lowercase (see _names).
    """
    common = common | lowercase_words(text)
    return list(dict.fromkeys([m.group() for m in _NUMBER.finditer(text)] + _names(text, common)))


def _fold(text: str) -> str:
    """Matching form: normalize_text, with thousands separators dropped
    first so "1,435" and "1435" agree (and "Kahneman,Tversky" stays two
    words), and units and magnitudes spelled one way.
    """
    text = _UNIT.sub(" ", re.sub(r"(?<=\d),(?=\d)", "", text))
    text = _MAGNITUDE.sub(lambda m: " " + _MAGNITUDES.get(m.group(1).lower(), ""), text)
    return normalize_text(text)


def _in(haystack: str, needle: str) -> bool:
    return contains(haystack, _fold(needle))


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
        common = lowercase_words(" ".join(s.text for s in research.sources))
        everything = " ".join(texts.values())
        # The book's own title and author are named everywhere; not specifics.
        known = _fold(f"{book.title} {book.author}")
        for n, idea in enumerate(ideas, 1):
            cited = " ".join(texts.get(s, "") for s in idea.sources) if idea.sources else everything
            missing = [
                s for s in specifics(idea.evidence, common) if not _in(cited, s) and not _in(known, s)
            ]
            if missing:
                where = f"its cited sources ({', '.join(idea.sources)})" if idea.sources else "the research"
                issues.append(f'idea {n} "{idea.title}": its evidence names {missing!r}, which aren\'t in {where}')
    return issues
