import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from precis import cli as cli_module
from precis import usage
from precis.eval import data, judge, metrics, runner
from precis.generate import Generated
from precis.research import Research, Source
from precis.schema import Book

REPO_EVALS = Path(__file__).resolve().parent.parent / "evals"


def _idea(title: str, summary: str = "summary", sources: list[str] | None = None) -> dict:
    return {"title": title, "summary": summary, "evidence": "evidence", "sources": sources or []}


def _book(ideas: list[dict], **extra) -> dict:
    return {"title": "T", "one_line_takeaway": "take", "synopsis": "syn", "ideas": ideas, **extra}


def _reference(title: str, author: str) -> dict:
    return _book([_idea(f"ref idea {n}") for n in range(3)], title=title, author=author)


def _write_eval_set(root: Path, slugs=("alpha", "beta")) -> None:
    for slug in slugs:
        data.write_json(
            root / "books" / f"{slug}.json",
            {"isbn": "1", "title": slug.title(), "author": "Ann Author", "kind": "non-fiction"},
        )
        data.write_json(root / "references" / f"{slug}.json", _reference(slug.title(), "Ann Author"))


# --- metrics -----------------------------------------------------------------


def test_duplicate_idea_rate_counts_both_ideas_of_a_near_duplicate_pair():
    book = _book(
        [
            _idea("Loss aversion", "Losses loom larger than equivalent gains"),
            _idea("Anchoring", "Arbitrary numbers bias estimates"),
            _idea("Loss aversion", "Equivalent gains loom smaller than losses"),
            _idea("Framing", "Descriptions change choices"),
        ]
    )
    assert metrics.duplicate_idea_rate(book) == 0.5


def test_idea_metrics_are_none_without_ideas():
    assert metrics.duplicate_idea_rate({"synopsis": "x"}) is None
    assert metrics.citation_coverage({"synopsis": "x"}) is None


def test_citation_coverage_counts_ideas_with_sources():
    assert metrics.citation_coverage(_book([_idea("a", sources=["S1"]), _idea("b")])) == 0.5


def test_book_metrics_counts_ideas_claims_and_warnings():
    book = _book([_idea("a"), _idea("b")], key_claims_for_review=[{"prompt": "q", "answer": "a"}], warnings=["w"])
    m = metrics.book_metrics(book)
    assert (m["ideas"], m["key_claims"], m["warnings"]) == (2, 1, 1)


def test_summarize_counts_failed_books_toward_cost_only():
    ok = {"usage": {"total_cost_usd": 0.2, "llm_calls": 4, "searches": 3}, "duration_seconds": 10,
          "ideas": 8, "warnings": 0, "duplicate_idea_rate": 0.1, "citation_coverage": None}
    failed = {"error": "boom", "usage": {"total_cost_usd": 0.1, "llm_calls": 1, "searches": 1}, "duration_seconds": 2}
    s = metrics.summarize([ok, failed])
    assert s["failed"] == 1
    assert s["total_cost_usd"] == pytest.approx(0.3)
    assert s["mean_duration_seconds"] == 10
    assert s["mean_ideas"] == 8
    assert s["mean_citation_coverage"] is None
    assert "1 failed" in metrics.format_summary("run", s)


# --- data ----------------------------------------------------------------------


def test_load_books_filters_by_slug_and_rejects_unknown_ones(tmp_path):
    _write_eval_set(tmp_path)
    assert [b.slug for b in data.load_books(tmp_path)] == ["alpha", "beta"]
    assert [b.slug for b in data.load_books(tmp_path, ["beta"])] == ["beta"]
    with pytest.raises(ValueError, match="not in the eval set: gamma"):
        data.load_books(tmp_path, ["gamma"])


def test_load_books_requires_a_reference(tmp_path):
    _write_eval_set(tmp_path)
    (tmp_path / "references" / "beta.json").unlink()
    with pytest.raises(ValueError, match="no reference summary"):
        data.load_books(tmp_path)


# --- the committed eval set ------------------------------------------------------


def test_eval_set_has_the_planned_mix_of_bibliographic_known_files():
    books = data.load_books(REPO_EVALS)
    kinds = [b.known_file.kind for b in books]
    assert (kinds.count("non-fiction"), kinds.count("fiction")) == (6, 2)
    for book in books:
        assert book.known_file.has_title and book.known_file.has_author, book.slug


def test_references_match_their_known_files_and_the_v2_shape():
    for book in data.load_books(REPO_EVALS):
        kf, ref = book.known_file, book.reference
        assert (ref.title, ref.author) == (kf.title, kf.author), book.slug
        if kf.kind == "non-fiction":
            assert 5 <= len(ref.ideas) <= 12, book.slug
            assert 5 <= len(ref.key_claims_for_review) <= 15, book.slug
        else:
            assert 3 <= len(ref.ideas) <= 6, book.slug
            assert not ref.key_claims_for_review, f"{book.slug}: fiction has no review deck"


# --- runner --------------------------------------------------------------------


def _generated(fingerprint_text: str = "research") -> Generated:
    found = Research(sources=[Source(id="S1", title="t", url="https://a.org", text=fingerprint_text)], warnings=[])
    return Generated(book=_generated_book(), research=found)


def _generated_book() -> Book:
    return Book(
        title="T", author="A", isbn="1", kind="non-fiction", one_line_takeaway="take", synopsis="syn",
        tags=["psychology", "science"], ideas=[_idea(f"idea {n}") for n in range(5)],
        key_claims_for_review=[{"prompt": f"Q{n}?", "answer": "A."} for n in range(5)],
    )  # fmt: skip


async def test_run_eval_stores_output_and_metrics_and_skips_done_books(tmp_path, monkeypatch):
    _write_eval_set(tmp_path)
    books = data.load_books(tmp_path)
    slugs = []

    async def fake_generate(known_file, *, slug, trust_known, fresh, on_progress):
        slugs.append(slug)
        usage.record_search()
        return _generated()

    monkeypatch.setattr(runner, "generate", fake_generate)
    results = await runner.run_eval(tmp_path, "r1", books, trust_known=False, on_progress=print)
    # Research is cached per book, not per run label.
    assert slugs == ["alpha", "beta"]
    assert results[0]["usage"]["searches"] == 1
    assert (results[0]["ideas"], results[0]["key_claims"]) == (5, 5)
    assert json.loads(data.book_path(tmp_path, "r1", "alpha").read_text())["schema_version"] == "2"
    assert results[0]["research"] == {"fingerprint": _generated().research.fingerprint, "sources": 1}

    again = await runner.run_eval(tmp_path, "r1", books, trust_known=False, on_progress=print)
    assert len(slugs) == 2
    assert [r["slug"] for r in again] == ["alpha", "beta"]


async def test_a_failing_book_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    _write_eval_set(tmp_path)

    async def flaky(known_file, *, slug, trust_known, fresh, on_progress):
        if known_file.title == "Alpha":
            raise RuntimeError("research failed")
        return _generated()

    monkeypatch.setattr(runner, "generate", flaky)
    results = await runner.run_eval(tmp_path, "r", data.load_books(tmp_path), trust_known=False, on_progress=print)
    assert results[0]["error"] == "RuntimeError: research failed"
    assert not data.book_path(tmp_path, "r", "alpha").exists()
    assert data.metrics_path(tmp_path, "r", "alpha").exists()
    assert results[1]["error"] is None


# --- judge ---------------------------------------------------------------------


def _verdict_reply(pick: str, overall: str | None = None, kind: judge.Kind = "non-fiction") -> str:
    verdict = {
        "criteria": {name: pick for name in judge.criteria_for(kind)},
        "overall": overall or pick,
        "reason": "because",
    }
    return f"Some reasoning.\n```json\n{json.dumps(verdict)}\n```"


def _eval_book(kind: judge.Kind = "non-fiction") -> data.EvalBook:
    return data.EvalBook(
        slug="s",
        known_file=data.KnownFile(isbn="1", title="T", author="A", kind=kind),
        reference=data.Reference.model_validate(_reference("T", "A")),
    )


def test_parse_verdict_takes_the_last_json_block():
    criteria = judge.criteria_for("non-fiction")
    reply = "```json\n{\"not\": \"it\"}\n```\n" + _verdict_reply("B")
    assert judge.parse_verdict(reply, criteria).overall == "B"
    with pytest.raises(ValueError):
        judge.parse_verdict("no block here", criteria)


def test_parse_verdict_requires_the_kinds_criteria():
    with pytest.raises(ValueError, match="expected"):
        judge.parse_verdict(_verdict_reply("A", kind="non-fiction"), judge.criteria_for("fiction"))


def test_criteria_differ_by_kind():
    assert "review_deck" in judge.criteria_for("non-fiction")
    assert "spoiler_safety" not in judge.criteria_for("non-fiction")
    assert "spoiler_safety" in judge.criteria_for("fiction")
    assert "review_deck" not in judge.criteria_for("fiction")


@pytest.mark.parametrize(
    ("first", "swapped", "expected"),
    [("A", "B", "candidate"), ("B", "A", "baseline"), ("A", "A", "tie"), ("tie", "tie", "tie"), ("A", "tie", "tie")],
)
def test_a_pick_counts_only_when_both_orders_agree(first, swapped, expected):
    assert judge._combine(first, swapped) == expected


def test_render_summary_shows_ideas_evidence_and_claims_but_not_sources():
    text = judge.render_summary(
        _book(
            [_idea("Anchoring", "numbers pull estimates", sources=["S1"])],
            key_claims_for_review=[{"prompt": "What is anchoring?", "answer": "A bias."}],
        )
    )
    assert "- Anchoring: numbers pull estimates\n  Evidence: evidence" in text
    assert "- Q: What is anchoring?\n  A: A bias." in text
    assert "S1" not in text


async def test_judge_book_judges_both_orders():
    candidate, baseline = _book([_idea("cand idea")]), _book([_idea("base idea")])
    replies = {True: _verdict_reply("A", kind="fiction"), False: _verdict_reply("B", kind="fiction")}

    async def fake_complete(client, *, messages, model):
        prompt = messages[1]["content"]
        assert "spoiler_safety" in prompt
        candidate_first = prompt.index("cand idea") < prompt.index("base idea")
        return replies[candidate_first]

    with patch.object(judge.llm, "complete", side_effect=fake_complete):
        result = await judge.judge_book(object(), "judge-model", _eval_book("fiction"), candidate, baseline)
    assert result["winner"] == "candidate"
    assert set(result["criteria"]) == set(judge.criteria_for("fiction"))
    assert set(result["criteria"].values()) == {"candidate"}


async def test_a_side_without_output_loses_without_a_judge_call():
    with patch.object(judge.llm, "complete", new=AsyncMock()) as complete:
        result = await judge.judge_book(object(), "m", None, None, _book([_idea("x")]))  # type: ignore[arg-type]
    assert result["winner"] == "baseline"
    complete.assert_not_called()


def test_score_counts_ties_as_half_and_kind_criteria_over_their_kind_only():
    results = {
        "a": {"winner": "candidate", "criteria": {name: "candidate" for name in judge.criteria_for("fiction")}},
        "b": {"winner": "tie", "criteria": {name: "baseline" for name in judge.criteria_for("non-fiction")}},
        "c": {"winner": "baseline", "reason": "candidate has no output for this book"},
    }
    s = judge.score(results)
    assert (s["candidate_wins"], s["baseline_wins"], s["ties"]) == (1, 1, 1)
    assert s["score"] == 0.5
    assert s["criteria"]["accuracy"] == 0.5
    assert s["criteria"]["spoiler_safety"] == 1.0
    assert s["criteria"]["review_deck"] == 0.0


async def test_judge_runs_writes_the_judgement(tmp_path):
    _write_eval_set(tmp_path)
    for label in ("new", "old"):
        data.write_json(data.book_path(tmp_path, label, "alpha"), _book([_idea(f"{label} idea")]))
    (data.run_dir(tmp_path, "old")).mkdir(parents=True, exist_ok=True)

    with (
        patch.object(judge.llm, "build_client"),
        patch.object(judge.llm, "complete", new=AsyncMock(return_value=_verdict_reply("tie"))),
    ):
        judgement = await judge.judge_runs(
            tmp_path, "new", "old", data.load_books(tmp_path), model="m", concurrency=2, on_progress=print
        )
    # beta has output in neither run, so it isn't judged.
    assert list(judgement["books"]) == ["alpha"]
    assert judgement["ties"] == 1
    assert judge.judgement_path(tmp_path, "new", "old").exists()
    assert "score 0.50" in judge.format_judgement(judgement)


def test_research_differs_flags_books_whose_runs_saw_different_or_unrecorded_research(tmp_path):
    for label, fingerprints in (("new", {"a": "x", "b": "x", "c": "x"}), ("old", {"a": "x", "b": "y"})):
        for slug, fingerprint in fingerprints.items():
            data.write_json(data.metrics_path(tmp_path, label, slug), {"research": {"fingerprint": fingerprint}})
    assert judge.research_differs(tmp_path, "new", "old", ["a", "b", "c"]) == ["b", "c"]
    judgement = {
        "candidate": "new", "baseline": "old", "judge_model": "m", "candidate_wins": 1, "baseline_wins": 0,
        "ties": 0, "score": 1.0, "criteria": {}, "research_differs": ["b", "c"],
        "judge_usage": {"llm_cost_usd": 0.0, "llm_calls": 0},
    }  # fmt: skip
    assert "different research for b, c" in judge.format_judgement(judgement)


async def test_judge_runs_rejects_an_unknown_run(tmp_path):
    _write_eval_set(tmp_path)
    with pytest.raises(ValueError, match="no run named"):
        await judge.judge_runs(tmp_path, "new", "old", [], model="m", concurrency=1, on_progress=print)


# --- cli -------------------------------------------------------------------------


def test_eval_judge_needs_a_judge_model(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli_module, "settings", cli_module.settings.__class__(judge_model=""))
    parsed = cli_module.build_parser().parse_args(["eval", "judge", "a", "b", "--evals-dir", str(tmp_path)])
    assert parsed.func(parsed) == 1
    assert "PRECIS_JUDGE_MODEL" in capsys.readouterr().err
