import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from precis import cli as cli_module
from precis import usage
from precis.eval import data, judge, metrics, runner
from precis.known_file import preflight_check

REPO_EVALS = Path(__file__).resolve().parent.parent / "evals"


def _book(points_by_chapter: list[list], **extra) -> dict:
    return {
        "title": "T",
        "one_line_takeaway": "take",
        "synopsis": "syn",
        "chapters": [
            {"number": i + 1, "title": f"C{i + 1}", "core_claim": "claim", "key_points": points}
            for i, points in enumerate(points_by_chapter)
        ],
        **extra,
    }


def _write_eval_set(root: Path, slugs=("alpha", "beta")) -> None:
    for slug in slugs:
        data.write_json(
            root / "books" / f"{slug}.json",
            {"isbn": "1", "title": slug.title(), "author": "Ann Author", "kind": "non-fiction", "chapters": ["One"]},
        )
        data.write_json(
            root / "references" / f"{slug}.json",
            _book([["the reference point"]], title=slug.title(), author="Ann Author"),
        )


# --- metrics -----------------------------------------------------------------


def test_duplicate_point_rate_counts_both_points_of_a_near_duplicate_pair():
    book = _book(
        [
            ["Loss aversion makes losses loom larger than equivalent gains", "Anchors bias numeric estimates"],
            ["Losses loom larger than equivalent gains because of loss aversion", "Framing changes choices"],
        ]
    )
    assert metrics.duplicate_point_rate(book) == 0.5


def test_duplicate_point_rate_is_none_without_chapters():
    assert metrics.duplicate_point_rate({"synopsis": "x"}) is None


def test_citation_coverage_needs_v2_points():
    assert metrics.citation_coverage(_book([["a string point"]])) is None
    v2 = _book([[{"point": "a", "evidence": "e", "sources": ["S1"]}, {"point": "b", "evidence": "e", "sources": []}]])
    assert metrics.citation_coverage(v2) == 0.5


def test_book_metrics_counts_flags_and_warnings():
    book = _book([["a"], ["b"]], warnings=["w"])
    book["chapters"][0]["quality_flag"] = "weak"
    m = metrics.book_metrics(book)
    assert (m["chapters"], m["key_points"], m["flagged_chapters"], m["warnings"]) == (2, 2, 1, 1)


def test_summarize_counts_failed_books_toward_cost_only():
    ok = {"usage": {"total_cost_usd": 0.2, "llm_calls": 4, "searches": 3}, "duration_seconds": 10,
          "flagged_chapters": 1, "warnings": 0, "duplicate_point_rate": 0.1, "citation_coverage": None}
    failed = {"error": "boom", "usage": {"total_cost_usd": 0.1, "llm_calls": 1, "searches": 1}, "duration_seconds": 2}
    s = metrics.summarize([ok, failed])
    assert s["failed"] == 1
    assert s["total_cost_usd"] == pytest.approx(0.3)
    assert s["mean_duration_seconds"] == 10
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


def test_eval_set_has_the_planned_mix_and_every_book_is_ready():
    books = data.load_books(REPO_EVALS)
    kinds = [
        "fiction" if b.known_file.kind == "fiction" else "narrative" if b.known_file.narrative else "full"
        for b in books
    ]
    assert (kinds.count("full"), kinds.count("narrative"), kinds.count("fiction")) == (4, 2, 2)
    for book in books:
        assert preflight_check(book.known_file) == [], book.slug


def test_references_match_their_known_files():
    for book in data.load_books(REPO_EVALS):
        kf, ref = book.known_file, book.reference
        assert (ref.title, ref.author) == (kf.title, kf.author), book.slug
        assert [c.title for c in ref.chapters] == kf.chapters, book.slug
        assert [c.number for c in ref.chapters] == list(range(1, len(kf.chapters) + 1)), book.slug
        if kf.parts:
            assert [p.title for p in ref.parts] == [p.title for p in kf.parts], book.slug
            assert [p.chapters for p in ref.parts] == [p.chapters for p in kf.parts], book.slug
        if not kf.is_full_nonfiction_path:
            assert ref.parts, f"{book.slug}: a book without chapters needs parts in its reference"


# --- runner --------------------------------------------------------------------


async def test_run_eval_stores_output_and_metrics_and_skips_done_books(tmp_path):
    _write_eval_set(tmp_path)
    books = data.load_books(tmp_path)
    calls = []

    async def fake_pipeline(known_file, thread, trust_known, on_progress):
        calls.append(thread)
        usage.record_search(deep=False)
        return _book([["a point"]])

    with patch.dict(runner.PIPELINES, {"fake": fake_pipeline}):
        results = await runner.run_eval(tmp_path, "r1", books, pipeline="fake", trust_known=False, on_progress=print)
        assert calls == ["eval-r1-alpha", "eval-r1-beta"]
        assert results[0]["usage"]["searches"] == 1
        assert results[0]["key_points"] == 1
        assert json.loads(runner.book_path(tmp_path, "r1", "alpha").read_text())["chapters"]

        again = await runner.run_eval(tmp_path, "r1", books, pipeline="fake", trust_known=False, on_progress=print)
    assert len(calls) == 2
    assert [r["slug"] for r in again] == ["alpha", "beta"]


async def test_a_failing_book_is_recorded_and_the_run_continues(tmp_path):
    _write_eval_set(tmp_path)

    async def flaky(known_file, thread, trust_known, on_progress):
        if known_file.title == "Alpha":
            raise RuntimeError("verify failed")
        return _book([["a"]])

    with patch.dict(runner.PIPELINES, {"flaky": flaky}):
        results = await runner.run_eval(
            tmp_path, "r", data.load_books(tmp_path), pipeline="flaky", trust_known=False, on_progress=print
        )
    assert results[0]["error"] == "RuntimeError: verify failed"
    assert not runner.book_path(tmp_path, "r", "alpha").exists()
    assert runner.metrics_path(tmp_path, "r", "alpha").exists()
    assert results[1]["error"] is None


# --- judge ---------------------------------------------------------------------


def _verdict_reply(pick: str, overall: str | None = None) -> str:
    verdict = {name: pick for name in judge.CRITERIA} | {"overall": overall or pick, "reason": "because"}
    return f"Some reasoning.\n```json\n{json.dumps(verdict)}\n```"


def test_parse_verdict_takes_the_last_json_block():
    reply = "```json\n{\"not\": \"it\"}\n```\n" + _verdict_reply("B")
    assert judge.parse_verdict(reply).overall == "B"
    with pytest.raises(ValueError):
        judge.parse_verdict("no block here")


@pytest.mark.parametrize(
    ("first", "swapped", "expected"),
    [("A", "B", "candidate"), ("B", "A", "baseline"), ("A", "A", "tie"), ("tie", "tie", "tie"), ("A", "tie", "tie")],
)
def test_a_pick_counts_only_when_both_orders_agree(first, swapped, expected):
    assert judge._combine(first, swapped) == expected


def test_render_summary_handles_v1_and_v2_points():
    text = judge.render_summary(
        _book([["plain point", {"point": "cited point", "evidence": "the marshmallow study", "sources": ["S1"]}]])
    )
    assert "- plain point" in text
    assert "- cited point\n  Evidence: the marshmallow study" in text


async def test_judge_book_judges_both_orders():
    book = data.EvalBook(
        slug="s",
        known_file=data.KnownFile(isbn="1", title="T", author="A", kind="non-fiction", chapters=["C1"]),
        reference=data.Reference.model_validate(_book([["ref"]], title="T", author="A")),
    )
    candidate, baseline = _book([["cand point"]]), _book([["base point"]])
    replies = {True: _verdict_reply("A"), False: _verdict_reply("B")}

    async def fake_complete(client, *, messages, model):
        prompt = messages[1]["content"]
        candidate_first = prompt.index("cand point") < prompt.index("base point")
        return replies[candidate_first]

    with patch.object(judge.llm, "complete", side_effect=fake_complete):
        result = await judge.judge_book(object(), "judge-model", book, candidate, baseline)
    assert result["winner"] == "candidate"
    assert set(result["criteria"].values()) == {"candidate"}


async def test_a_side_without_output_loses_without_a_judge_call():
    with patch.object(judge.llm, "complete", new=AsyncMock()) as complete:
        result = await judge.judge_book(object(), "m", None, None, _book([["x"]]))  # type: ignore[arg-type]
    assert result["winner"] == "baseline"
    complete.assert_not_called()


def test_score_counts_ties_as_half():
    results = {
        "a": {"winner": "candidate", "criteria": {name: "candidate" for name in judge.CRITERIA}},
        "b": {"winner": "tie", "criteria": {name: "baseline" for name in judge.CRITERIA}},
        "c": {"winner": "baseline", "reason": "candidate has no output for this book"},
    }
    s = judge.score(results)
    assert (s["candidate_wins"], s["baseline_wins"], s["ties"]) == (1, 1, 1)
    assert s["score"] == 0.5
    assert s["criteria"]["accuracy"] == 0.5


async def test_judge_runs_writes_the_judgement(tmp_path):
    _write_eval_set(tmp_path)
    for label in ("new", "old"):
        data.write_json(runner.book_path(tmp_path, label, "alpha"), _book([[f"{label} point"]]))
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
