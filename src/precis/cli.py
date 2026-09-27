"""CLI entrypoint. See docs/blueprint.md's CLI contract section.

Errors go to stderr with a non-zero exit code, never a traceback; progress
goes to stderr too, so stdout carries only the command's output.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from precis import research as book_research
from precis import usage
from precis.config import settings
from precis.eval import data as eval_data
from precis.eval import judge as eval_judge
from precis.eval import metrics as eval_metrics
from precis.eval import runner as eval_runner
from precis.generate import generate
from precis.known_file import create_known_file, preflight_check, slugify_title
from precis.schema import KnownFile, TagVocabulary


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr)


def _slug_from_path(path: str) -> str:
    """The known-file's filename stem — the book's identity for its
    research cache, matching the consumer's convention of naming the
    known-file and the output after the same slug.
    """
    return Path(path).stem


def _load_ready_known_file(path: str) -> KnownFile | None:
    """The known-file at `path`, or None after printing why it can't be
    used (unreadable, invalid, or not ready for generation).
    """
    try:
        known_file = KnownFile.model_validate_json(Path(path).read_text())
    except (OSError, ValueError) as exc:
        print(f"could not load known-file {path!r}: {exc}", file=sys.stderr)
        return None
    if problems := preflight_check(known_file):
        print("known-file is not ready for generation:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return None
    return known_file


def _write_output(model: BaseModel, output_path: str | None, *, exclude_none: bool = False) -> None:
    # Known-files keep None explicit — a "not filled in yet" signal for the
    # reader; the book omits it ("absent, not null").
    text = model.model_dump_json(indent=2, exclude_none=exclude_none)
    if output_path:
        Path(output_path).write_text(text)
    else:
        print(text)


def _unwritable(path: str) -> str | None:
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        return f"directory {parent!r} doesn't exist"
    if os.path.isdir(path):
        return "it's a directory — give a file path"
    if os.path.exists(path):
        return None if os.access(path, os.W_OK) else "file isn't writable"
    return None if os.access(parent, os.W_OK) else f"directory {parent!r} isn't writable"


def _run_paid[T](coro: Coroutine[Any, Any, T], failure: str) -> T | None:
    """Runs a command's paid work in a usage scope, printing its cost even
    when it fails (the money is spent either way). None after printing a
    failure.
    """
    with usage.track() as run_usage:
        try:
            return asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a clean stderr message, not a traceback
            print(f"{failure}: {exc}", file=sys.stderr)
            return None
        finally:
            _print_progress(run_usage.summary())


def _known_file_filename(isbn: str, known_file: KnownFile, used_names: set[str]) -> str:
    """Title-slug filename, falling back to the isbn when the lookup found
    no title or the slug collides with an earlier file in the same batch.
    """
    base = (slugify_title(known_file.title) if known_file.has_title else "") or isbn
    filename = f"{base}.json"
    suffix = 1
    while filename in used_names:
        filename = f"{base}-{isbn}.json" if suffix == 1 else f"{base}-{isbn}-{suffix}.json"
        suffix += 1
    return filename


_BATCH_KIND_DEFAULT: Literal["fiction", "non-fiction"] = "non-fiction"


def _cmd_create_known_file(args: argparse.Namespace) -> int:
    isbns: list[str] = args.isbn
    batch = len(isbns) > 1
    if batch and args.kind:
        print(
            "--kind can't be used with more than one isbn — kind is book-specific. Omit it; each known-file "
            f"defaults to kind: {_BATCH_KIND_DEFAULT} and must be corrected by hand.",
            file=sys.stderr,
        )
        return 1
    if batch and not args.output_dir:
        print("--output-dir is required when creating known-files for more than one isbn", file=sys.stderr)
        return 1
    if args.output and args.output_dir:
        print("--output and --output-dir can't both be given", file=sys.stderr)
        return 1

    kind: Literal["fiction", "non-fiction"] = args.kind or _BATCH_KIND_DEFAULT
    known_files = []
    for isbn in isbns:
        known_file, notes = create_known_file(isbn, kind=kind)
        known_files.append((isbn, known_file))
        for note in notes:
            _print_progress(f"{isbn}: {note}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        used_names: set[str] = set()
        for isbn, known_file in known_files:
            filename = _known_file_filename(isbn, known_file, used_names)
            used_names.add(filename)
            _write_output(known_file, os.path.join(args.output_dir, filename))
    else:
        _write_output(known_files[0][1], args.output)

    if not args.kind:
        _print_progress(
            f"kind defaulted to '{_BATCH_KIND_DEFAULT}' for {len(known_files)} known-file(s) — "
            "correct `kind` by hand before generating."
        )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    known_file = _load_ready_known_file(args.known_file)
    if known_file is None:
        return 1
    # Before any paid work, so a finished book is never lost to a bad path.
    if args.output and (problem := _unwritable(args.output)):
        print(f"can't write --output {args.output!r}: {problem}", file=sys.stderr)
        return 1

    generated = _run_paid(
        generate(
            known_file,
            slug=_slug_from_path(args.known_file),
            trust_known=args.trust_known,
            fresh=args.fresh,
            on_progress=_print_progress,
        ),
        "generation failed",
    )
    if generated is None:
        return 1
    book = generated.book
    try:
        _write_output(book, args.output, exclude_none=True)
    except OSError as exc:
        # Checked up front, but a disk can still fill or a mount vanish —
        # the book is paid for, so put it on stdout rather than lose it.
        print(f"couldn't write --output {args.output!r} ({exc}); printing the book to stdout instead", file=sys.stderr)
        _write_output(book, None, exclude_none=True)
        return 1
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    known_file = _load_ready_known_file(args.known_file)
    if known_file is None:
        return 1
    found = _run_paid(
        book_research.research(
            known_file,
            slug=_slug_from_path(args.known_file),
            trust_known=args.trust_known,
            fresh=args.fresh,
            on_progress=_print_progress,
        ),
        "research failed",
    )
    if found is None:
        return 1
    for source in found.sources:
        _print_progress(f"  {source.id}: {source.url} ({len(source.text):,} chars)")
    print(found.render())
    return 0


def _cmd_tags(args: argparse.Namespace) -> int:
    _write_output(TagVocabulary(), args.output)
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    try:
        books = eval_data.load_books(args.evals_dir, args.book)
    except (OSError, ValueError) as exc:
        print(f"could not load the eval set: {exc}", file=sys.stderr)
        return 1
    try:
        results = asyncio.run(
            eval_runner.run_eval(
                args.evals_dir,
                args.label,
                books,
                trust_known=args.trust_known,
                fresh=args.fresh,
                on_progress=_print_progress,
            )
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a clean stderr message, not a traceback
        print(f"eval run failed: {exc}", file=sys.stderr)
        return 1
    _print_progress(eval_metrics.format_summary(args.label, eval_metrics.summarize(results)))
    return 1 if any(r.get("error") for r in results) else 0


def _cmd_eval_judge(args: argparse.Namespace) -> int:
    model = args.judge_model or settings.judge_model
    if not model:
        print("no judge model: set PRECIS_JUDGE_MODEL or pass --judge-model", file=sys.stderr)
        return 1
    try:
        books = eval_data.load_books(args.evals_dir, args.book)
        judgement = asyncio.run(
            eval_judge.judge_runs(
                args.evals_dir,
                args.candidate,
                args.baseline,
                books,
                model=model,
                concurrency=settings.concurrency,
                on_progress=_print_progress,
            )
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a clean stderr message, not a traceback
        print(f"judging failed: {exc}", file=sys.stderr)
        return 1
    for label in (args.candidate, args.baseline):
        run_metrics = [
            m for book in books if (m := eval_data.read_json(eval_data.metrics_path(args.evals_dir, label, book.slug)))
        ]
        _print_progress(eval_metrics.format_summary(label, eval_metrics.summarize(run_metrics)))
    _print_progress(eval_judge.format_judgement(judgement))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-known-file", help="ISBN(s) -> known-file(s), looked up on Open Library")
    create.add_argument("isbn", nargs="+")
    create.add_argument("--kind", choices=["fiction", "non-fiction"])
    create.add_argument("--output", help="single-isbn only")
    create.add_argument("--output-dir", help="required for more than one isbn")
    create.set_defaults(func=_cmd_create_known_file)

    generate_cmd = subparsers.add_parser("generate", help="research a book and write its notes")
    generate_cmd.add_argument("known_file")
    generate_cmd.add_argument("--output")
    generate_cmd.add_argument("--trust-known", action="store_true", help="warn instead of failing the book/author checks")
    generate_cmd.add_argument("--fresh", action="store_true", help="search again instead of using the research cache")
    generate_cmd.set_defaults(func=_cmd_generate)

    research_cmd = subparsers.add_parser(
        "research", help="search the web for a book (or reuse its cache) and print the research the notes are written from"
    )
    research_cmd.add_argument("known_file")
    research_cmd.add_argument("--trust-known", action="store_true", help="warn instead of failing the book/author checks")
    research_cmd.add_argument("--fresh", action="store_true", help="search again instead of using the cache")
    research_cmd.set_defaults(func=_cmd_research)

    tags_cmd = subparsers.add_parser(
        "tags", help="print the closed tag vocabulary, for a consumer repo to sync its own copy against"
    )
    tags_cmd.add_argument("--output")
    tags_cmd.set_defaults(func=_cmd_tags)

    eval_cmd = subparsers.add_parser("eval", help="run the eval set and judge runs against each other")
    eval_subparsers = eval_cmd.add_subparsers(dest="eval_command", required=True)

    eval_run = eval_subparsers.add_parser(
        "run", help="generate every eval book into runs/<label>/ (books already there are skipped)"
    )
    eval_run.add_argument("label", help="names the run, e.g. sonnet-5")
    eval_run.add_argument("--book", action="append", help="only this eval book (slug); repeatable")
    eval_run.add_argument("--trust-known", action="store_true")
    eval_run.add_argument("--fresh", action="store_true", help="search again instead of using each book's research cache")
    eval_run.add_argument("--evals-dir", default=settings.evals_dir)
    eval_run.set_defaults(func=_cmd_eval_run)

    eval_judge_cmd = eval_subparsers.add_parser("judge", help="pairwise-judge a candidate run against a baseline run")
    eval_judge_cmd.add_argument("candidate")
    eval_judge_cmd.add_argument("baseline")
    eval_judge_cmd.add_argument("--judge-model", help="defaults to PRECIS_JUDGE_MODEL")
    eval_judge_cmd.add_argument("--book", action="append", help="only this eval book (slug); repeatable")
    eval_judge_cmd.add_argument("--evals-dir", default=settings.evals_dir)
    eval_judge_cmd.set_defaults(func=_cmd_eval_judge)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
