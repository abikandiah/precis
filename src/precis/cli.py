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

from precis import llm, usage
from precis import research as book_research
from precis.book_file import BookFileError, read_book_file
from precis.config import ConfigError, settings
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


def _add_source_options(command: argparse.ArgumentParser) -> None:
    """--book-file and --overview: what the notes are written from (full
    notes from the reader's own copy, or an overview from search).
    """
    source = command.add_mutually_exclusive_group()
    source.add_argument(
        "--book-file", help="your own copy of the book (.epub, .pdf or .txt), in place of the known-file's book_file"
    )
    source.add_argument(
        "--overview", action="store_true", help="an overview from search, even when the known-file has a book_file"
    )


def _book_text(args: argparse.Namespace, known_file: KnownFile) -> str | None | Literal[False]:
    """The reader's own copy of the book: `--book-file` as given, or the
    known-file's `book_file` relative to the known-file (a leading ~ is the
    home directory). None when there's neither, or `--overview` asks for an
    overview anyway; False after printing why the file can't be read —
    before any paid work.
    """
    if args.overview:
        return None
    if args.book_file:
        path = Path(args.book_file)
    elif known_file.book_file:
        path = Path(known_file.book_file).expanduser()
        if not path.is_absolute():
            path = Path(args.known_file).parent / path
    else:
        return None
    try:
        return read_book_file(path)
    except BookFileError as exc:
        print(f"book_file: {exc}", file=sys.stderr)
        return False


def _write_output(model: BaseModel, output_path: str | None, *, exclude_none: bool = False) -> None:
    # Known-files keep None explicit — a "not filled in yet" signal for the
    # reader; the book omits it ("absent, not null").
    text = model.model_dump_json(indent=2, exclude_none=exclude_none)
    if output_path:
        Path(output_path).write_text(text)
    else:
        print(text)


def _write_or_report(model: BaseModel, output_path: str | None) -> bool:
    """_write_output, with a failure printed rather than raised."""
    try:
        _write_output(model, output_path)
    except OSError as exc:
        print(f"couldn't write {output_path!r}: {exc}", file=sys.stderr)
        return False
    return True


def _exists_unless_forced(path: str, force: bool) -> str | None:
    """A known-file is edited by hand after it's created, so an existing
    one is only replaced when asked.
    """
    return None if force or not os.path.exists(path) else "it already exists — pass --force to replace it"


def _unwritable(path: str) -> str | None:
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        return f"directory {parent!r} doesn't exist"
    if os.path.isdir(path):
        return "it's a directory — give a file path"
    if os.path.exists(path):
        return None if os.access(path, os.W_OK) else "file isn't writable"
    return None if os.access(parent, os.W_OK) else f"directory {parent!r} isn't writable"


def _is_input(output: str, known_file: str) -> str | None:
    """The known-file is edited by hand, so the book never replaces it."""
    same = os.path.exists(output) and os.path.samefile(output, known_file)
    return "it's the known-file — give another path" if same else None


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

    if args.output and (problem := _unwritable(args.output) or _exists_unless_forced(args.output, args.force)):
        print(f"can't write --output {args.output!r}: {problem}", file=sys.stderr)
        return 1
    if args.output_dir:
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as exc:
            print(f"can't create --output-dir {args.output_dir!r}: {exc}", file=sys.stderr)
            return 1

    kind: Literal["fiction", "non-fiction"] = args.kind or _BATCH_KIND_DEFAULT
    used_names: set[str] = set()
    ok = True
    # Each file is written as soon as it's looked up, so a failure partway
    # keeps the ones before it.
    for isbn in isbns:
        known_file, notes = create_known_file(isbn, kind=kind)
        for note in notes:
            _print_progress(f"{isbn}: {note}")
        if not args.output_dir:
            ok = _write_or_report(known_file, args.output)
            continue
        filename = _known_file_filename(isbn, known_file, used_names)
        used_names.add(filename)
        path = os.path.join(args.output_dir, filename)
        if problem := _exists_unless_forced(path, args.force):
            print(f"{isbn}: not writing {path!r}: {problem}", file=sys.stderr)
            ok = False
        else:
            ok = _write_or_report(known_file, path) and ok

    if not args.kind:
        _print_progress(
            f"kind defaulted to '{_BATCH_KIND_DEFAULT}' for {len(isbns)} known-file(s) — "
            "correct `kind` by hand before generating."
        )
    return 0 if ok else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    known_file = _load_ready_known_file(args.known_file)
    if known_file is None:
        return 1
    # Before any paid work, so a finished book is never lost to a bad path.
    if args.output and (problem := _unwritable(args.output) or _is_input(args.output, args.known_file)):
        print(f"can't write --output {args.output!r}: {problem}", file=sys.stderr)
        return 1
    if (book_text := _book_text(args, known_file)) is False:
        return 1
    # The model client reads these after research, by when an overview's
    # searches have spent their credits: a bad one fails here instead
    # (a ConfigError, reported by _dispatch).
    settings.check_llm()

    book = _run_paid(
        generate(
            known_file,
            slug=_slug_from_path(args.known_file),
            trust_known=args.trust_known,
            fresh=args.fresh,
            on_progress=_print_progress,
            book_text=book_text,
        ),
        "generation failed",
    )
    if book is None:
        return 1
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
    if (book_text := _book_text(args, known_file)) is False:
        return 1
    found = _run_paid(
        book_research.research(
            known_file,
            slug=_slug_from_path(args.known_file),
            trust_known=args.trust_known,
            fresh=args.fresh,
            on_progress=_print_progress,
            book_text=book_text,
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
    return 0 if _write_or_report(TagVocabulary(), args.output) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-known-file", help="ISBN(s) -> known-file(s), looked up on Open Library")
    create.add_argument("isbn", nargs="+")
    create.add_argument("--kind", choices=["fiction", "non-fiction"])
    create.add_argument("--output", help="single-isbn only")
    create.add_argument("--output-dir", help="required for more than one isbn")
    create.add_argument("--force", action="store_true", help="replace known-files that already exist")
    create.set_defaults(func=_cmd_create_known_file)

    generate_cmd = subparsers.add_parser("generate", help="research a book and write its notes")
    generate_cmd.add_argument("known_file")
    generate_cmd.add_argument("--output")
    generate_cmd.add_argument("--trust-known", action="store_true", help="warn instead of failing the book/author checks")
    generate_cmd.add_argument("--fresh", action="store_true", help="search again instead of using the research cache")
    _add_source_options(generate_cmd)
    generate_cmd.set_defaults(func=_cmd_generate)

    research_cmd = subparsers.add_parser(
        "research", help="search the web for a book (or reuse its cache) and print the research the notes are written from"
    )
    research_cmd.add_argument("known_file")
    research_cmd.add_argument("--trust-known", action="store_true", help="warn instead of failing the book/author checks")
    research_cmd.add_argument("--fresh", action="store_true", help="search again instead of using the cache")
    _add_source_options(research_cmd)
    research_cmd.set_defaults(func=_cmd_research)

    tags_cmd = subparsers.add_parser(
        "tags", help="print the closed tag vocabulary, for a consumer repo to sync its own copy against"
    )
    tags_cmd.add_argument("--output")
    tags_cmd.set_defaults(func=_cmd_tags)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    llm.report_sdk_retries()
    sys.exit(_dispatch(args))


def _dispatch(args: argparse.Namespace) -> int:
    """Runs the command. A bad setting surfaces only once a command reads
    it (config.py) — `generate` reads the model's up front, before any paid
    work — and is reported here: a clean stderr message, never a traceback.
    So is Ctrl-C.
    """
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"precis: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("precis: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    main()
