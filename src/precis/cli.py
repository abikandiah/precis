"""CLI entrypoint. See docs/blueprint.md's CLI contract section.

No output file is ever written on a whole-book structural failure — errors
go to stderr with a non-zero exit code instead.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Literal

from pydantic import BaseModel

from precis.known_file import create_known_file, preflight_check, slugify_title
from precis.pipeline import graph as pipeline_graph
from precis.pipeline.nodes import draft
from precis.schema import Chapter, KnownFile


def _load_known_file(path: str) -> KnownFile:
    with open(path) as f:
        return KnownFile.model_validate_json(f.read())


def _load_known_file_or_report(path: str) -> KnownFile | None:
    """Loads a known-file, or prints a clean message and returns None on
    failure — the caller returns 1 immediately when this returns None.
    Shared by both generate and generate-chapter, which otherwise had
    byte-for-byte identical try/except blocks here.
    """
    try:
        return _load_known_file(path)
    except (OSError, ValueError) as exc:
        print(f"could not load known-file {path!r}: {exc}", file=sys.stderr)
        return None


def _write_output(model: BaseModel, output_path: str | None) -> None:
    text = model.model_dump_json(indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    else:
        print(text)


def _known_file_filename(isbn: str, known_file: KnownFile, used_names: set[str]) -> str:
    """Title-slug filename, falling back to the isbn when the lookup didn't
    find a title (still a placeholder) or when the slug collides with an
    earlier file in the same batch — including repeat collisions (e.g. the
    same isbn passed more than once), which the first-collision-only
    fallback used to silently overwrite instead of resolving.
    """
    slug = slugify_title(known_file.title) if known_file.has_title else ""
    base = slug or isbn
    filename = f"{base}.json"
    if filename not in used_names:
        return filename
    filename = f"{base}-{isbn}.json"
    suffix = 2
    while filename in used_names:
        filename = f"{base}-{isbn}-{suffix}.json"
        suffix += 1
    return filename


def _report_preflight_problems(known_file: KnownFile) -> int | None:
    """Prints and returns exit code 1 if the known-file isn't ready;
    returns None if it's clear to proceed. Shared by both subcommands —
    generate-chapter needs the same isbn/narrative-edge-case checks as
    generate, even though its own chapter-range check (below) happens to
    also catch an empty chapters list.
    """
    problems = preflight_check(known_file)
    if not problems:
        return None
    print("known-file is not ready for generation:", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    return 1


_BATCH_KIND_DEFAULT: Literal["fiction", "non-fiction"] = "non-fiction"


def _cmd_create_known_file(args: argparse.Namespace) -> int:
    isbns: list[str] = args.isbn
    batch = len(isbns) > 1

    if batch and args.kind:
        print(
            "--kind can't be used with more than one isbn — kind is book-specific "
            "and can't be applied uniformly across a batch. Omit --kind; each "
            f"created known-file will default to kind: {_BATCH_KIND_DEFAULT} and "
            "must be corrected by hand.",
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
    known_files = [(isbn, create_known_file(isbn, kind=kind, narrative=args.narrative)) for isbn in isbns]

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
        print(
            f"kind defaulted to '{_BATCH_KIND_DEFAULT}' for {len(known_files)} known-file(s) — "
            "review and correct `kind`/`narrative` by hand before generating.",
            file=sys.stderr,
        )
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    known_file = _load_known_file_or_report(args.known_file)
    if known_file is None:
        return 1

    if (exit_code := _report_preflight_problems(known_file)) is not None:
        return exit_code

    try:
        book = asyncio.run(
            pipeline_graph.run_whole_book(known_file, trust_known=args.trust_known, fresh=args.fresh)
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a clean stderr message, not a traceback
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1

    _write_output(book, args.output)
    return 0


def _cmd_generate_chapter(args: argparse.Namespace) -> int:
    known_file = _load_known_file_or_report(args.known_file)
    if known_file is None:
        return 1

    if (exit_code := _report_preflight_problems(known_file)) is not None:
        return exit_code

    if not (1 <= args.chapter <= len(known_file.chapters)):
        print(
            f"--chapter {args.chapter} is out of range for a known-file "
            f"with {len(known_file.chapters)} chapters",
            file=sys.stderr,
        )
        return 1

    try:
        result = asyncio.run(
            draft.run_one(
                {
                    "known_file": known_file.model_dump(),
                    "chapter_number": args.chapter,
                    "chapter_title": known_file.chapters[args.chapter - 1],
                }
            )
        )
        chapter_dicts = result.get("chapters") or []
        if not chapter_dicts:
            raise ValueError("chapter drafting returned no chapter — this is a bug in Stage 2, not user error")
        chapter = Chapter.model_validate(chapter_dicts[0])
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a clean stderr message, not a traceback
        print(f"chapter generation failed: {exc}", file=sys.stderr)
        return 1

    _write_output(chapter, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_known_file_cmd = subparsers.add_parser(
        "create-known-file", help="phase 1: ISBN(s) -> known-file(s)"
    )
    create_known_file_cmd.add_argument("isbn", nargs="+")
    create_known_file_cmd.add_argument("--kind", choices=["fiction", "non-fiction"])
    create_known_file_cmd.add_argument("--narrative", action="store_true")
    create_known_file_cmd.add_argument("--output", help="single-isbn only")
    create_known_file_cmd.add_argument("--output-dir", help="required for more than one isbn")
    create_known_file_cmd.set_defaults(func=_cmd_create_known_file)

    generate = subparsers.add_parser("generate", help="whole-book generation")
    generate.add_argument("known_file")
    generate.add_argument("--output")
    generate.add_argument("--trust-known", action="store_true")
    generate.add_argument("--fresh", action="store_true")
    generate.set_defaults(func=_cmd_generate)

    generate_chapter = subparsers.add_parser(
        "generate-chapter", help="targeted regeneration of a single chapter"
    )
    generate_chapter.add_argument("known_file")
    generate_chapter.add_argument("--chapter", type=int, required=True)
    generate_chapter.add_argument("--output")
    generate_chapter.set_defaults(func=_cmd_generate_chapter)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
