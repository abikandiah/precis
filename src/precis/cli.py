"""CLI entrypoint. See docs/blueprint.md's CLI contract section.

No output file is ever written on a whole-book structural failure — errors
go to stderr with a non-zero exit code instead.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pydantic import BaseModel

from precis.known_file import preflight_check
from precis.pipeline import graph as pipeline_graph
from precis.pipeline.nodes import draft
from precis.schema import Chapter, KnownFile


def _load_known_file(path: str) -> KnownFile:
    with open(path) as f:
        return KnownFile.model_validate_json(f.read())


def _write_output(model: BaseModel, output_path: str | None) -> None:
    text = model.model_dump_json(indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    else:
        print(text)


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


def _cmd_generate(args: argparse.Namespace) -> int:
    known_file = _load_known_file(args.known_file)

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
    known_file = _load_known_file(args.known_file)

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
