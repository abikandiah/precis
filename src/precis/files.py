"""Atomic file writing for the caches."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Writes `path` whole or not at all, through a temp file of its own so
    two writers of one path don't replace each other's. The file gets the
    mode a plain write would (0666 less the umask), not mkstemp's 0600.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            umask = os.umask(0)
            os.umask(umask)
            os.fchmod(f.fileno(), 0o666 & ~umask)
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
