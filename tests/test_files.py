import os
import stat

from precis.files import write_atomic


def test_write_atomic_gives_the_mode_a_plain_write_would(tmp_path):
    umask = os.umask(0o022)
    try:
        path = tmp_path / "sub" / "book.json"
        write_atomic(path, "{}\n")
    finally:
        os.umask(umask)
    assert path.read_text() == "{}\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert list(path.parent.iterdir()) == [path]
