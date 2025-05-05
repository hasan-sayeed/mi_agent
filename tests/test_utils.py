import os
import pytest
from mi_agent.utils import ensure_dir, move_file, is_code_error

def test_ensure_dir_creates(tmp_path):
    d = tmp_path / "newdir"
    assert not d.exists()
    ensure_dir(str(d))
    assert d.exists() and d.is_dir()

def test_move_file(tmp_path):
    # create a dummy file
    src = tmp_path / "foo.txt"
    src.write_text("hello")
    dst_dir = tmp_path / "out"
    new_path = move_file(str(src), str(dst_dir))
    assert os.path.basename(new_path) == "foo.txt"
    assert os.path.exists(new_path)
    assert not src.exists()

@pytest.mark.parametrize("output,expected", [
    ("everything fine", False),
    ("Traceback (most recent call last):", True),
    ("some ERROR happened", True),
    ("no exceptions", True),
    ("normal output", False),
])
def test_is_code_error(output, expected):
    assert is_code_error(output) is expected
