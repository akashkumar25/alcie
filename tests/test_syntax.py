import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "alcie"
FILES = sorted(ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)
