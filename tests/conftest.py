"""Test setup: run without the heavy, GPU-oriented dependencies.

``clip`` (OpenAI CLIP, installed from git) is only used for feature extraction at
training time, so a stub is enough to import the memory-management modules.
Memory buffers append to ``logs/*.log`` relative to the working directory, so each
test runs inside a temporary directory that already contains ``logs/``.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("clip") is None:
    sys.modules["clip"] = types.ModuleType("clip")


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path
