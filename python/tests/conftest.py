"""Shared test helpers for dart_cuda Python tests.

Run from the repo root with:

    PYTHONPATH=python python3 -m pytest python/tests -q

These tests require the CUDA shared library at
``native/lib/libmat_mul.so`` and a working CUDA device. They skip
cleanly if the library or device is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_PKG = REPO_ROOT / "python"

if str(PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(PYTHON_PKG))


def _engine_available() -> bool:
    lib = REPO_ROOT / "native" / "lib" / "libmat_mul.so"
    if not lib.exists():
        return False
    try:
        from dart_cuda.core.tensor.gpu_tensor import Tensor  # noqa: F401
        t = Tensor.fill([1, 1], 0.0)
        t.dispose()
        return True
    except Exception:
        return False


_ENGINE_OK = _engine_available()


def pytest_collection_modifyitems(config, items):  # pragma: no cover
    if _ENGINE_OK:
        return
    skip = pytest.mark.skip(reason="CUDA engine (libmat_mul.so) not available")
    for item in items:
        item.add_marker(skip)


# ---------------------------------------------------------------------------
# tracker fixture: collects intermediate tensors and disposes them after test.
# ---------------------------------------------------------------------------
@pytest.fixture
def tracker():
    items = []
    yield items
    for t in items:
        try:
            t.dispose()
        except Exception:
            pass


def close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def all_close(seq_a, seq_b, tol: float = 1e-3) -> bool:
    if len(seq_a) != len(seq_b):
        return False
    return all(close(float(a), float(b), tol) for a, b in zip(seq_a, seq_b))
