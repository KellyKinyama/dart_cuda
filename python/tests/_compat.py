"""Minimal pytest-compatible stub so the test suite runs without pytest installed.

Only the subset of pytest used by `python/tests/*.py` is implemented:

* ``approx(expected, rel=1e-6, abs=1e-3)`` — comparable via ``==`` to scalars
  or sequences.
* ``raises(ExceptionType)`` — context manager.
* ``fixture`` — decorator that simply marks a function; the runner injects
  the result by name.
* ``mark.skip(reason="...")`` — decorator that marks a test as skipped.
* ``skip(reason)``, ``SkipException`` — used inside tests.

If real pytest is importable, the runner uses it instead and this module is
ignored.
"""

from __future__ import annotations

import math
from typing import Iterable, Union


class SkipException(Exception):
    pass


def skip(reason: str = "") -> None:
    raise SkipException(reason)


class Approx:
    def __init__(self, expected, rel: float | None = None, abs: float | None = None):
        self.expected = expected
        self.rel = rel
        self.abs = abs if abs is not None else 1e-3

    def _tol(self, ref: float) -> float:
        t = self.abs
        if self.rel is not None:
            t = max(t, self.rel * builtin_abs(ref))
        return t

    def __eq__(self, other) -> bool:  # noqa: D401
        e = self.expected
        if isinstance(e, (int, float)):
            try:
                return builtin_abs(float(other) - float(e)) <= self._tol(e)
            except (TypeError, ValueError):
                return False
        # sequence comparison
        try:
            other_seq = list(other)
        except TypeError:
            return False
        e_seq = list(e)
        if len(e_seq) != len(other_seq):
            return False
        return all(
            builtin_abs(float(o) - float(x)) <= self._tol(float(x))
            for o, x in zip(other_seq, e_seq)
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"approx({self.expected!r}, abs={self.abs}, rel={self.rel})"


def approx(expected, rel: float | None = None, abs: float | None = None) -> Approx:
    return Approx(expected, rel=rel, abs=abs)


builtin_abs = abs  # save before any user-level shadowing


class _RaisesCtx:
    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(
                f"DID NOT RAISE {self.exc_type.__name__}"
            )
        if not issubclass(exc_type, self.exc_type):
            return False
        self.value = exc_val
        return True


def raises(exc_type):
    return _RaisesCtx(exc_type)


def fixture(func=None, **_kwargs):
    """Decorator: mark a function as a fixture for the runner."""
    def _wrap(f):
        f._is_pytest_fixture = True
        return f
    if func is None:
        return _wrap
    return _wrap(func)


class _Mark:
    def skip(self, reason: str = ""):
        def deco(func):
            func._pytest_skip = reason or "skipped"
            return func
        return deco


mark = _Mark()


# Allow ``import pytest`` to land on this module and expose the surface.
__all__ = [
    "approx",
    "raises",
    "fixture",
    "mark",
    "skip",
    "SkipException",
]
