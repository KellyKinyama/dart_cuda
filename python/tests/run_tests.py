"""Run the dart_cuda Python test suite using either real pytest (if installed)
or the stdlib-only stub in ``_compat.py``.

Usage from the repo root:

    PYTHONPATH=python python3 python/tests/run_tests.py            # all tests
    PYTHONPATH=python python3 python/tests/run_tests.py test_layers test_attention
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PYTHON_PKG = REPO_ROOT / "python"

if str(PYTHON_PKG) not in sys.path:
    sys.path.insert(0, str(PYTHON_PKG))


def _install_pytest_stub() -> bool:
    """Install local stub as ``pytest`` if real pytest is missing.

    Returns True if the stub was installed, False if real pytest is in use.
    """
    try:
        import pytest  # noqa: F401
        return False
    except ModuleNotFoundError:
        from python.tests import _compat  # noqa: F401  (when invoked as module)
    except ImportError:
        pass

    try:
        import pytest  # noqa: F401
        return False
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location("pytest", HERE / "_compat.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load pytest stub")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["pytest"] = module
    return True


def _engine_available() -> bool:
    lib = REPO_ROOT / "native" / "lib" / "libmat_mul.so"
    if not lib.exists():
        return False
    try:
        from dart_cuda.core.tensor.gpu_tensor import Tensor
        t = Tensor.fill([1, 1], 0.0)
        t.dispose()
        return True
    except Exception:
        return False


def _run_module(module, results: dict) -> None:
    """Run all test classes & functions in `module` with our minimal fixture
    injection."""
    import pytest  # may be real or stub

    # Module-level fixtures (e.g. tracker from conftest) are collected into
    # this dict, indexed by name. We re-resolve per-test invocation.
    module_fixtures = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and getattr(obj, "_is_pytest_fixture", False):
            module_fixtures[name] = obj

    # Plus conftest fixtures.
    try:
        import conftest as cf  # type: ignore
        for name, obj in inspect.getmembers(cf):
            if inspect.isfunction(obj) and getattr(obj, "_is_pytest_fixture", False):
                module_fixtures.setdefault(name, obj)
    except Exception:
        pass

    # Discover tests: top-level test_* functions and Test* classes.
    items = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("test_") and inspect.isfunction(obj):
            items.append((f"{module.__name__}::{name}", None, obj))
        elif name.startswith("Test") and inspect.isclass(obj):
            instance = obj()
            for mname, mobj in inspect.getmembers(instance):
                if mname.startswith("test_") and inspect.ismethod(mobj):
                    items.append(
                        (f"{module.__name__}::{name}::{mname}", instance, mobj)
                    )

    for label, instance, fn in items:
        skip_reason = getattr(fn, "_pytest_skip", None)
        if skip_reason:
            results["skipped"] += 1
            print(f"  s  {label}  (skipped: {skip_reason})")
            continue

        sig = inspect.signature(fn)
        kwargs = {}
        teardown_gens = []
        try:
            for pname, p in sig.parameters.items():
                if pname == "self":
                    continue
                if pname in module_fixtures:
                    fix = module_fixtures[pname]
                    res = fix()
                    if inspect.isgenerator(res):
                        kwargs[pname] = next(res)
                        teardown_gens.append(res)
                    else:
                        kwargs[pname] = res
                else:
                    # No fixture; pass None and hope.
                    kwargs[pname] = None
            fn(**kwargs)
            print(f"  .  {label}")
            results["passed"] += 1
        except getattr(pytest, "SkipException", type("X", (), {})) as e:
            print(f"  s  {label}  (skipped: {e})")
            results["skipped"] += 1
        except AssertionError as e:
            results["failed"] += 1
            results["failures"].append((label, traceback.format_exc()))
            print(f"  F  {label}")
        except Exception:
            results["failed"] += 1
            results["failures"].append((label, traceback.format_exc()))
            print(f"  E  {label}")
        finally:
            for g in teardown_gens:
                try:
                    next(g)
                except StopIteration:
                    pass
                except Exception:
                    traceback.print_exc()


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    using_stub = _install_pytest_stub()
    if using_stub:
        print("[run_tests] using local pytest stub (_compat.py)")
    else:
        print("[run_tests] using installed pytest")

    if not _engine_available():
        print("[run_tests] CUDA engine (native/lib/libmat_mul.so) unavailable — skipping all tests")
        return 0

    # Make conftest importable.
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

    if argv:
        names = [a if a.startswith("test_") else f"test_{a}" for a in argv]
    else:
        names = sorted(
            p.stem for p in HERE.glob("test_*.py")
        )

    results = {"passed": 0, "failed": 0, "skipped": 0, "failures": []}
    for name in names:
        print(f"\n>>> {name}")
        try:
            mod = importlib.import_module(name)
        except Exception:
            results["failed"] += 1
            results["failures"].append((name, traceback.format_exc()))
            print(f"  E  {name} (import error)")
            continue
        _run_module(mod, results)

    print(
        f"\n=== {results['passed']} passed, "
        f"{results['failed']} failed, "
        f"{results['skipped']} skipped ==="
    )
    if results["failures"]:
        print("\n--- FAILURE DETAILS ---")
        for label, tb in results["failures"]:
            print(f"\n[{label}]\n{tb}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
