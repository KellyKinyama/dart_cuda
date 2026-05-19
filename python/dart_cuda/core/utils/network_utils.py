"""Alternative network persistence — mirrors `lib/core/utils/network_utils.dart`.

The Dart `network_utils.dart` is a near-duplicate of `persistence.dart`; this
file re-exports the same helpers for parity.
"""

from .persistence import (
    load_module_binary,
    loadModuleBinary,
    save_module_binary,
    saveModuleBinary,
)

__all__ = [
    "save_module_binary",
    "load_module_binary",
    "saveModuleBinary",
    "loadModuleBinary",
]
