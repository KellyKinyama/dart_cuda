"""Module binary persistence — mirrors `lib/core/utils/persistence.dart`.

Saves and loads parameters as a flat little-endian float32 blob, so files
written by Dart and Python are interchangeable.
"""

from __future__ import annotations

import os
import struct
from typing import Protocol

from ..tensor.gpu_tensor import Tensor


class _HasParameters(Protocol):
    def parameters(self) -> list[Tensor]: ...


def save_module_binary(module: _HasParameters, file_path: str) -> None:
    params = module.parameters()
    print("Syncing GPU weights for saving...")
    with open(file_path, "wb") as f:
        for p in params:
            data = p.fetch_data()
            f.write(struct.pack(f"<{len(data)}f", *data))
    size = os.path.getsize(file_path)
    print(f"Saved: {file_path} ({size} bytes)")


def load_module_binary(module: _HasParameters, file_path: str) -> bool:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    params = module.parameters()
    total_expected = sum(p.length for p in params)

    with open(file_path, "rb") as f:
        raw = f.read()
    floats = list(struct.unpack(f"<{len(raw)//4}f", raw))

    if len(floats) != total_expected:
        raise ValueError(
            f"Weight file size mismatch! Model expects {total_expected} floats, "
            f"file contains {len(floats)} floats."
        )

    print("Injecting weights into GPU...")
    offset = 0
    for p in params:
        length = p.length
        p.data = floats[offset : offset + length]
        offset += length
    print("Model weights successfully restored to GPU memory.")
    return True


# Dart-style aliases
saveModuleBinary = save_module_binary
loadModuleBinary = load_module_binary
