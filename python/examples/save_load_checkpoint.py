"""Save and reload a small MLP, verifying weights round-trip through disk.

Demonstrates `save_module_binary` / `load_module_binary`, which use the same
flat little-endian float32 layout as the Dart helpers — so a Dart checkpoint
loads in Python and vice versa.
"""

from __future__ import annotations

import os
import tempfile

import dart_cuda as dc
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.utils.persistence import load_module_binary, save_module_binary


def snapshot(model: MLP) -> list[list[float]]:
    return [list(p.fetch_data()) for p in model.parameters()]


def main() -> None:
    src = MLP(3, [8, 4, 2])
    dst = MLP(3, [8, 4, 2])

    src_before = snapshot(src)
    dst_before = snapshot(dst)
    assert src_before != dst_before, "models should start with different weights"

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mlp.bin")
        save_module_binary(src, path)
        load_module_binary(dst, path)

    src_after = snapshot(src)
    dst_after = snapshot(dst)

    print("src unchanged after save:", src_after == src_before)
    print("dst matches src after load:", dst_after == src_after)

    # Quick numerical proof
    x = dc.Tensor.from_list([1, 3], [0.1, -0.2, 0.3])
    tracker: list[dc.Tensor] = [x]
    y_src = src.forward(x, tracker).fetch_data()
    y_dst = dst.forward(x, tracker).fetch_data()
    print("forward(src) =", y_src)
    print("forward(dst) =", y_dst)
    for t in tracker:
        t.dispose()


if __name__ == "__main__":
    main()
