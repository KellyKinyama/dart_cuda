"""Convert a PyTorch MLP checkpoint into a dart_cuda binary checkpoint.

Run with `pip install torch` first. Demonstrates the full round-trip:

  PyTorch MLP  ->  state_dict  ->  dart_cuda MLP  ->  dart_cuda .bin
                                                        |
                                                        v
                              load_module_binary back into a fresh MLP
                              and verify the forward pass matches PyTorch.
"""

from __future__ import annotations

import os
import tempfile

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This example requires PyTorch: pip install torch"
    ) from exc

import dart_cuda as dc
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.utils.persistence import load_module_binary
from dart_cuda.core.utils.torch_interop import (
    map_mlp_state_dict,
    save_for_dart_cuda,
)


def main() -> None:
    # 1. Build a PyTorch MLP (matching dart_cuda's GELU-between-hidden layout).
    torch.manual_seed(0)
    torch_mlp = nn.Sequential(
        nn.Linear(2, 8),
        nn.GELU(),
        nn.Linear(8, 1),
    )
    torch_mlp.eval()

    # 2. Build a dart_cuda MLP with the same shape.
    dc_mlp = MLP(2, [8, 1])

    # 3. Convert the state_dict and save as a dart_cuda binary.
    sd = torch_mlp.state_dict()
    mapping = map_mlp_state_dict(
        dc_mlp, sd,
        weight_keys=["0.weight", "2.weight"],
        bias_keys=["0.bias",   "2.bias"],
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mlp.bin")
        save_for_dart_cuda(dc_mlp, mapping, path)
        size = os.path.getsize(path)
        print(f"wrote dart_cuda checkpoint: {path} ({size} bytes)")

        # 4. Load into a fresh dart_cuda model to confirm the file is valid.
        dc_mlp2 = MLP(2, [8, 1])
        load_module_binary(dc_mlp2, path)

    # 5. Compare PyTorch vs dart_cuda forward passes.
    sample = [0.4, -0.7]
    with torch.no_grad():
        torch_out = torch_mlp(torch.tensor([sample])).item()

    tracker: list[dc.Tensor] = []
    x = dc.Tensor.from_list([1, 2], sample); tracker.append(x)
    dc_out = dc_mlp.forward(x, tracker).fetch_data()[0]
    dc_out2 = dc_mlp2.forward(x, tracker).fetch_data()[0]

    for t in tracker:
        t.dispose()

    print(f"\nsample input: {sample}")
    print(f"  PyTorch                : {torch_out:+.6f}")
    print(f"  dart_cuda (after apply): {dc_out:+.6f}")
    print(f"  dart_cuda (after load) : {dc_out2:+.6f}")
    print(f"  abs diff               : {abs(torch_out - dc_out):.2e}")


if __name__ == "__main__":
    main()
