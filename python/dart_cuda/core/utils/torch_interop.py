"""PyTorch → dart_cuda checkpoint conversion helpers.

PyTorch saves a pickled `state_dict` (a `dict[str, Tensor]`); dart_cuda
expects a flat little-endian `float32` blob laid out in `module.parameters()`
order. This module bridges the two.

Requirements: `torch` (and obviously a matching architecture in dart_cuda).

Typical usage
-------------

```python
import torch
from dart_cuda.core.utils.torch_interop import (
    convert_state_dict,
    save_for_dart_cuda,
)

sd = torch.load("model.pt", map_location="cpu")
# Map of dart_cuda parameter index -> tensor coming from PyTorch.
# Use list_module_parameters(model) to see the slot order.
mapping = {
    0: sd["embedding.weight"],          # wte: [vocab, embed]
    1: sd["pos_embedding"],             # wpe: [block, embed]
    2: sd["block.0.attn.q.weight"].T,   # Layer.w: needs transpose
    3: sd["block.0.attn.q.bias"],
    # ...
}
save_for_dart_cuda(model_dc, mapping, "model.bin")
```
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Iterable, List, Sequence

from ..tensor.gpu_tensor import Tensor


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def list_module_parameters(module: Any) -> List[Dict[str, Any]]:
    """Return a list of `{index, shape, length}` for every parameter slot.

    Use this before writing a mapping so you know the expected size of every
    slot in `module.parameters()` order.
    """
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(module.parameters()):
        out.append({"index": i, "shape": list(p.shape), "length": p.length})
    return out


def total_parameter_count(module: Any) -> int:
    return sum(p.length for p in module.parameters())


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _flatten_to_floats(value: Any, *, expected_len: int, slot: int) -> List[float]:
    """Coerce any tensor-like / array-like object to a flat `list[float]`.

    Supports torch tensors, numpy arrays, and Python lists. Validates length.
    Caller is responsible for transposes/reshapes — this function does not
    reorder data.
    """
    # torch.Tensor
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            flat = value.detach().cpu().contiguous().float().view(-1).tolist()
            if len(flat) != expected_len:
                raise ValueError(
                    f"slot {slot}: tensor has {len(flat)} floats, "
                    f"dart_cuda parameter expects {expected_len}"
                )
            return flat
    except ImportError:
        pass

    # numpy
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            flat = np.ascontiguousarray(value, dtype=np.float32).reshape(-1).tolist()
            if len(flat) != expected_len:
                raise ValueError(
                    f"slot {slot}: array has {len(flat)} floats, "
                    f"dart_cuda parameter expects {expected_len}"
                )
            return flat
    except ImportError:
        pass

    # plain iterable
    flat = [float(v) for v in value]
    if len(flat) != expected_len:
        raise ValueError(
            f"slot {slot}: got {len(flat)} floats, "
            f"dart_cuda parameter expects {expected_len}"
        )
    return flat


def apply_mapping(module: Any, mapping: Dict[int, Any]) -> None:
    """Push PyTorch-sourced tensors directly into dart_cuda parameters.

    `mapping[i]` must be the host-side weight tensor that corresponds to the
    `i`-th entry of `module.parameters()`. Any slot not present in the mapping
    is left untouched.
    """
    params = list(module.parameters())
    if not mapping:
        raise ValueError("mapping is empty")
    for i, src in mapping.items():
        if i < 0 or i >= len(params):
            raise IndexError(f"slot {i} out of range (module has {len(params)} parameters)")
        p: Tensor = params[i]
        p.data = _flatten_to_floats(src, expected_len=p.length, slot=i)


def save_for_dart_cuda(module: Any, mapping: Dict[int, Any], file_path: str) -> None:
    """Apply `mapping` to `module`, then write a dart_cuda binary checkpoint.

    The resulting file is bit-identical to one produced by
    `save_module_binary` and can be loaded later with `load_module_binary`
    (Python) or `saveModuleBinary` (Dart).
    """
    apply_mapping(module, mapping)
    # Write directly to avoid a second host↔device round-trip.
    with open(file_path, "wb") as f:
        for p in module.parameters():
            data = p.fetch_data()
            f.write(struct.pack(f"<{len(data)}f", *data))


# ---------------------------------------------------------------------------
# Convenience helpers — common rearrangements
# ---------------------------------------------------------------------------

def transpose_linear(weight: Any) -> Any:
    """`nn.Linear.weight` is `[out, in]`; dart_cuda `Layer.w` is `[in, out]`.

    Returns the transposed tensor in whatever framework it came from.
    """
    try:
        import torch  # type: ignore

        if isinstance(weight, torch.Tensor):
            return weight.t().contiguous()
    except ImportError:
        pass
    try:
        import numpy as np  # type: ignore

        if isinstance(weight, np.ndarray):
            return np.ascontiguousarray(weight.T)
    except ImportError:
        pass
    raise TypeError(f"transpose_linear: unsupported type {type(weight)!r}")


def flatten_conv2d_weight(weight: Any) -> Any:
    """`nn.Conv2d.weight` is `[O, I, kH, kW]`; dart_cuda `Conv2d` uses
    `[O, I*kH*kW]` for the im2col path. Returns a reshape (no transpose)."""
    try:
        import torch  # type: ignore

        if isinstance(weight, torch.Tensor):
            o = weight.shape[0]
            return weight.detach().contiguous().view(o, -1)
    except ImportError:
        pass
    try:
        import numpy as np  # type: ignore

        if isinstance(weight, np.ndarray):
            o = weight.shape[0]
            return np.ascontiguousarray(weight).reshape(o, -1)
    except ImportError:
        pass
    raise TypeError(f"flatten_conv2d_weight: unsupported type {type(weight)!r}")


def map_mlp_state_dict(
    module_dc: Any,
    state_dict: Dict[str, Any],
    *,
    weight_keys: Sequence[str],
    bias_keys: Sequence[str],
) -> Dict[int, Any]:
    """Build a `{slot: tensor}` mapping for a dart_cuda `MLP`.

    `MLP.parameters()` returns `[w_0, b_0, w_1, b_1, ...]` for each `Layer`.
    Provide the PyTorch state-dict key for each linear weight / bias in the
    same order; this helper transposes weights for you.
    """
    if len(weight_keys) != len(bias_keys):
        raise ValueError("weight_keys and bias_keys must have equal length")
    mapping: Dict[int, Any] = {}
    idx = 0
    for wk, bk in zip(weight_keys, bias_keys):
        mapping[idx] = transpose_linear(state_dict[wk])
        mapping[idx + 1] = state_dict[bk]
        idx += 2
    return mapping
