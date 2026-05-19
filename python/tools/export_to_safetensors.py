"""Export a PyTorch checkpoint to a `.safetensors` file consumable by
`lib/core/loaders/safetensors.dart`.

Usage:
    python3 python/tools/export_to_safetensors.py \
        --input  /path/to/pytorch_model.pt \
        --output /path/to/model.safetensors

If the input contains a full nn.Module (rather than a state_dict), the
script will call `.state_dict()` automatically. All tensors are cast to
float32 because the Dart loader treats F32 as the canonical dtype
(F16 / BF16 are also supported on the Dart side but most of our layers
allocate F32, so converting here keeps things simple).

Requires: `pip install torch safetensors`.

Listing key names + shapes (no conversion):
    python3 python/tools/export_to_safetensors.py --input ckpt.pt --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_state_dict(path: Path):
    import torch  # imported lazily so `--help` works without torch installed

    obj = torch.load(str(path), map_location="cpu")
    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()
    if not isinstance(obj, dict):
        raise SystemExit(
            f"unsupported checkpoint contents in {path}: expected a dict "
            f"or nn.Module, got {type(obj).__name__}"
        )
    return obj


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", "-i", required=True, type=Path)
    p.add_argument("--output", "-o", type=Path)
    p.add_argument(
        "--list",
        action="store_true",
        help="Just print key names + shapes; do not write anything.",
    )
    args = p.parse_args(argv)

    state = _load_state_dict(args.input)

    if args.list:
        for k, v in state.items():
            shape = tuple(v.shape) if hasattr(v, "shape") else ("?",)
            dtype = getattr(v, "dtype", "?")
            print(f"{k}\t{tuple(shape)}\t{dtype}")
        return 0

    if args.output is None:
        raise SystemExit("--output is required unless --list is given")

    import torch  # noqa: F401
    from safetensors.torch import save_file

    # Cast everything to float32 and make sure tensors are contiguous; the
    # Dart loader assumes C-contiguous, little-endian layouts.
    cast = {k: v.detach().float().contiguous() for k, v in state.items()}

    save_file(cast, str(args.output))
    print(f"wrote {len(cast)} tensors to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
