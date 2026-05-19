"""Image utilities — mirrors `lib/loaders/images.dart`.

Requires Pillow: `pip install Pillow`.
"""

from __future__ import annotations

from typing import List

from ..core.tensor.gpu_tensor import Tensor


def _require_pil():
    try:
        from PIL import Image  # noqa: F401

        return Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Pillow is required for image loading: pip install Pillow"
        ) from exc


def image_to_tensor(path: str, target_size: int) -> Tensor:
    Image = _require_pil()
    img = Image.open(path).convert("RGB").resize((target_size, target_size))
    flat: List[float] = []
    for r, g, b in img.getdata():
        flat.append(r / 255.0)
        flat.append(g / 255.0)
        flat.append(b / 255.0)
    return Tensor.from_list([1, target_size * target_size * 3], flat)


imageToTensor = image_to_tensor
