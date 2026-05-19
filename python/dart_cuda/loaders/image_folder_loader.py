"""Image folder classification + triplet loader — mirrors
`lib/loaders/image_folder_loader.dart`.

Requires Pillow.
"""

from __future__ import annotations

import array
import os
import random
from typing import Iterable, List, NamedTuple, Optional, Tuple


def _require_pil():
    try:
        from PIL import Image  # noqa: F401

        return Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required: pip install Pillow") from exc


class _Sample(NamedTuple):
    flat: "array.array"  # HxWx3 normalized [0,1]
    label: int


class Triplet(NamedTuple):
    anchor: "array.array"
    positive: "array.array"
    negative: "array.array"


class ImageFolderLoader:
    def __init__(
        self,
        root_path: str,
        *,
        image_size: int,
        patch_size: int,
        max_per_class: int = 1 << 30,
        max_classes: Optional[int] = None,
        val_split: float = 0.2,
        seed: int = 7,
    ):
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )
        self.root_path = root_path
        self.image_size = image_size
        self.patch_size = patch_size
        self._rng = random.Random(seed)
        if not os.path.isdir(root_path):
            raise ValueError(f"root_path does not exist: {root_path}")

        class_dirs = sorted(
            d
            for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
        )
        if max_classes is not None:
            class_dirs = class_dirs[:max_classes]

        self.classes: List[str] = []
        self._train: List[_Sample] = []
        self._val: List[_Sample] = []
        split_rng = random.Random(seed)

        for name in class_dirs:
            full = os.path.join(root_path, name)
            files = sorted(
                os.path.join(full, f)
                for f in os.listdir(full)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            if not files:
                continue
            cap = min(len(files), max_per_class)
            self.classes.append(name)
            class_idx = len(self.classes) - 1
            indices = list(range(cap))
            split_rng.shuffle(indices)
            n_val = round(cap * val_split)
            for k in range(cap):
                flat = self._decode(files[indices[k]])
                if flat is None:
                    continue
                s = _Sample(flat, class_idx)
                if k < n_val:
                    self._val.append(s)
                else:
                    self._train.append(s)

        self._train_by_class: List[List[int]] = [[] for _ in self.classes]
        for i, s in enumerate(self._train):
            self._train_by_class[s.label].append(i)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def num_train(self) -> int:
        return len(self._train)

    @property
    def num_val(self) -> int:
        return len(self._val)

    @property
    def patch_pixels(self) -> int:
        return self.patch_size * self.patch_size * 3

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def triplet_ready_classes(self) -> List[int]:
        return [c for c, idxs in enumerate(self._train_by_class) if len(idxs) >= 2]

    def _decode(self, path: str) -> Optional["array.array"]:
        try:
            Image = _require_pil()
            img = (
                Image.open(path)
                .convert("RGB")
                .resize((self.image_size, self.image_size))
            )
            buf = array.array("f")
            for r, g, b in img.getdata():
                buf.append(r / 255.0)
                buf.append(g / 255.0)
                buf.append(b / 255.0)
            return buf
        except Exception:
            return None

    def patchify(self, flat: "array.array") -> "array.array":
        H = W = self.image_size
        P = self.patch_size
        per_patch = P * P * 3
        n_p = self.num_patches
        out = array.array("f", [0.0] * (n_p * per_patch))
        patches_per_row = W // P
        for py in range(H // P):
            for px in range(patches_per_row):
                p_idx = py * patches_per_row + px
                out_base = p_idx * per_patch
                w = 0
                for dy in range(P):
                    y = py * P + dy
                    for dx in range(P):
                        x = px * P + dx
                        in_base = (y * W + x) * 3
                        out[out_base + w] = flat[in_base]
                        out[out_base + w + 1] = flat[in_base + 1]
                        out[out_base + w + 2] = flat[in_base + 2]
                        w += 3
        return out

    def sample_train_batch(self, batch_size: int) -> List[Tuple["array.array", int]]:
        out: List[Tuple["array.array", int]] = []
        for _ in range(batch_size):
            s = self._train[self._rng.randrange(len(self._train))]
            out.append((self.patchify(s.flat), s.label))
        return out

    def val_samples(self) -> Iterable[Tuple["array.array", int]]:
        for s in self._val:
            yield self.patchify(s.flat), s.label

    def sample_triplet(self) -> Triplet:
        ready = self.triplet_ready_classes
        if len(ready) < 2:
            raise RuntimeError(
                f"Need >=2 classes with >=2 train samples each, got {len(ready)}"
            )
        a_class = ready[self._rng.randrange(len(ready))]
        n_class = a_class
        while n_class == a_class:
            n_class = ready[self._rng.randrange(len(ready))]
        a_idx_list = self._train_by_class[a_class]
        i1 = self._rng.randrange(len(a_idx_list))
        i2 = i1
        while i2 == i1:
            i2 = self._rng.randrange(len(a_idx_list))
        n_idx_list = self._train_by_class[n_class]
        i3 = self._rng.randrange(len(n_idx_list))
        return Triplet(
            anchor=self.patchify(self._train[a_idx_list[i1]].flat),
            positive=self.patchify(self._train[a_idx_list[i2]].flat),
            negative=self.patchify(self._train[n_idx_list[i3]].flat),
        )

    def sample_triplets(self, n: int) -> List[Triplet]:
        return [self.sample_triplet() for _ in range(n)]
