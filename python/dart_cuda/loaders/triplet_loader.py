"""Triplet loader — mirrors `lib/loaders/triplet_loader.dart`.

Requires Pillow.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

from ..core.tensor.gpu_tensor import Tensor


def _require_pil():
    try:
        from PIL import Image  # noqa: F401

        return Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required: pip install Pillow") from exc


class TripletLoader:
    def __init__(self, root_path: str, image_size: int):
        self.root_path = root_path
        self.image_size = image_size
        self._identity_map: Dict[str, List[str]] = {}
        self._random = random.Random()
        self._scan_dataset()

    def _scan_dataset(self) -> None:
        for name in os.listdir(self.root_path):
            full = os.path.join(self.root_path, name)
            if os.path.isdir(full):
                files = [
                    os.path.join(full, f)
                    for f in os.listdir(full)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if files:
                    self._identity_map[name] = files

    def _process_image(self, file_path: str) -> List[float]:
        Image = _require_pil()
        img = (
            Image.open(file_path)
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )
        out: List[float] = []
        for r, g, b in img.getdata():
            out.append(r / 255.0)
            out.append(g / 255.0)
            out.append(b / 255.0)
        return out

    def next_batch(self, batch_size: int) -> Dict[str, Tensor]:
        anchors: List[float] = []
        positives: List[float] = []
        negatives: List[float] = []
        people = list(self._identity_map.keys())
        if len(people) < 2:
            raise RuntimeError("Need at least 2 identities for triplets")

        for _ in range(batch_size):
            person_a = people[self._random.randrange(len(people))]
            images_a = self._identity_map[person_a]
            i1 = self._random.randrange(len(images_a))
            i2 = self._random.randrange(len(images_a))
            if len(images_a) > 1:
                while i1 == i2:
                    i2 = self._random.randrange(len(images_a))
            person_b = person_a
            while person_b == person_a:
                person_b = people[self._random.randrange(len(people))]
            images_b = self._identity_map[person_b]

            anchors.extend(self._process_image(images_a[i1]))
            positives.extend(self._process_image(images_a[i2]))
            negatives.extend(
                self._process_image(images_b[self._random.randrange(len(images_b))])
            )

        feat = self.image_size * self.image_size * 3
        return {
            "anchor": Tensor.from_list([batch_size, feat], anchors),
            "positive": Tensor.from_list([batch_size, feat], positives),
            "negative": Tensor.from_list([batch_size, feat], negatives),
        }

    nextBatch = next_batch
