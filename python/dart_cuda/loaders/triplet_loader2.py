"""Triplet loader (RAM-cached float arrays) — mirrors
`lib/loaders/triplet_loader2.dart`.

Requires Pillow.
"""

from __future__ import annotations

import array
import os
import random
from typing import Dict, List


def _require_pil():
    try:
        from PIL import Image  # noqa: F401

        return Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required: pip install Pillow") from exc


class TripletLoader:
    def __init__(self, root_path: str, image_size: int, num_of_files: int):
        self.root_path = root_path
        self.image_size = image_size
        self._identity_map: Dict[str, List["array.array"]] = {}
        self._random = random.Random()
        self._scan_dataset(num_of_files)

    def _process_image(self, file_path: str) -> "array.array":
        Image = _require_pil()
        img = (
            Image.open(file_path)
            .convert("RGB")
            .resize((self.image_size, self.image_size))
        )
        buf = array.array("f")
        for r, g, b in img.getdata():
            buf.append(r / 255.0)
            buf.append(g / 255.0)
            buf.append(b / 255.0)
        return buf

    def _scan_dataset(self, num_of_files: int) -> None:
        if not os.path.isdir(self.root_path):
            return
        print("Caching images to RAM as float arrays...")
        scanned = 0
        for name in os.listdir(self.root_path):
            full = os.path.join(self.root_path, name)
            if os.path.isdir(full):
                processed: List["array.array"] = []
                for fname in os.listdir(full):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        processed.append(
                            self._process_image(os.path.join(full, fname))
                        )
                if processed:
                    self._identity_map[name] = processed
            if scanned > num_of_files:
                break
            scanned += 1
        print(f"Loaded {len(self._identity_map)} identities.")

    def next_batch(self, batch_size: int) -> Dict[str, "array.array"]:
        features = self.image_size * self.image_size * 3
        anchor = array.array("f", [0.0] * (batch_size * features))
        positive = array.array("f", [0.0] * (batch_size * features))
        negative = array.array("f", [0.0] * (batch_size * features))
        people = list(self._identity_map.keys())
        for i in range(batch_size):
            person_a = people[self._random.randrange(len(people))]
            images_a = self._identity_map[person_a]
            idx1 = self._random.randrange(len(images_a))
            idx2 = self._random.randrange(len(images_a))
            if len(images_a) > 1:
                while idx1 == idx2:
                    idx2 = self._random.randrange(len(images_a))
            person_b = person_a
            while person_b == person_a:
                person_b = people[self._random.randrange(len(people))]
            images_b = self._identity_map[person_b]
            idx3 = self._random.randrange(len(images_b))

            anchor[i * features : (i + 1) * features] = images_a[idx1]
            positive[i * features : (i + 1) * features] = images_a[idx2]
            negative[i * features : (i + 1) * features] = images_b[idx3]
        return {"anchor": anchor, "positive": positive, "negative": negative}

    nextBatch = next_batch
