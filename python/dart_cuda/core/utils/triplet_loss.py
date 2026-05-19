"""Triplet loss — mirrors `lib/core/utils/triplet_loss.dart`."""

from __future__ import annotations

from typing import List

from ..layers.nn import Module
from ..tensor.gpu_tensor import Tensor


class TripletLossGPU(Module):
    def __init__(self, *, margin: float = 0.2):
        self.margin = margin

    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
        tracker: List[Tensor],
    ) -> Tensor:
        diff_pos = anchor - positive
        diff_neg = anchor - negative
        dist_p = diff_pos.pow(2.0).sum()
        dist_n = diff_neg.pow(2.0).sum()
        raw_loss = (dist_p - dist_n) + self.margin
        loss = raw_loss.relu()
        tracker.extend([diff_pos, diff_neg, dist_p, dist_n, raw_loss, loss])
        return loss

    def parameters(self) -> List[Tensor]:
        return []
