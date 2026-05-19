"""End-to-end training smoke test for the standard Transformer (GPT)."""

from __future__ import annotations

import pytest

from dart_cuda.core.tensor.gpu_tensor import Tensor
from dart_cuda.core.transformers.standard import Transformer


def test_gpt_training_reduces_loss():
    vocab = 8
    seq = [1, 2, 3, 4, 5]
    tgt = [2, 3, 4, 5, 0]

    model = Transformer(
        vocab_size=vocab, embed_size=8, block_size=len(seq),
        num_layers=1, num_heads=2,
    )

    losses = []
    for _ in range(30):
        tracker = []
        logits = model.forward(seq, tracker)
        loss = logits.cross_entropy(tgt)
        losses.append(float(loss.data[0]))
        model.zero_grad()
        loss.backward()
        model.step(0.1)
        for t in tracker:
            t.dispose()
        logits.dispose()
        loss.dispose()

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"

    for p in model.parameters():
        p.dispose()
