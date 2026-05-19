"""Train the standard (softmax-attention) GPT-style Transformer
on a tiny copy-shift task to verify end-to-end gradients flow.

Run from the repo root:

    PYTHONPATH=python python3 python/examples/train_standard_transformer.py
"""

from __future__ import annotations

from dart_cuda.core.tensor.gpu_tensor import Tensor
from dart_cuda.core.transformers.standard import Transformer


def main() -> None:
    vocab = 16
    seq = [3, 5, 7, 9, 4, 6, 8, 2]
    tgt = [5, 7, 9, 4, 6, 8, 2, 0]

    model = Transformer(
        vocab_size=vocab,
        embed_size=16,
        block_size=len(seq),
        num_layers=2,
        num_heads=4,
    )

    lr = 0.05
    for step in range(80):
        tracker = []
        logits = model.forward(seq, tracker)
        loss = logits.cross_entropy(tgt)
        loss_val = loss.data[0]
        loss.backward()
        model.step(lr)
        model.zero_grad()
        if step % 10 == 0 or step == 79:
            print(f"step {step:3d}  loss={loss_val:.4f}")
        for t in tracker:
            t.dispose()
        logits.dispose()
        loss.dispose()


if __name__ == "__main__":
    main()
