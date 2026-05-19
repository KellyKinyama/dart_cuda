"""Single-layer transformer encoder forward pass.

Tokens go in, hidden states come out. Useful for verifying the full
transformer stack compiles and runs end-to-end.
"""

from __future__ import annotations

import dart_cuda as dc
from dart_cuda.core.transformers.aft.transformer_encoder import TransformerEncoder


def main() -> None:
    encoder = TransformerEncoder(
        vocabSize=64,
        embedSize=32,
        blockSize=16,
        numLayers=2,
        numHeads=4,
    )

    tracker: list[dc.Tensor] = []
    tokens = [3, 7, 1, 9, 4, 2, 0, 5]            # T = 8
    out = encoder.forward(tokens, tracker)        # [T, embedSize]
    data = out.fetch_data()

    print("output shape :", out.shape)
    print("first row    :", data[: encoder.embed_size])
    print("|out|        :", sum(v * v for v in data) ** 0.5)

    for t in tracker:
        t.dispose()


if __name__ == "__main__":
    main()
