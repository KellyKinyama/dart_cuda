"""End-to-end XOR smoke test for the Python bindings.

Mirrors the spirit of `example/train_xor.dart`. Trains a tiny MLP for a few
steps and prints the loss curve. Requires `native/lib/libmat_mul.so`.
"""

from __future__ import annotations

import dart_cuda as dc
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.optimizers.adam import Adam


def main() -> None:
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [0.0, 1.0, 1.0, 0.0]

    model = MLP(2, [8, 1])
    optimizer = Adam(model.parameters(), lr=1e-2)

    for step in range(200):
        total_loss = 0.0
        for inp, tgt in zip(inputs, targets):
            tracker: list[dc.Tensor] = []
            x = dc.Tensor.from_list([1, 2], inp)
            y = dc.Tensor.from_list([1, 1], [tgt])
            tracker.extend([x, y])

            pred = model.forward(x, tracker)
            loss = pred.mse_loss(y)
            tracker.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.fetch_data()[0]
            for t in tracker:
                t.dispose()

        if step % 20 == 0:
            print(f"step {step:3d}  loss={total_loss/len(inputs):.6f}")

    # Final predictions
    print("\nFinal predictions:")
    for inp, tgt in zip(inputs, targets):
        tracker: list[dc.Tensor] = []
        x = dc.Tensor.from_list([1, 2], inp)
        tracker.append(x)
        pred = model.forward(x, tracker)
        print(f"  {inp} -> {pred.fetch_data()[0]:+.3f}  (target {tgt})")
        for t in tracker:
            t.dispose()

    optimizer.dispose()


if __name__ == "__main__":
    main()
