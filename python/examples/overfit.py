"""Tiny MLP overfitting a single point — fastest possible sanity check.

Mirrors `example/overfit.dart`.
"""

from __future__ import annotations

import dart_cuda as dc
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.optimizers.adam import Adam


def main() -> None:
    model = MLP(2, [4, 1])
    opt = Adam(model.parameters(), lr=5e-2)

    for step in range(100):
        tracker: list[dc.Tensor] = []
        x = dc.Tensor.from_list([1, 2], [0.5, -0.25])
        y = dc.Tensor.from_list([1, 1], [0.8])
        tracker += [x, y]

        pred = model.forward(x, tracker)
        loss = pred.mse_loss(y)
        tracker.append(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step:3d}  pred={pred.fetch_data()[0]:+.5f}  "
                  f"loss={loss.fetch_data()[0]:.6f}")

        for t in tracker:
            t.dispose()

    opt.dispose()


if __name__ == "__main__":
    main()
