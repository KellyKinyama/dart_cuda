# Layers

Every layer subclasses `dart_cuda.core.layers.nn.Module` and exposes:

- `forward(x, tracker, ...)` — returns one or more `Tensor`s, **appends every
  intermediate** to `tracker` so the training loop can dispose them.
- `parameters() -> list[Tensor]` — flat list of trainable parameters.
- `zero_grad()` / `step(lr)` — defaults provided by `Module`.

## `Layer` — dense linear (+ optional GELU)

```python
from dart_cuda.core.layers.nn import Layer

l = Layer(nin=32, nout=64, useGelu=True)    # Xavier init
tracker: list[dc.Tensor] = []
y = l.forward(x, tracker)
```

## `MLP` — chained `Layer`s

```python
from dart_cuda.core.layers.mlp import MLP

mlp = MLP(nin=2, nouts=[16, 16, 1])         # GELU between hidden layers
y = mlp.forward(x, tracker)
```

## `LayerNorm`

```python
from dart_cuda.core.layers.layer_norm import LayerNorm

ln = LayerNorm(dim=64, eps=1e-5)
y = ln.forward(x, tracker)
```

## `FeedForward` — transformer FFN block

```python
from dart_cuda.core.layers.feed_forward import FeedForward

ff = FeedForward(dim=128)        # 128 -> 512 (GELU) -> 128
y = ff.forward(x, tracker)
```

## `Conv2d` — im2col + matmul

```python
from dart_cuda.core.layers.conv_2d import Conv2d

conv = Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
)
y = conv.forward(x, tracker)
```

## Writing your own `Module`

```python
from dart_cuda.core.layers.nn import Layer, Module
import dart_cuda as dc

class TwoLayerNet(Module):
    def __init__(self, nin, nhidden, nout):
        self.l1 = Layer(nin, nhidden, useGelu=True)
        self.l2 = Layer(nhidden, nout, useGelu=False)

    def forward(self, x, tracker):
        h = self.l1.forward(x, tracker)
        return self.l2.forward(h, tracker)

    def parameters(self):
        return [*self.l1.parameters(), *self.l2.parameters()]
```
