# dart_cuda (Python bindings)

Python `ctypes` bindings for the [`dart_cuda`](../) CUDA tensor / autograd
engine. The package mirrors the Dart `lib/` layout one-for-one so the same
native `libmat_mul.so` powers both languages.

## Quick links

- [Getting started](docs/getting_started.md) — build the `.so`, install, run XOR
- [Tensor API](docs/tensor.md) · [Layers](docs/layers.md) · [Attention](docs/attention.md) · [Transformers](docs/transformers.md)
- [Optimizers](docs/optimizers.md) · [Persistence](docs/persistence.md) · [Loaders](docs/loaders.md)
- [PyTorch interop](docs/pytorch_interop.md) — convert `.pt` checkpoints into dart_cuda format
- [Memory model](docs/memory_model.md) — required reading before writing a training loop
- [Dart ↔ Python parity](docs/parity.md) — naming, aliases, file mapping
- [Examples](examples/) — runnable scripts

## Install

```bash
# 1. Build the native engine (from repo root)
nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC

# 2. Install the package
cd python
pip install -e .

# Optional: PIL-based image loaders
pip install -e ".[images]"
```

The loader resolves `libmat_mul.so` via, in order:

1. `$DART_CUDA_LIB`
2. `./native/lib/libmat_mul.so`
3. Walking up from the package source until it finds `native/lib/libmat_mul.so`

## Hello tensor

```python
import dart_cuda as dc

a = dc.Tensor.from_list([2, 2], [1, 2, 3, 4])
b = dc.Tensor.from_list([2, 2], [5, 6, 7, 8])
c = a.matmul(b)
print(c.fetch_data())          # [19.0, 22.0, 43.0, 50.0]

a.dispose(); b.dispose(); c.dispose()
```

## Training loop pattern

Every `forward()` takes a `tracker: list[Tensor]` and appends every
intermediate so it can be freed at the end of the step. Parameters and views
are **never** disposed.

```python
import dart_cuda as dc
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.optimizers.adam import Adam

model = MLP(2, [8, 1])
opt   = Adam(model.parameters(), lr=1e-2)

for step in range(200):
    tracker: list[dc.Tensor] = []
    x = dc.Tensor.from_list([1, 2], [1.0, 0.0])
    y = dc.Tensor.from_list([1, 1], [1.0])
    tracker += [x, y]

    pred = model.forward(x, tracker)
    loss = pred.mse_loss(y)
    tracker.append(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()

    for t in tracker:
        t.dispose()

opt.dispose()
```

Full version in [examples/train_xor.py](examples/train_xor.py).

## Examples

| Script | What it shows |
|---|---|
| [examples/train_xor.py](examples/train_xor.py) | MLP + Adam end-to-end on XOR |
| [examples/overfit.py](examples/overfit.py) | Fastest sanity check: overfit one point |
| [examples/transformer_encoder_forward.py](examples/transformer_encoder_forward.py) | Token IDs → hidden states |
| [examples/save_load_checkpoint.py](examples/save_load_checkpoint.py) | Dart-compatible binary checkpoints |
| [examples/load_pytorch_checkpoint.py](examples/load_pytorch_checkpoint.py) | Convert a PyTorch MLP to dart_cuda (requires `torch`) |

Run with:

```bash
cd python
PYTHONPATH=. python3 examples/train_xor.py
```

## What's ported

| Area | Modules |
|---|---|
| `core/tensor` | `Tensor`, `CudaEngine` (full ctypes FFI) |
| `core/layers` | `Module`, `Layer`, `LayerNorm`, `FeedForward`, `MLP`, `Conv2d` |
| `core/attention` | `AFTAttention`, `AFTCrossAttention`, `MultiHeadAFT`, `MultiHeadAFTCross` |
| `core/transformers/aft` | encoder & decoder blocks, encoder, decoder, text decoder block |
| `core/transformers/vision` | `ViTBackbone`, `ViTFaceEmbeddingGPU`, `ViTObjectDetector` |
| `core/transformers/modalities` | text decoder, text/audio/video transformers, multimodal fusion |
| `core/optimizers` | `Adam`, `SDG` |
| `core/utils` | `persistence` (Dart-compatible checkpoints), `triplet_loss`, `hungarian_algorithm`, `network_utils` |
| `loaders` | `images`, `image_folder_loader`, `triplet_loader`, `triplet_loader2` |

## Not ported (deferred)

- `core/models/mu_zero/*` and `core/models/chess/*` — depend on the Dart
  `bishop` chess package.
- `core/transformers/deepseek/*` — can be added on request.
- `loaders/chess.dart`, `loaders/dataset.dart` — large embedded data.

## API parity

Public methods are exposed in both `snake_case` (Pythonic) and `camelCase`
(Dart-style) — `Tensor.fromList`, `Tensor.from_list`, `t.fetchData()`,
`t.fetch_data()`, etc. See [docs/parity.md](docs/parity.md) for the full
table.

## Memory model

GPU memory is owned by the C++ engine; Python's GC can't see into the
autograd graph. Every Python `Tensor` exposes `dispose()` and **you must
call it**, typically via the `tracker` pattern. Views (`reshape`,
`slice`) share storage with their parent and must not be disposed.

See [docs/memory_model.md](docs/memory_model.md) for the rules.
