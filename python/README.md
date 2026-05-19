# dart_cuda (Python bindings)

Python `ctypes` bindings for the [`dart_cuda`](../) CUDA tensor / autograd
engine. The package mirrors the Dart `lib/` layout one-for-one so the same
native `libmat_mul.so` powers both languages.

## Build the native library

From the repo root:

```bash
nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC
```

The Python loader looks for the library in this order:

1. The `DART_CUDA_LIB` environment variable (absolute path).
2. `./native/lib/libmat_mul.so` relative to the current working directory.
3. Walking up from the package source until `native/lib/libmat_mul.so` is found.

## Install (editable)

```bash
cd python
pip install -e .
# Optional image-loader extras
pip install -e ".[images]"
```

## Quick start

```python
import dart_cuda as dc

a = dc.Tensor.from_list([2, 2], [1.0, 2.0, 3.0, 4.0])
b = dc.Tensor.from_list([2, 2], [5.0, 6.0, 7.0, 8.0])
c = a.matmul(b)
print(c.fetch_data())  # [19, 22, 43, 50]

a.dispose(); b.dispose(); c.dispose()
```

A complete MLP/XOR example lives in `examples/train_xor.py`.

## Memory model

The C++ autograd graph stores raw `Tensor*` pointers that Python's garbage
collector can't see, so every Python `Tensor` exposes an explicit `dispose()`.
Mirroring the Dart codebase, training loops typically push intermediates into a
`tracker: list[Tensor]` and call `dispose()` on each entry at the end of the
step.

Views (`reshape`, `slice` with `is_view=True`) share storage with their
parent and **must not** be disposed independently.

## API parity

Every module under `dart_cuda/` is a 1:1 port of the corresponding file in
`lib/`. Public methods are exposed in both `snake_case` (Pythonic) and
`camelCase` (Dart-style) for ease of porting code back and forth — for example:

- `Tensor.from_list(...)` ≡ `Tensor.fromList(...)`
- `tensor.fetch_data()` ≡ `tensor.fetchData()`
- `tensor.zero_grad()` ≡ `tensor.zeroGrad()`

## Status

Ported:

- `core/tensor` — full tensor type and engine FFI
- `core/layers` — `Module`, `Layer`, `LayerNorm`, `FeedForward`, `MLP`, `Conv2d`
- `core/attention` — AFT (self + cross + multi-head + multi-head cross)
- `core/transformers/aft` — encoder/decoder blocks, encoder, decoder, text decoder block
- `core/transformers/vision` — ViT backbone, face embedding head, object detector
- `core/transformers/modalities` — text decoder, text/audio/video transformers, multimodal fusion
- `core/optimizers` — Adam, SDG
- `core/utils` — persistence (Dart-compatible binary checkpoints), triplet loss, Hungarian algorithm
- `loaders` — `images`, `image_folder_loader`, `triplet_loader`, `triplet_loader2`

Deferred / not ported:

- `core/models/mu_zero/*` and `core/models/chess/*` (depend on the Dart `bishop` chess package)
- `core/transformers/deepseek/*` (can be added later)
- `loaders/chess.dart`, `loaders/dataset.dart` (large embedded data)
