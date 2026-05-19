# Getting started

## Requirements

- **CUDA toolkit** with `nvcc` (the same one used to build the Dart engine).
- **Python ≥ 3.9** (standard library only — `ctypes`).
- *Optional:* Pillow, if you want the image loaders.

## 1. Build the native engine

From the repository root:

```bash
nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC
```

This produces `native/lib/libmat_mul.so`, the single shared object that both
Dart and Python wrap.

## 2. Install the Python package

```bash
cd python
pip install -e .

# Optional: enable PIL-based image loaders
pip install -e ".[images]"
```

## 3. Locate the library at runtime

The loader searches, in order:

1. `$DART_CUDA_LIB` — absolute path override.
2. `./native/lib/libmat_mul.so` (relative to the current working directory).
3. Walking up from the package source until `native/lib/libmat_mul.so` is found.

If none match, importing `dart_cuda` raises `FileNotFoundError` listing every
path tried.

```bash
export DART_CUDA_LIB=/abs/path/to/libmat_mul.so   # if running from elsewhere
```

## 4. First program

```python
import dart_cuda as dc

a = dc.Tensor.from_list([2, 2], [1, 2, 3, 4])
b = dc.Tensor.from_list([2, 2], [5, 6, 7, 8])
c = a.matmul(b)
print(c.fetch_data())        # [19.0, 22.0, 43.0, 50.0]

# Free GPU memory explicitly (see docs/memory_model.md).
a.dispose(); b.dispose(); c.dispose()
```

## 5. Run the XOR smoke test

```bash
cd python
PYTHONPATH=. python3 examples/train_xor.py
```

Expected output (truncated):

```
step   0  loss=0.439754
step  20  loss=0.251786
...
step 180  loss=0.000000

Final predictions:
  [0.0, 0.0] -> +0.000  (target 0.0)
  [0.0, 1.0] -> +1.000  (target 1.0)
  [1.0, 0.0] -> +1.000  (target 1.0)
  [1.0, 1.0] -> +0.000  (target 0.0)
```

If that runs, the FFI, autograd, and Adam optimizer are all wired correctly.
