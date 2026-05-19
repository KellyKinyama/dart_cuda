# Persistence — Dart ⇄ Python checkpoints

Checkpoints are flat little-endian `float32` blobs: each parameter, in the
order returned by `module.parameters()`, is written end-to-end. **The file
format is identical to the Dart `saveModuleBinary`**, so a checkpoint trained
in Dart loads in Python and vice versa.

## Save

```python
from dart_cuda.core.utils.persistence import save_module_binary

save_module_binary(model, "checkpoints/model.bin")
```

## Load

```python
from dart_cuda.core.utils.persistence import load_module_binary

load_module_binary(model, "checkpoints/model.bin")
```

The loader validates that `len(file) / 4 == sum(p.length for p in parameters)`
and raises `ValueError` on mismatch — the most common cause is mismatched
architecture hyperparameters between save and load.

## Dart ⇄ Python interop

Because both sides use the same `getTensorData` / `setTensorData` host↔device
copies and the same parameter ordering, this works:

```dart
// Dart
await saveModuleBinary(model, 'model.bin');
```

```python
# Python — load the file produced by Dart
from dart_cuda.core.utils.persistence import load_module_binary
load_module_binary(model_py, "model.bin")
```

…as long as `model.parameters()` returns parameters in the same order on both
sides. The Python ports preserve the Dart ordering (e.g. `Layer` returns
`[weight, bias]`; `LayerNorm` returns `[gamma, beta]`).

## Aliases

```python
from dart_cuda.core.utils.persistence import saveModuleBinary, loadModuleBinary
from dart_cuda.core.utils.network_utils import save_module_binary  # re-export
```
