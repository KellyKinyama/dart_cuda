# dart_cuda Python test suite

These tests cover the Python bindings end-to-end: tensor construction,
host/device round-trip, dispose semantics, autograd, elementwise
ops, layers (LayerNorm, RMSNorm, FeedForward, MLP), attention
modules, MoE variants, the standard softmax-attention Transformer
family (encoder/decoder/MTP), and the Adam/SGD optimizers.

## Running

The tests use `pytest`-style assertions and fixtures. If `pytest`
is installed, just run:

```bash
PYTHONPATH=python python3 -m pytest python/tests -q
```

If `pytest` is *not* installed, an included stdlib-only runner does
the same work using a minimal pytest stub (`_compat.py`):

```bash
PYTHONPATH=python python3 python/tests/run_tests.py
PYTHONPATH=python python3 python/tests/run_tests.py test_layers test_attention
```

Either path requires:
* `native/lib/libmat_mul.so` (build with `nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu -Xcompiler -fPIC`).
* A CUDA-capable device.

If the shared library or device is unavailable, all tests are skipped
cleanly.

## Layout

| File | Coverage |
|------|----------|
| `test_tensor_basics.py` | Factories, fetch/data setters, **dispose semantics** (idempotency, views) |
| `test_tensor_autograd.py` | Sum, mean, sigmoid, matmul, pow, log, transpose, softmax fwd/bwd, normalize, cross-entropy |
| `test_tensor_ops.py` | Elementwise add/sub/mul/div, scalar broadcast, concat, slice, reshape views, embedding |
| `test_layers.py` | Linear `Layer`, `LayerNorm`, `RMSNorm`, `FeedForward`, `MLP` overfit smoke |
| `test_attention.py` | Causal mask, SDPA helper, `SelfAttention`, `CrossAttention`, `MultiHeadAttention`, `MultiHeadCrossAttention` |
| `test_transformers.py` | `TransformerBlock`/`Encoder`/`Decoder`, GPT, `EncoderDecoderTransformer`, `MTPModule` |
| `test_moe.py` | `MoEFeedForward`, `SwiGLUMoEFeedForward`, routing-bias update |
| `test_optimizers.py` | Adam, SGD reduce loss on tiny regression + dispose |
| `test_training_smoke.py` | End-to-end GPT loss-decrease check |
