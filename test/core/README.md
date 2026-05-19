# Dart core test suite

Run all tests from the repo root after building `native/lib/libmat_mul.so`:

```
dart pub get
dart test test/core/
```

Run a single file:

```
dart test test/core/tensor_basics_test.dart
```

## Coverage

| File | What it exercises |
| --- | --- |
| `tensor_basics_test.dart` | Factories (`fromList`, `fill`, `zeros`), `data` / `fetchData` / `fetchRow`, `reshape` view semantics, `getRow`, `slice`, **`dispose()` idempotency**, view-dispose-is-noop. |
| `tensor_autograd_test.dart` | Analytic gradient checks: `sum`, `mean`, `sigmoid`, `pow`, `log`, `relu`, `matmul`, `softmax` (forward + sum-grad-zero), `crossEntropy` (label-smoothed), `normalize`. |
| `tensor_ops_test.dart` | Element-wise `+ - * /` (Tensor⊕Tensor and Tensor⊕scalar), `Tensor.concat`, `Tensor.concatAxis0`, `Tensor.embeddings`. |
| `layers_test.dart` | `Layer`, `LayerNorm`, `FeedForward`, `MLP` (parameter layout, forward shape, backward populates grads, tiny-regression loss decreases). |
| `aft_test.dart` | `AFTAttention`, `AFTCrossAttention`, `MultiHeadAFT`, `MultiHeadAFTCross` (forward shapes + backward smoke). |
| `optimizers_test.dart` | `Adam` and `SDG` reduce loss; **`Adam.dispose()` empties `m` / `v` buffers** (dispose contract: deletes underlying tensor pointers). |

## Dispose contract

All tests dispose every parameter and every tracked intermediate at the end
of the test (or after `backward()`). The underlying contract is:

* `Tensor.dispose()` calls `engine.destroyTensor(handle)` and sets the
  `_isDisposed` flag. Subsequent calls are a no-op.
* Views (e.g. the result of `reshape`) hold `isView == true` and skip the
  destroy call; disposing them is a no-op for the parent's handle.
* Optimizer `dispose()` frees the per-parameter moment buffers and clears
  the bookkeeping lists.

Tests verify each of these invariants explicitly.
