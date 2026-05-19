# Memory model

The native C++ engine **owns** every byte of GPU memory. Python `Tensor`
objects are just opaque handles (`void*`) plus a shape. Python's garbage
collector cannot see into the autograd graph, so memory has to be freed by
hand.

## The three rules

1. **Every tensor you `from_list` / op-produce is yours to `dispose()`.**
   Forgetting leaks GPU memory until process exit.
2. **Never `dispose()` a view.** `reshape()` and `slice()` share their
   parent's storage (`is_view=True`); the parent must be the one freed.
3. **Disposed tensors raise on use.** Subsequent ops on a disposed handle
   produce undefined behaviour from the engine — guard with care.

## The `tracker` pattern

Mirrors the Dart codebase. Every `forward()` takes a `tracker: list[Tensor]`
and appends every intermediate it allocates. The training loop disposes the
whole tracker at the end of the step:

```python
tracker: list[dc.Tensor] = []

x = dc.Tensor.from_list([1, 2], inp);  tracker.append(x)
y = dc.Tensor.from_list([1, 1], tgt);  tracker.append(y)

pred = model.forward(x, tracker)         # internals append to tracker
loss = pred.mse_loss(y);                 tracker.append(loss)

optimizer.zero_grad()
loss.backward()
optimizer.step()

for t in tracker:
    t.dispose()
```

## What you should *not* dispose

- **Parameters** (`model.parameters()`) — kept alive for the lifetime of
  the model; the optimizer mutates them in place.
- **Views** (`reshape`, `slice` with `is_view=True`).
- **`m`/`v` buffers inside the optimizer** — call `optimizer.dispose()`
  at shutdown instead.

## Bound checks

`Adam.step()` (and `SDG.step()`) read the first element of every parameter
after the update and raise `RuntimeError("parameter[i] became NaN")` if it's
not finite. This is cheap (one float per param) and catches divergence early.

## Symptoms

| Symptom | Likely cause |
|---|---|
| Out-of-memory during long runs | Missing `dispose()` in training loop |
| Segfault on `.fetch_data()` | Operating on a disposed tensor |
| Double-free on shutdown | Disposed a view, or disposed a parameter |
| `RuntimeError: parameter[i] became NaN` | Diverging optimizer (lower `lr`, raise `gradClip`) |
