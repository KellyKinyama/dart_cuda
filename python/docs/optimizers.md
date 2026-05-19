# Optimizers

Both optimizers maintain Adam-style first- and second-moment buffers on the
GPU (`m`, `v`), and call the corresponding native kernel.

## `Adam`

```python
from dart_cuda.core.optimizers.adam import Adam

opt = Adam(
    model.parameters(),
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    gradClip=1.0,
)

for step in range(num_steps):
    tracker: list[dc.Tensor] = []
    pred = model.forward(x, tracker)
    loss = pred.cross_entropy(target)
    tracker.append(loss)

    opt.zero_grad()
    loss.backward()
    opt.step()           # NaN-checks param[0] internally; raises on blow-up

    for t in tracker:
        t.dispose()

opt.dispose()            # frees m and v tensors
```

## `SDG`

Same constructor and same call surface as `Adam`; invokes the native
`sdgStep` kernel instead. Drop-in replacement.

```python
from dart_cuda.core.optimizers.stochastic_grad_desc import SDG

opt = SDG(model.parameters(), lr=1e-2)
```

## Notes

- `step()` raises `RuntimeError("parameter[i] became NaN")` if the first
  element of any parameter is NaN after the update. This catches divergence
  early without sweeping every element each step.
- `opt.tracker` accumulates references to the running `m`/`v` tensors for
  external tracking — typically you can ignore it.
- Always call `opt.dispose()` at the end of training to free the moment
  buffers (GPU memory).
