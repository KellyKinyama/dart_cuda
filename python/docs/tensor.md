# Tensor

`dart_cuda.Tensor` is the only object that touches GPU memory. Every operation
allocates one or more new tensors on the device. Shapes are logically rank-N
but the engine treats them as `rows × cols` where `rows = shape[0]` and
`cols = prod(shape[1:])`.

## Factories

```python
import dart_cuda as dc

t0 = dc.Tensor.fill([3, 4], 1.0)              # constant fill
t1 = dc.Tensor.zeros([3, 4])                  # zeros
t2 = dc.Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])
t3 = dc.Tensor.random([4, 4], scale=0.02)     # uniform[-scale, scale]
```

Dart-style aliases also work: `Tensor.fromList(...)`.

## Element access

```python
t = dc.Tensor.from_list([2, 3], [1, 2, 3, 4, 5, 6])

t.data                # list[float] copied off the device
t.fetch_data()        # same as t.data
t.fetch_row(0)        # [1.0, 2.0, 3.0]
t.grad                # accumulated gradient (list[float])

t.data = [7, 8, 9, 10, 11, 12]   # host -> device copy
```

## Arithmetic and broadcasts

```python
a = dc.Tensor.from_list([2, 2], [1, 2, 3, 4])
b = dc.Tensor.from_list([2, 2], [5, 6, 7, 8])

c = a + b           # tensor + tensor
d = a - 0.5         # tensor - scalar (broadcast)
e = 2.0 * a         # scalar * tensor
f = a / b           # element-wise
g = -a              # unary negation
m = a.matmul(b)     # matrix multiply
```

## Activations and reductions

```python
t.relu(); t.gelu(); t.sigmoid(); t.tanh()
t.log();  t.abs();  t.softmax()
t.pow(2.0)
t.sum()     # scalar tensor
t.mean()    # scalar tensor
```

## Shape ops (views vs new tensors)

```python
v = t.reshape([3, 2])            # VIEW — shares storage with t
s = t.slice(0, 1)                # VIEW — first row
r = t.get_row(0)                 # new tensor (copy)
n = t.normalize(eps=1e-10)       # L2 normalize, new tensor
```

Views (`reshape`, `slice`) have `is_view=True` and **must not** be `dispose()`'d.

## Losses

```python
logits = model.forward(x, tracker)
ce  = logits.cross_entropy(target_idx)     # int target index
mse = pred.mse_loss(target_tensor)         # tensor target
```

## Autograd lifecycle

```python
tracker: list[dc.Tensor] = []

pred = model.forward(x, tracker)
loss = pred.cross_entropy(target)
tracker.append(loss)

optimizer.zero_grad()
loss.backward()        # populates .grad on every parameter
optimizer.step()

for t in tracker:
    t.dispose()        # free intermediates
```

See [memory_model.md](memory_model.md) for the full rules.

## Static helpers

```python
dc.Tensor.aft(q, k, v, pos_bias, masked=False)
dc.Tensor.aft_cross(q, k, v, pos_bias)
dc.Tensor.concat([t1, t2, t3])              # along feature axis
dc.Tensor.concat_axis0([t1, t2])            # along sequence axis
dc.Tensor.layer_norm(x, gamma, beta, eps=1e-5)
dc.Tensor.embeddings(token_ids, wte, wpe)
dc.Tensor.l2_normalize(t, eps=1e-10)
```
