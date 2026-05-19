# Using PyTorch checkpoints in dart_cuda

PyTorch checkpoints are not a drop-in for dart_cuda, but you can convert them
in a few lines. There are three concerns: **file format**, **weight layout**,
and **operator semantics**.

## TL;DR

| Layer | Transferable? | Required transform |
|---|---|---|
| `nn.Linear` → `Layer` | ✅ | `weight.T` (transpose) |
| `nn.Embedding` → embedding matrix | ✅ | none |
| `nn.LayerNorm` → `LayerNorm` | ✅ | `weight`→gamma, `bias`→beta |
| `nn.Conv2d` → `Conv2d` | ✅ | reshape `[O, I, kH, kW]` → `[O, I*kH*kW]` |
| `nn.MultiheadAttention` → `MultiHeadAFT` | ⚠️ shape-only | weights load, but AFT ≠ softmax-attention, so numerics differ |
| Custom transformer blocks | ⚠️ depends | only the linear sub-layers transfer; positional bias is dart_cuda-specific |

## 1. File format

PyTorch saves a pickled `dict[str, Tensor]` (`torch.save(state_dict, ...)`)
that requires the `torch` package to read. dart_cuda saves a flat
little-endian `float32` blob in `module.parameters()` order
(see [persistence.md](persistence.md)).

You don't have to deserialize PyTorch → ours by hand — this package ships a
helper:

```python
import torch
from dart_cuda.core.utils.torch_interop import (
    list_module_parameters,
    apply_mapping,
    save_for_dart_cuda,
    transpose_linear,
    flatten_conv2d_weight,
)

sd = torch.load("model.pt", map_location="cpu")
print(list_module_parameters(model_dc))   # see what each slot expects
```

## 2. Layout differences

### Linear → `Layer`

PyTorch `nn.Linear.weight` has shape `[out, in]` because PyTorch computes
`y = x @ weight.T + bias`. dart_cuda's `Layer.w` is `[in, out]` because it
computes `y = x @ w + b`. **Always transpose**:

```python
mapping = {
    0: transpose_linear(sd["fc1.weight"]),    # Layer.w
    1: sd["fc1.bias"],                        # Layer.b
    2: transpose_linear(sd["fc2.weight"]),
    3: sd["fc2.bias"],
}
```

### Conv2d

PyTorch stores `[O, I, kH, kW]`; dart_cuda uses im2col + matmul with the
weight already flattened to `[O, I*kH*kW]`:

```python
mapping[k] = flatten_conv2d_weight(sd["conv.weight"])    # [O, I*kH*kW]
mapping[k + 1] = sd["conv.bias"]
```

### LayerNorm

`nn.LayerNorm` stores `weight` and `bias`; dart_cuda's `LayerNorm` exposes
the same vectors as `gamma` and `beta`. They're numerically interchangeable —
just route them to the right slot:

```python
mapping[k] = sd["ln.weight"]   # -> gamma
mapping[k + 1] = sd["ln.bias"] # -> beta
```

### Embedding

`nn.Embedding.weight` is `[num_embeddings, embedding_dim]`, identical to
dart_cuda's `wte` / `wpe`. **No transform needed.**

## 3. Operator semantics (the gotcha)

dart_cuda's transformer blocks use **AFT** (Attention-Free Transformer), not
standard softmax attention. The Q/K/V projections are ordinary linears whose
weights load fine, but the attention output is computed as

$$\text{out}_t = \sigma(Q_t) \odot \frac{\sum_{t'} \exp(K_{t'} + \text{posBias}_{t,t'}) \odot V_{t'}}{\sum_{t'} \exp(K_{t'} + \text{posBias}_{t,t'})}$$

…not the familiar $\text{softmax}(QK^\top / \sqrt{d}) V$. So **weights ported
from a softmax-attention transformer will warm-start (and may fine-tune
quickly) but won't reproduce the source model's outputs**. There is no
mathematically clean conversion in general.

If the source is itself an AFT or RWKV-style model, the projections + bias
table line up and the transfer is exact.

## 4. End-to-end example — MLP transfer

```python
import torch
import torch.nn as nn
from dart_cuda.core.layers.mlp import MLP
from dart_cuda.core.utils.torch_interop import (
    map_mlp_state_dict, save_for_dart_cuda,
)

# Train (or load) a PyTorch MLP.
torch_mlp = nn.Sequential(
    nn.Linear(2, 8), nn.GELU(),
    nn.Linear(8, 1),
)
# ... torch.save(torch_mlp.state_dict(), "torch_mlp.pt") ...

sd = torch_mlp.state_dict()

# Build a matching dart_cuda model.
dc_mlp = MLP(2, [8, 1])

# Helper that handles the transposes for you.
mapping = map_mlp_state_dict(
    dc_mlp, sd,
    weight_keys=["0.weight", "2.weight"],
    bias_keys=["0.bias",   "2.bias"],
)

save_for_dart_cuda(dc_mlp, mapping, "mlp.bin")
# Now `mlp.bin` is readable by load_module_binary in both Python and Dart.
```

## 5. End-to-end example — ViT-ish backbone

```python
from dart_cuda.core.utils.torch_interop import (
    apply_mapping, list_module_parameters, transpose_linear,
)

slots = list_module_parameters(model_dc)
for s in slots:
    print(s)
# {'index': 0, 'shape': [192, 128], 'length': 24576}   # patch_projection.w
# {'index': 1, 'shape': [1, 128],   'length': 128}     # patch_projection.b
# {'index': 2, 'shape': [1, 128],   'length': 128}     # cls_token
# ...

sd = torch.load("vit.pt", map_location="cpu")
mapping = {
    0: transpose_linear(sd["patch_embed.proj.weight"]),
    1: sd["patch_embed.proj.bias"],
    2: sd["cls_token"].view(1, -1),
    3: sd["pos_embed"].view(-1, sd["pos_embed"].shape[-1]),
    # ... attention/FFN linears: only the linear weights transfer cleanly ...
}
apply_mapping(model_dc, mapping)
```

## 6. Going the other direction (dart_cuda → PyTorch)

```python
import struct, torch
from collections import OrderedDict

with open("dart_cuda_model.bin", "rb") as f:
    floats = list(struct.unpack(f"<{ (raw := f.read()) and len(raw)//4 }f", raw))

# Walk dart_cuda parameter shapes and slice into named PyTorch tensors,
# applying transpose_linear in reverse where needed.
state_dict = OrderedDict()
offset = 0
for name, shape, transform in PARAM_SCHEMA:           # you define the schema
    n = 1
    for d in shape: n *= d
    t = torch.tensor(floats[offset:offset+n]).view(shape)
    if transform == "linear_T":
        t = t.t().contiguous()
    state_dict[name] = t
    offset += n
torch.save(state_dict, "model.pt")
```

## Tips

- Call `list_module_parameters(model_dc)` first — it tells you the slot
  index, shape, and float count for every parameter. That makes mapping
  errors easy to spot.
- The converter validates the float count per slot, so a wrong transpose
  raises immediately instead of silently producing garbage.
- Save your mapping in code (not as another binary): re-running the
  conversion is fast and you keep a record of how the weights were laid out.
