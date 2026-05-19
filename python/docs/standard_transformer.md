# Standard (softmax) transformer modules

The `dart_cuda.core.transformers.standard` package mirrors the classic
post-2017 Transformer family (multi-head **softmax** attention), in
contrast to the `dart_cuda.core.transformers.aft` package which uses
Apple's Attention-Free Transformer.

Both packages share the same API conventions: `Module` subclasses with
`forward(...)` and `parameters()`, plus a `tracker` list that collects
every intermediate `Tensor` so callers can `dispose()` them after the
backward pass.

## Building blocks

| Class | File | Purpose |
|---|---|---|
| `SelfAttention` | `core/attention/self_attention.py` | Single softmax-attention head (optionally causal) |
| `CrossAttention` | `core/attention/cross_attention.py` | Single-head attention with Q from decoder, K/V from encoder |
| `MultiHeadAttention` | `core/attention/multi_head_attention.py` | Stacks `SelfAttention` heads + output projection |
| `MultiHeadCrossAttention` | `core/attention/multi_head_cross_attention.py` | Stacks `CrossAttention` heads + projection |
| `LayerNorm` | `core/layers/layer_norm.py` | Standard layer normalisation (existing) |
| `RMSNorm` | `core/layers/rms_norm.py` | LLaMA/DeepSeek RMS norm |
| `FeedForward` | `core/layers/feed_forward.py` | Two-layer GELU FFN (existing) |
| `MoEFeedForward` | `core/layers/moe_feed_forward.py` | DeepSeek-style top-K sigmoid-gated MoE (ReLU experts) |
| `SwiGLUMoEFeedForward` | `core/layers/swiglu_moe_feed_forward.py` | Top-K MoE with SwiGLU experts |

## Composed transformer modules

| Class | Purpose |
|---|---|
| `TransformerBlock` | One pre-norm block (MHA + FFN), optional causal mask |
| `TransformerEncoderBlock` | Pre-norm block with bi-directional self-attention |
| `TransformerDecoderBlock` | Pre-norm block with masked self-attn + cross-attn + FFN |
| `Transformer` | GPT-style decoder-only transformer with LM head |
| `TransformerEncoder` | Encoder stack with token+position embeddings |
| `TransformerDecoder` | Decoder stack that attends to encoder output |
| `EncoderDecoderTransformer` | Encoder + decoder wired together |
| `MTPModule` | DeepSeek-V3 Multi-Token Prediction head |

All of the above live in `dart_cuda.core.transformers.standard` and can
be imported from the package's `__init__`:

```python
from dart_cuda.core.transformers.standard import (
    TransformerBlock,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    EncoderDecoderTransformer,
    MTPModule,
)
```

## Scaled dot-product attention

The shared SDPA helper lives in `self_attention.py`:

```python
from dart_cuda.core.attention.self_attention import scaled_dot_product_attention

out = scaled_dot_product_attention(q, k, v, tracker, masked=True)
```

It composes existing tensor ops (transpose, matmul, scalar division,
softmax) so all gradient flow happens through the standard autograd
graph — no fused CUDA kernel is required beyond the new `transpose`
op added in this release.

A causal mask of shape `[T, T]` filled with `-1e9` above the diagonal
is constructed lazily by `_causal_mask(T)` and added before the
softmax when `masked=True`.

## Native engine additions

To support these modules two native primitives were added:

* `transpose_tensor(a)` — tile-based 2D transpose with an
  `atomicAdd`-style backward that accumulates `grad_in += grad_out.T`.
* A proper backward for `softmax_forward` — earlier versions left
  the gradient at zero because softmax was only used inside the
  Hungarian cost matrix path. The new backward implements the
  standard row-wise formula
  `dL/dx_i = y_i * (gy_i - sum_j y_j gy_j)`.

Both are exercised by the standard attention path and any existing
code that called `Tensor.softmax()` now actually receives a gradient.

## Minimal end-to-end example

```python
from dart_cuda.core.tensor.gpu_tensor import Tensor
from dart_cuda.core.transformers.standard import Transformer

vocab = 16
seq = [3, 5, 7, 9, 4, 6, 8, 2]
tgt = [5, 7, 9, 4, 6, 8, 2, 0]

model = Transformer(vocab_size=vocab, embed_size=16, block_size=len(seq),
                    num_layers=2, num_heads=4)

for step in range(60):
    tracker = []
    logits = model.forward(seq, tracker)
    loss = logits.cross_entropy(tgt)
    loss.backward()
    model.step(0.05)
    model.zero_grad()
    for t in tracker:
        t.dispose()
    logits.dispose()
    loss.dispose()
```

A full runnable version sits at
[`examples/train_standard_transformer.py`](../examples/train_standard_transformer.py).

## Encoder-decoder example

```python
from dart_cuda.core.transformers.standard import EncoderDecoderTransformer

model = EncoderDecoderTransformer(
    source_vocab_size=20,
    target_vocab_size=15,
    embed_size=8,
    source_block_size=6,
    target_block_size=4,
    num_layers=2,
    num_heads=2,
)
tracker = []
logits = model.forward([1, 2, 3, 4, 5], [1, 2, 3], tracker)
loss = logits.cross_entropy([2, 3, 0])
loss.backward()
```

## MTP (Multi-Token Prediction)

```python
from dart_cuda.core.transformers.standard import MTPModule

mtp = MTPModule(embed_size=16, num_heads=4, encoder_embed_size=16)
tracker = []
hidden = mtp.forward(prev_hidden, shifted_embeds, encoder_out, tracker)
# Apply the shared LM head from your main trunk:
# logits = main_model.lm_head.forward(hidden, tracker)
```
