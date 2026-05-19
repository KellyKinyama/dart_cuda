# Attention

All attention modules use the AFT (Attention-Free Transformer) kernel exposed
by the native engine: cheaper than standard QKᵀ-softmax attention while
keeping a learnable positional bias.

## `AFTAttention` — single-head self-attention

```python
from dart_cuda.core.attention.aft import AFTAttention

attn = AFTAttention(
    embedSize=128,
    headSize=128,
    maxSeqLen=64,
    masked=True,        # causal mask for decoders
)
y = attn.forward(x, tracker)

# Cross-attention via the `kv=` argument (used by text_decoder_block)
y = attn.forward(x_decoder, tracker, kv=encoder_output)
```

## `AFTCrossAttention` — separate decoder/encoder dims

```python
from dart_cuda.core.attention.aft_cross_attention import AFTCrossAttention

cross = AFTCrossAttention(
    decoderEmbedSize=128,
    encoderEmbedSize=256,
    headSize=128,
    maxTDec=64,
    maxTEnc=128,
)
y = cross.forward(x_dec, x_enc, tracker)
```

## `MultiHeadAFT` — N self-attention heads + projection

```python
from dart_cuda.core.attention.aft_multi_head_attention import MultiHeadAFT

mha = MultiHeadAFT(numHeads=4, embedSize=128, maxSeqLen=64, masked=False)
y = mha.forward(x, tracker)
```

`embedSize` must be divisible by `numHeads`. Each head receives the full
input, produces `embedSize / numHeads` features, then the heads are
concatenated and projected back to `embedSize`.

## `MultiHeadAFTCross` — N cross-attention heads + projection

```python
from dart_cuda.core.attention.aft_multi_head_cross_attention import MultiHeadAFTCross

mhca = MultiHeadAFTCross(
    numHeads=4,
    decoderEmbedSize=128,
    encoderEmbedSize=128,
    maxTDec=64,
    maxTEnc=64,
)
y = mhca.forward(x_dec, x_enc, tracker)
```

## Choosing masked vs unmasked

- **`masked=True`** — causal, used in decoders / autoregressive generation.
- **`masked=False`** — bidirectional, used in encoders (ViT, BERT-style).
