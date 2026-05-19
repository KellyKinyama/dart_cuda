# dart_cuda Python documentation

| Topic | Page |
|---|---|
| [Getting started](getting_started.md) | Build the `.so`, install the package, run the first program |
| [Tensor API](tensor.md) | `Tensor` factories, ops, autograd, lifecycle |
| [Layers](layers.md) | `Module`, `Layer`, `MLP`, `Conv2d`, `LayerNorm`, `FeedForward` |
| [Attention](attention.md) | AFT self / cross / multi-head variants |
| [Transformers](transformers.md) | AFT encoder, decoder, ViT, modalities, multimodal fusion |
| [Standard transformer](standard_transformer.md) | Softmax-attention encoder/decoder, GPT, MoE, MTP |
| [Optimizers](optimizers.md) | `Adam`, `SDG` |
| [Persistence](persistence.md) | Save / load checkpoints, Dart ⇄ Python interop |
| [PyTorch interop](pytorch_interop.md) | Convert `.pt` state-dicts into dart_cuda checkpoints |
| [Loaders](loaders.md) | `images`, `ImageFolderLoader`, triplet loaders |
| [Memory model](memory_model.md) | Why `dispose()` matters, the `tracker` pattern, views |
| [Dart ↔ Python parity](parity.md) | snake_case / camelCase aliases, naming conventions |
| [Examples](../examples/) | Runnable scripts (`train_xor.py`, etc.) |

See [`../README.md`](../README.md) for a high-level overview.
