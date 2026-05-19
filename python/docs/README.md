# dart_cuda Python documentation

| Topic | Page |
|---|---|
| [Getting started](getting_started.md) | Build the `.so`, install the package, run the first program |
| [Tensor API](tensor.md) | `Tensor` factories, ops, autograd, lifecycle |
| [Layers](layers.md) | `Module`, `Layer`, `MLP`, `Conv2d`, `LayerNorm`, `FeedForward` |
| [Attention](attention.md) | AFT self / cross / multi-head variants |
| [Transformers](transformers.md) | Encoder, decoder, ViT, modalities, multimodal fusion |
| [Optimizers](optimizers.md) | `Adam`, `SDG` |
| [Persistence](persistence.md) | Save / load checkpoints, Dart ⇄ Python interop |
| [Loaders](loaders.md) | `images`, `ImageFolderLoader`, triplet loaders |
| [Memory model](memory_model.md) | Why `dispose()` matters, the `tracker` pattern, views |
| [Dart ↔ Python parity](parity.md) | snake_case / camelCase aliases, naming conventions |
| [Examples](../examples/) | Runnable scripts (`train_xor.py`, etc.) |

See [`../README.md`](../README.md) for a high-level overview.
