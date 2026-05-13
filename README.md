# 🚀 G-Tensor — High-Performance Dart & CUDA Deep Learning Engine

**G-Tensor** (`dart_cuda`) is a custom deep-learning framework that combines
the developer productivity of **Dart** with the raw computational power of
**NVIDIA CUDA**.

Unlike standard wrappers, G-Tensor features:

- A custom **Attention-Free Transformer (AFT)** implementation
- A manual **autograd** engine
- Hand-optimised CUDA kernels for causal masking, layer normalisation, and
  cross-entropy with label smoothing

---

## 🏗 System architecture

The engine is split into three layers:

1. **Dart API (frontend)** — high-level `Tensor` class with operator
   overloading (`+`, `-`, `*`, `matmul`) and `Module` classes for building
   neural networks.
2. **FFI bridge** — a low-level Dart FFI layer that handles memory
   addresses and dispatches calls to compiled C++/CUDA binaries.
3. **CUDA kernels (backend)** — hand-written `.cu` kernels optimised for
   parallel GPU execution, with custom broadcasting logic and stable
   gradient calculations.

---

## 🧠 The AFT causal mechanism

The core of this engine is the **Attention-Free Transformer**. Standard
multi-head attention has $O(T^2)$ complexity; AFT reduces this to $O(T d)$
by re-arranging the interaction between queries, keys and values.

### Formulation

$$
Z_t \;=\; \sigma(Q_t) \;\odot\;
\frac{\sum_{i=1}^{t} \exp(K_i + w_{t,i}) \odot V_i}
     {\sum_{i=1}^{t} \exp(K_i + w_{t,i})}
$$

Where:

- $\sigma$ is the sigmoid activation
- $\odot$ is the elementwise (Hadamard) product
- $w_{t,i}$ is a learned pairwise position bias

### Causal masking

To prevent the model from attending to future tokens, a triangular causal
mask is applied. In G-Tensor this is handled by a specialised
`engine.mulTensors` call during the forward pass, ensuring the gradients
from steps $t+1, \dots, T$ are exactly zero.

---

## ✨ Key features

- **Custom autograd** — fully functional backpropagation through the
  computational graph.
- **Explicit memory management** — `tracker` + `dispose` system to prevent
  VRAM leaks in Dart's garbage-collected environment.
- **Broadcasting** — adding row-vector biases to activation matrices via
  custom CUDA indexing (e.g. `[1, 128]` bias added to `[64, 128]`
  activations).
- **Advanced loss kernels** — numerically stable cross-entropy with built-in
  `LogSoftmax` and label smoothing ($\epsilon = 0.1$).

---

## 📁 Project layout

```
lib/                    Reusable library code
├── core/
│   ├── tensor/         GPU tensor + low-level engine bindings
│   ├── layers/         nn.dart, layer_norm, feed_forward, mlp, conv_2d
│   ├── attention/      AFT attention primitives
│   ├── transformers/
│   │   ├── aft/        Pure AFT encoder/decoder + blocks
│   │   ├── deepseek/   DeepSeek-MoE decoder
│   │   ├── vision/     ViT backbones, face embedding, object detection
│   │   └── modalities/ text/audio/video/multi-modal wrappers
│   ├── models/         MuZero agent, chess MCTS+UCI, …
│   ├── optimizers/     adam, sgd, cross_entropy
│   └── utils/          persistence, network_utils, triplet_loss, …
└── loaders/            Dataset / image / triplet loaders

native/
├── src/                CUDA sources (.cu, .h)
└── lib/                Compiled shared libraries (.so) — gitignored

example/                Runnable demos & training scripts
├── bin/                CLI entry-points (`dart run example/bin/...`)
├── mu_zero/            MuZero demos (incl. UCI engine)
├── tool/               Training / match orchestration scripts
└── tools/              Stockfish + helper scripts

test/                   `dart test` suite
```

See [STRUCTURE.md](STRUCTURE.md) for the full file-by-file map.

---

## 🛠 Installation & setup

### Prerequisites

- Dart SDK ≥ 3.0
- NVIDIA CUDA Toolkit ≥ 11.0 (`nvcc` on `PATH`)
- A CUDA-capable GPU

### Build the CUDA backend

The shared libraries live under `native/lib/`. From the repo root:

```bash
nvcc --shared -o native/lib/libmat_mul.so native/src/engine.cu    -Xcompiler -fPIC
nvcc --shared -o native/lib/dart_cuda.so  native/src/dart_cuda.cu -Xcompiler -fPIC
```

To rebuild against the v2 engine:

```bash
nvcc --shared -o native/lib/libmat_mul.so native/src/engine_v2.cu -Xcompiler -fPIC
```

The Dart FFI loader resolves these libraries relative to
`Directory.current.path`, so run all `dart` commands from the repo root.

### Install Dart dependencies

```bash
dart pub get
```

---

## 💻 Usage

### 1. Training with explicit memory management

Because Dart is garbage-collected but CUDA memory is not, intermediate
tensors must be tracked and freed each step:

```dart
for (int step = 0; step < 1000; step++) {
  final tracker = <Tensor>[];
  optimizer.zeroGrad();

  // Forward pass
  final logits = gpt.forward(inputIdx, dummyEnc, tracker);
  final loss   = logits.crossEntropy(targetIds);

  // Backward pass
  loss.backward();
  optimizer.step();

  // Free intermediate tensors to prevent CUDA OOM
  for (final t in tracker) {
    if (!gpt.parameters().contains(t)) t.dispose();
  }
  loss.dispose();
}
```

### 2. Autoregressive generation

The engine supports greedy and nucleus sampling for text generation:

```dart
void generate(String prompt) {
  final tokens = tokenizer.encode(prompt);
  final logits = gpt.forward(tokens, dummyEnc, <Tensor>[]);

  // Fetch only the last row for the next-token prediction
  final lastLogits = logits.fetchRow(tokens.length - 1);
  final nextToken  = sampleNucleus(lastLogits, temp: 0.8, topP: 0.9);
  print(tokenizer.decode([nextToken]));
}
```

---

## 🎮 Running the examples

```bash
# Train a tiny GPT on Tiny-Shakespeare
dart run example/bin/shakespear.dart

# XOR sanity check
dart run example/train_xor.dart

# Multi-modal (audio + video + text) demo
dart run example/bin/example_multi_modal.dart

# MuZero chess: train then play
dart run example/mu_zero/example.dart

# UCI engine (use in any chess GUI)
dart run example/mu_zero/muzero_chess_uci.dart --no-train
```

See [commands.md](commands.md) for the full set of MuZero / Stockfish
training recipes.

---

## 📊 Performance benchmarks

| Operation     | Input shape   | G-Tensor (CUDA) |
| ------------- | ------------- | --------------- |
| AFT forward   | `[64, 128]`   | ~0.9 ms         |
| Cross-entropy | `[64, 50257]` | ~1.1 ms         |
| LayerNorm     | `[64, 128]`   | ~0.3 ms         |

---

## 🤝 Roadmap

- **2D batching** — optimise `matmul` kernels for `[batch, seq, hidden]`.
- **KV-caching** — faster autoregressive inference.
- **RMSNorm** — Llama-style normalisation.
- **Linformer / vanilla attention** — alternative families alongside AFT
  (placeholders ready under `lib/core/transformers/`).

---

## 📜 License

See [LICENSE](LICENSE).

**Maintainer:** kkinyama  ·  **Project:** `dart_cuda`
