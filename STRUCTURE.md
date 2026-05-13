# Project structure

The Dart sources are organised into three top-level groups.

```
lib/
├── core/                 # The reusable deep-learning library
│   ├── tensor/           # GPU tensor + low-level engine bindings
│   │   ├── gpu_tensor.dart
│   │   ├── engine.dart
│   │   ├── tensor.dart
│   │   ├── matrix.dart
│   │   └── mat_mul.dart
│   ├── layers/           # Building-block layers
│   │   ├── nn.dart            # base Layer / Module
│   │   ├── layer_norm.dart
│   │   ├── feed_forward.dart
│   │   ├── conv_2d.dart
│   │   └── mlp{,2,3}.dart
│   ├── attention/        # AFT attention primitives
│   │   ├── aft.dart
│   │   ├── aft_cross_attention.dart
│   │   ├── aft_multi_head_attention.dart
│   │   └── aft_multi_head_cross_attention.dart
│   ├── transformers/     # Transformer architectures, grouped by family
│   │   ├── aft/          # Pure AFT encoder/decoder + blocks
│   │   │   ├── transformer_{encoder,decoder}{,_block}.dart
│   │   │   ├── text_decoder_block.dart
│   │   │   └── muzero_transformer_decoder.dart
│   │   ├── deepseek/     # DeepSeek-style MoE decoder (built on AFT)
│   │   │   └── deepseek_aft_decoder.dart
│   │   ├── vision/       # ViT backbones & vision heads
│   │   │   ├── vit_backbone.dart
│   │   │   ├── vit_face_embedding.dart
│   │   │   └── vit_object_detector.dart
│   │   └── modalities/   # Modality-specific wrappers (built on AFT)
│   │       ├── text_decoder.dart
│   │       ├── text_transformer.dart
│   │       ├── audio_transformer.dart
│   │       ├── video_transformer.dart
│   │       ├── multi_modal_transformer{,2}.dart
│   │       └── multi_modal_transformer_encoder.dart
│   ├── models/           # Higher-level task models (non-transformer logic)
│   │   ├── mu_zero/      # MuZero agent, chess player, MCTS, training
│   │   └── chess/        # Chess-specific MCTS + UCI helpers
│   ├── optimizers/       # adam, sgd, cross_entropy
│   └── utils/            # network_utils, persistence, triplet_loss,
│                         # hungarian_algorithm, open_cv
│
├── loaders/              # Dataset / file loaders
│   ├── dataset.dart
│   ├── chess.dart        # PGN dataset
│   ├── images.dart       # image -> tensor
│   └── triplet_loader{,2}.dart
│
└── dart_cuda.dart        # public package entry-point

example/                  # Runnable demos & training scripts
├── bin/                  # cli entry-points (`dart run example/bin/...`)
├── mu_zero/              # MuZero demos (incl. UCI engine)
├── train_xor{,_2,_3}.dart
├── overfit.dart
├── audio_video.dart
├── chess_gpt.dart
├── face_{embeddings,training,…}.dart
├── main_face_gpu.dart
├── mlp_learn.dart
└── …

tool/                     # Build / training orchestration scripts
test/                     # `dart test` suite
test/scratch/             # ad-hoc tensor sanity checks
```

## Import conventions

* Library code under `lib/` should import via `package:dart_cuda/...`
  using the new paths, e.g.
  `import 'package:dart_cuda/core/tensor/gpu_tensor.dart';`
* `example/` and `test/` are not part of the published package, so they
  use `package:dart_cuda/...` for library code and **relative** imports
  for sibling files within the same example/test directory.

## Where to find things

| Looking for…                    | Path |
| ------------------------------- | ---- |
| GPU tensor / autograd           | `lib/core/tensor/gpu_tensor.dart` |
| Adam optimizer                  | `lib/core/optimizers/adam.dart` |
| Base `Layer` / `Module` class   | `lib/core/layers/nn.dart` |
| AFT attention                   | `lib/core/attention/` |
| Transformer encoder/decoder     | `lib/core/transformers/aft/` |
| DeepSeek MoE decoder            | `lib/core/transformers/deepseek/` |
| ViT backbones / vision heads    | `lib/core/transformers/vision/` |
| Text/audio/video/multi-modal    | `lib/core/transformers/modalities/` |
| MuZero chess agent              | `lib/core/models/mu_zero/` |
| Image / PGN dataset loaders     | `lib/loaders/` |
| Train-from-scratch demos        | `example/` and `example/bin/` |
| MuZero UCI engine entry-point   | `example/mu_zero/muzero_chess_uci.dart` |
