# Dart ↔ Python parity

Every Python file under `dart_cuda/` is a 1:1 port of the corresponding file
in `lib/`. Public methods are exposed in **both** `snake_case` (Pythonic) and
`camelCase` (Dart-style) so code can be moved back and forth with minimal
edits.

## File mapping

| Dart (`lib/...`) | Python (`dart_cuda/...`) |
|---|---|
| `core/tensor/cuda_engine.dart` | `core/tensor/cuda_engine.py` |
| `core/tensor/gpu_tensor.dart` | `core/tensor/gpu_tensor.py` |
| `core/layers/nn.dart` | `core/layers/nn.py` |
| `core/layers/layer_norm.dart` | `core/layers/layer_norm.py` |
| `core/layers/feed_forward.dart` | `core/layers/feed_forward.py` |
| `core/layers/mlp.dart` | `core/layers/mlp.py` |
| `core/layers/conv_2d.dart` | `core/layers/conv_2d.py` |
| `core/attention/aft.dart` | `core/attention/aft.py` |
| `core/attention/aft_cross_attention.dart` | `core/attention/aft_cross_attention.py` |
| `core/attention/aft_multi_head_attention.dart` | `core/attention/aft_multi_head_attention.py` |
| `core/attention/aft_multi_head_cross_attention.dart` | `core/attention/aft_multi_head_cross_attention.py` |
| `core/transformers/aft/*.dart` | `core/transformers/aft/*.py` |
| `core/transformers/vision/*.dart` | `core/transformers/vision/*.py` |
| `core/transformers/modalities/*.dart` | `core/transformers/modalities/*.py` |
| `core/optimizers/adam.dart` | `core/optimizers/adam.py` |
| `core/optimizers/stochastic_grad_desc.dart` | `core/optimizers/stochastic_grad_desc.py` |
| `core/utils/persistence.dart` | `core/utils/persistence.py` |
| `core/utils/network_utils.dart` | `core/utils/network_utils.py` |
| `core/utils/triplet_loss.dart` | `core/utils/triplet_loss.py` |
| `core/utils/hungarian_algorithm.dart` | `core/utils/hungarian_algorithm.py` |
| `loaders/images.dart` | `loaders/images.py` |
| `loaders/image_folder_loader.dart` | `loaders/image_folder_loader.py` |
| `loaders/triplet_loader.dart` | `loaders/triplet_loader.py` |
| `loaders/triplet_loader2.dart` | `loaders/triplet_loader2.py` |

## Naming aliases

| Dart (camelCase) | Python (snake_case) | Both work in Python |
|---|---|---|
| `Tensor.fromList` | `Tensor.from_list` | ✅ |
| `t.fetchData()` | `t.fetch_data()` | ✅ |
| `t.fetchRow(i)` | `t.fetch_row(i)` | ✅ |
| `t.zeroGrad()` | `t.zero_grad()` | ✅ |
| `t.printMatrix()` | `t.print_matrix()` | ✅ |
| `Tensor.concatAxis0` | `Tensor.concat_axis0` | ✅ |
| `Tensor.layerNorm` | `Tensor.layer_norm` | ✅ |
| `Tensor.l2Normalize` | `Tensor.l2_normalize` | ✅ |
| `Tensor.aftCross` | `Tensor.aft_cross` | ✅ |
| `t.crossEntropy` | `t.cross_entropy` | ✅ |
| `t.mseLoss` | `t.mse_loss` | ✅ |
| `t.computeCostMatrix` | `t.compute_cost_matrix` | ✅ |
| `AFTAttention.queryLayer` | `attn.query_layer` | ✅ |
| `ViTBackbone.patchProjection` | `vit.patch_projection` | ✅ |
| `face.getFaceEmbedding(...)` | `face.get_face_embedding(...)` | ✅ |
| `saveModuleBinary` | `save_module_binary` | ✅ |
| `loadModuleBinary` | `load_module_binary` | ✅ |

## Constructor argument style

Dart-style camelCase kwargs are preserved (`embedSize`, `numHeads`, `useGelu`,
`maxSeqLen`, `gradClip`, ...) because they appear all over the original code
and changing them would obscure the parity. Use them positionally or as kwargs
either way.

## Things that are not aliased

- `__init__` constructors use Dart's exact argument names — no snake_case
  alias is provided (they're typically called with kwargs anyway).
- Free-function utilities in `loaders/` use `snake_case` only.
