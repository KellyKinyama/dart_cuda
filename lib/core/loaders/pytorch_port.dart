// Helpers to copy weights from a PyTorch `state_dict` (exported via
// `safetensors.torch.save_file`) into existing `Tensor` parameters.
//
// PyTorch's `nn.Linear` stores its weight as `[out, in]` (computes
// `x @ w.T`), while our `Layer` / attention / FF code stores weights
// as `[in, out]` (computes `x @ w`). The `transpose: true` flag on
// [PortSpec] handles that on the fly during copy.
//
// Typical use (after exporting `model.state_dict()` to `model.safetensors`):
//
//     final records = await loadSafetensorsFile('model.safetensors');
//     loadPytorchInto(records, {
//       // safetensors key                      our Tensor              transpose?
//       'wte.weight':           PortSpec(myModel.wte),
//       'blocks.0.attn.wq.weight': PortSpec(myModel.blocks[0].attn.wq, transpose: true),
//       'blocks.0.attn.wq.bias':   PortSpec(myModel.blocks[0].attn.bq),
//       ...
//     });

import 'dart:typed_data';

import 'package:dart_cuda/core/tensor/gpu_tensor.dart';
import 'package:dart_cuda/core/loaders/safetensors.dart';

/// Describes how a safetensors record should be copied into a target
/// `Tensor`. Set [transpose] when the source is a PyTorch
/// `nn.Linear.weight` (`[out, in]`) and the target is our column-major
/// matmul weight (`[in, out]`).
class PortSpec {
  final Tensor target;
  final bool transpose;
  const PortSpec(this.target, {this.transpose = false});
}

/// Copy values from `records` into the [PortSpec.target] tensors named by
/// the keys of [mapping]. Validates dtype / shape and reports every
/// mismatched or missing key in a single exception so the caller can fix
/// the whole mapping in one pass.
///
/// Any record present in `records` but absent from `mapping` is reported
/// as "unused" (a warning string) so the caller can audit coverage.
List<String> loadPytorchInto(
  Map<String, SafeTensorRecord> records,
  Map<String, PortSpec> mapping, {
  bool strict = false,
}) {
  final errors = <String>[];
  final warnings = <String>[];

  for (final entry in mapping.entries) {
    final key = entry.key;
    final spec = entry.value;
    final rec = records[key];
    if (rec == null) {
      errors.add('missing key in safetensors: "$key"');
      continue;
    }

    final expectedShape = spec.target.shape;
    final srcShape = rec.shape;
    final wantTranspose = spec.transpose;

    // Validate shape (post-transpose if requested). We allow one specific
    // benign rank mismatch: a rank-1 source `[N]` matching a rank-2 target
    // `[1, N]` — that is exactly how PyTorch stores `nn.Linear.bias`
    // (`[out]`) vs how our `Layer.b` stores it (`[1, out]`). Data layout
    // is identical, so we accept the copy.
    final effectiveSrc = wantTranspose ? srcShape.reversed.toList() : srcShape;
    final shapeOk =
        _shapesEqual(effectiveSrc, expectedShape) ||
        _isBiasBroadcast(effectiveSrc, expectedShape);
    if (!shapeOk) {
      errors.add(
        'shape mismatch for "$key": safetensors=$srcShape'
        '${wantTranspose ? " (transposed→$effectiveSrc)" : ""}, '
        'target=$expectedShape',
      );
      continue;
    }

    final values = rec.asFloat32();
    final dst = wantTranspose
        ? _transpose2d(values, srcShape[0], srcShape[1])
        : values;
    spec.target.data = dst;
  }

  for (final k in records.keys) {
    if (!mapping.containsKey(k)) warnings.add('unused safetensors key: "$k"');
  }

  if (errors.isNotEmpty) {
    throw StateError('loadPytorchInto failed:\n  ${errors.join("\n  ")}');
  }
  if (strict && warnings.isNotEmpty) {
    throw StateError(
      'loadPytorchInto (strict): unused keys present:\n'
      '  ${warnings.join("\n  ")}',
    );
  }
  return warnings;
}

bool _shapesEqual(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// True for the canonical PyTorch-bias case: source `[N]` ⇄ target
/// `[1, N]`. The on-disk byte layout is identical, so the copy is safe.
bool _isBiasBroadcast(List<int> src, List<int> dst) {
  if (src.length == 1 && dst.length == 2 && dst[0] == 1 && dst[1] == src[0]) {
    return true;
  }
  return false;
}

/// Transpose a row-major `[rows, cols]` flat buffer into `[cols, rows]`.
Float32List _transpose2d(Float32List src, int rows, int cols) {
  if (src.length != rows * cols) {
    throw ArgumentError(
      'transpose2d: buffer size ${src.length} != rows*cols ($rows*$cols)',
    );
  }
  final out = Float32List(src.length);
  for (var r = 0; r < rows; r++) {
    for (var c = 0; c < cols; c++) {
      out[c * rows + r] = src[r * cols + c];
    }
  }
  return out;
}
