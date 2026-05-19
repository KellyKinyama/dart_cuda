/// Standard scaled-dot-product self-attention (single head).
///
/// Port of the Python `dart_cuda.core.attention.self_attention` module.
/// Forward (T = sequence length, D_in = embed size, H = head size):
///   Q = X @ Wq      [T, H]
///   K = X @ Wk      [T, H]
///   V = X @ Wv      [T, H]
///   S = Q @ K^T / sqrt(H)            [T, T]
///   if masked: add upper-triangular -1e9 (causal mask)
///   P = softmax(S, axis=-1)          [T, T]
///   out = P @ V                       [T, H]
import 'dart:math' as math;

import 'package:dart_cuda/core/layers/nn.dart';
import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

/// Upper-triangular additive causal mask: -1e9 above the diagonal, 0 below.
Tensor causalMask(int T) {
  final vals = List<double>.filled(T * T, 0.0);
  for (int i = 0; i < T; i++) {
    for (int j = i + 1; j < T; j++) {
      vals[i * T + j] = -1e9;
    }
  }
  return Tensor.fromList([T, T], vals);
}

/// Composable scaled-dot-product attention built from transpose + matmul +
/// softmax. All intermediates (including [mask]) are appended to [tracker].
Tensor scaledDotProductAttention(
  Tensor q,
  Tensor k,
  Tensor v,
  List<Tensor> tracker, {
  bool masked = false,
}) {
  final H = q.shape.last;
  final scale = 1.0 / math.sqrt(H.toDouble());
  final kt = k.transpose();
  final scores = q.matmul(kt);
  final scaled = scores * scale;
  tracker.addAll([kt, scores, scaled]);

  Tensor pre = scaled;
  if (masked) {
    final mask = causalMask(q.shape[0]);
    final maskedScores = scaled + mask;
    tracker.addAll([mask, maskedScores]);
    pre = maskedScores;
  }

  final weights = pre.softmax();
  final out = weights.matmul(v);
  tracker.addAll([weights, out]);
  return out;
}

class SelfAttention extends Module {
  final int embedSize;
  final int headSize;
  final bool masked;
  final Layer key;
  final Layer query;
  final Layer value;

  SelfAttention(this.embedSize, this.headSize, {this.masked = false})
    : key = Layer(embedSize, headSize, useGelu: false),
      query = Layer(embedSize, headSize, useGelu: false),
      value = Layer(embedSize, headSize, useGelu: false);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final q = query.forward(x, tracker);
    final k = key.forward(x, tracker);
    final v = value.forward(x, tracker);
    return scaledDotProductAttention(q, k, v, tracker, masked: masked);
  }

  @override
  List<Tensor> parameters() => [
    ...query.parameters(),
    ...key.parameters(),
    ...value.parameters(),
  ];
}
