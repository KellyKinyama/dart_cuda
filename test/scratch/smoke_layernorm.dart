import 'package:dart_cuda/core/tensor/gpu_tensor.dart';

void main() {
  print('layer_norm tensor (no affine)');
  final a = Tensor.fromList([3, 8], List.generate(24, (i) => (i + 1).toDouble()));
  final out = a.normalize(eps: 1e-5);
  print('layerNorm out: ${out.data}');
  out.dispose();
  a.dispose();

  print('layer_norm with affine');
  final x = Tensor.fromList([3, 8], List.generate(24, (i) => (i + 1).toDouble()));
  final g = Tensor.fromList([1, 8], List.filled(8, 1.0));
  final b = Tensor.fromList([1, 8], List.filled(8, 0.0));
  final y = Tensor.layerNorm(x, g, b, 1e-5);
  print('layerNorm affine out: ${y.data}');
  y.dispose(); g.dispose(); b.dispose(); x.dispose();

  print('softmax');
  final s = Tensor.fromList([2, 5], [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]).softmax();
  print('softmax: ${s.data}');
  s.dispose();

  print('OK');
}
