import 'gpu_tensor.dart';
import 'nn.dart';

class LayerNorm extends Module {
  final Tensor gamma;
  final Tensor beta;
  final double eps;

  LayerNorm(int dim, {this.eps = 1e-5})
    : gamma = Tensor.fill([1, dim], 1.0),
      beta = Tensor.fill([1, dim], 0.0);

  Tensor forward(Tensor x, List<Tensor> tracker) {
    final out = Tensor.layerNorm(x, gamma, beta, eps);
    tracker.add(out);
    return out;
  }

  @override
  List<Tensor> parameters() => [gamma, beta];
}
