import 'gpu_tensor.dart';
import 'dart:math';

abstract class Module {
  List<Tensor> parameters();
  void zeroGrad() {
    for (var p in parameters()) p.zeroGrad();
  }

  void step(double lr) {
    for (var p in parameters()) p.step(lr);
  }
}

class Layer extends Module {
  late Tensor w;
  late Tensor b;
  final bool useGelu;

  Layer(int nin, int nout, {this.useGelu = true}) {
    // Xavier/Glorot Initialization
    double scale = sqrt(2.0 / (nin + nout));
    w = Tensor.fromList(
      [nin, nout],
      List.generate(nin * nout, (_) => (Random().nextDouble() * 2 - 1) * scale),
    );
    // Initialize bias slightly positive to prevent "dead" neurons
    b = Tensor.fill([1, nout], 0.01);
  }

  Tensor forward(Tensor x, List<Tensor> tracker) {
    // 1. xw = x * w
    final xw = x.matmul(w);

    // 2. out = xw + b (Broadcasting handled by our C++ max(shape) logic)
    final out = xw + b;

    tracker.addAll([xw, out]);

    if (useGelu) {
      // 3. act = gelu(out)
      final act = out.gelu();
      tracker.add(act);
      return act;
    }

    return out;
  }

  @override
  List<Tensor> parameters() => [w, b];
}
