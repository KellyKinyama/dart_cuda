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
  final bool useGelu; // We'll use Tanh or Sigmoid as a placeholder for Gelu

  Layer(int nin, int nout, {this.useGelu = true}) {
    // Xavier/Glorot Initialization
    double scale = sqrt(2.0 / (nin + nout));
    w = Tensor.fromList([nin, nout], 
        List.generate(nin * nout, (_) => (Random().nextDouble() * 2 - 1) * scale));
    b = Tensor.fill([1, nout], 0.1);
  }

  // We return the intermediate tensors so the caller can dispose them
  Tensor forward(Tensor x, List<Tensor> tracker) {
    final xw = x.matmul(w);
    tracker.add(xw);
    
    // Note: Our current C++ engine needs a 'bias add' 
    // For now, we'll use + and assume broadcasting is handled or shapes match
    final out = xw + b; 
    tracker.add(out);

    if (useGelu) {
      final act = out.sigmoid(); // Using Sigmoid as placeholder for Gelu
      tracker.add(act);
      return act;
    }
    return out;
  }

  @override
  List<Tensor> parameters() => [w, b];
}