import 'gpu_tensor.dart';

class Adam {
  final List<Tensor> params;
  final double lr, beta1, beta2, eps, gradClip;
  int t = 0;

  // GPU-resident buffers for first (m) and second (v) moments
  final List<Tensor> m = [];
  final List<Tensor> v = [];

  Adam(
    this.params, {
    this.lr = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.gradClip = 1.0,
  }) {
    for (var p in params) {
      // Initialize moments with zeros on GPU.
      m.add(Tensor.zeros(p.shape));
      v.add(Tensor.zeros(p.shape));
    }
  }

  /// Wipes gradients on the GPU. Called before every backward pass.
  void zeroGrad() {
    for (var p in params) {
      engine.zeroGrad(p.handle);
    }
  }

  /// Performs the parameter update.
  void step() {
    t++; // Crucial for Bias Correction in the first few epochs

    for (int i = 0; i < params.length; i++) {
      // 1. Clip Gradients: Clamps outliers to prevent NaN weight updates
      engine.clipGradients(params[i].handle, gradClip);

      // 2. Adam Update: The heavy lifting happens inside the CUDA kernel
      engine.adamStep(
        params[i].handle,
        m[i].handle,
        v[i].handle,
        t,
        lr,
        beta1,
        beta2,
        eps,
      );
    }
  }

  /// Vital: Manually free GPU memory for moment buffers
  void dispose() {
    for (var tensor in m) tensor.dispose();
    for (var tensor in v) tensor.dispose();
    m.clear();
    v.clear();
  }
}
