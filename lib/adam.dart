import 'gpu_tensor.dart';

class Adam {
  final List<Tensor> params;
  final double lr, beta1, beta2, eps;
  int t = 0;

  // We store the first and second moments on the GPU
  final List<Tensor> m = [];
  final List<Tensor> v = [];

  Adam(
    this.params, {
    this.lr = 0.001,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
  }) {
    for (var p in params) {
      // Initialize moments with zeros on GPU
      m.add(Tensor.zeros(p.shape));
      v.add(Tensor.zeros(p.shape));
    }
  }

  void zeroGrad() {
    for (var p in params) {
      // You need a kernel for this, or a C++ function
      // DLLEXPORT void zero_grad(void* handle)
      engine.zeroGrad(p.handle);
    }
  }

  void step() {
    t++;
    for (int i = 0; i < params.length; i++) {
      // This calls a custom CUDA kernel that does:
      // m = beta1 * m + (1 - beta1) * grad
      // v = beta2 * v + (1 - beta2) * grad^2
      // param = param - lr * m_hat / (sqrt(v_hat) + eps)
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
}
