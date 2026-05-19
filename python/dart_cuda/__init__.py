"""dart_cuda: Python bindings for the dart_cuda CUDA tensor engine.

Mirrors the Dart `package:dart_cuda` API (see `lib/` in the repo root).
Both Python and Dart wrap the same `native/lib/libmat_mul.so` shared
library, so models and serialized checkpoints round-trip between them.

Public entry point:

    from dart_cuda import Tensor, engine
    from dart_cuda.core.layers.nn import Module, Layer
    from dart_cuda.core.layers.mlp import MLP
    from dart_cuda.core.optimizers.adam import Adam
"""

from .core.tensor.gpu_tensor import Tensor, engine, CudaEngine

__all__ = ["Tensor", "engine", "CudaEngine"]
__version__ = "0.1.0"
