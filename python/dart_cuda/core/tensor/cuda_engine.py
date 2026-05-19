"""ctypes bindings for `native/lib/libmat_mul.so` (compiled from
`native/src/engine.cu`).

Mirrors `lib/core/tensor/cuda_engine.dart` 1:1 — same function names,
same argument orders. Users should import the `engine` singleton (and
the `Tensor` class) from `dart_cuda.core.tensor.gpu_tensor`, which
re-exports both.

Library lookup order:
  1. ``$DART_CUDA_LIB`` env var (absolute path to the .so).
  2. ``<cwd>/native/lib/libmat_mul.so`` (matches Dart's behaviour).
  3. Walk up from this file looking for ``native/lib/libmat_mul.so``.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    POINTER,
    c_bool,
    c_float,
    c_int,
    c_int32,
    c_void_p,
)
from pathlib import Path


def _find_library() -> str:
    env = os.environ.get("DART_CUDA_LIB")
    if env and os.path.exists(env):
        return env

    candidates = []
    cwd = Path.cwd()
    candidates.append(cwd / "native" / "lib" / "libmat_mul.so")

    # Walk up from this source file to find a repo root with native/lib/.
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidates.append(parent / "native" / "lib" / "libmat_mul.so")

    for c in candidates:
        if c.exists():
            return str(c)

    raise FileNotFoundError(
        "Could not locate libmat_mul.so. Build it with:\n"
        "  nvcc --shared -o native/lib/libmat_mul.so "
        "native/src/engine.cu -Xcompiler -fPIC\n"
        "Or set $DART_CUDA_LIB to its absolute path."
    )


class CudaEngine:
    """Wraps the native CUDA engine. Singleton instance lives in `engine`."""

    def __init__(self, lib_path: str | None = None):
        self.lib_path = lib_path or _find_library()
        self._lib = ctypes.CDLL(self.lib_path)
        self._bind_all()

    # ------------------------------------------------------------------
    def _b(self, name: str, restype, argtypes):
        fn = getattr(self._lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

    def _bind_all(self):
        VP = c_void_p
        FP = POINTER(c_float)
        IP = POINTER(c_int32)
        VPP = POINTER(c_void_p)

        # Construction / destruction
        self.createTensor = self._b("create_tensor", VP, [c_int32, c_int32, FP])
        self.destroyTensor = self._b("destroy_tensor", None, [VP])

        # Host transfer
        self.getTensorData = self._b("get_tensor_data", None, [VP, FP])
        self.getTensorGrad = self._b("get_tensor_grad", None, [VP, FP])
        self.tensorToHost = self._b("tensor_to_host", None, [VP, FP])
        self.setTensorData = self._b("set_tensor_data", None, [VP, FP])

        # Autograd / optim
        self.backward = self._b("backward", None, [VP])
        self.zeroGrad = self._b("zero_grad", None, [VP])
        self.tensorStep = self._b("tensor_step", None, [VP, c_float])
        self.clipGradients = self._b("clip_gradients", None, [VP, c_float])

        # Elementwise binary ops
        self.addTensors = self._b("add_tensors", VP, [VP, VP])
        self.subTensors = self._b("sub_tensors", VP, [VP, VP])
        self.mulTensors = self._b("mul_tensors", VP, [VP, VP])
        self.divTensors = self._b("div_tensors", VP, [VP, VP])
        self.addTensorScalar = self._b("add_tensor_scalar", VP, [VP, VP])
        self.subTensorScalar = self._b("sub_tensor_scalar", VP, [VP, VP])
        self.mulTensorScalar = self._b("mul_tensor_scalar", VP, [VP, VP])
        self.divTensorScalar = self._b("div_tensor_scalar", VP, [VP, VP])
        self.matmulTensors = self._b("matmul_tensors", VP, [VP, VP])

        # Unary
        self.powTensor = self._b("pow_tensor", VP, [VP, c_float])
        self.reluTensor = self._b("relu_tensor", VP, [VP])
        self.geluTensor = self._b("gelu_tensor", VP, [VP])
        self.tanhTensor = self._b("tanh_tensor", VP, [VP])
        self.sigmoidTensor = self._b("sigmoid_tensor", VP, [VP])
        self.logTensor = self._b("log_tensor", VP, [VP])
        self.abs_tensor = self._b("abs_tensor", VP, [VP])
        self.softmax_forward = self._b("softmax_forward", VP, [VP])
        self.sumTensor = self._b("sum_tensor", VP, [VP])
        self.meanTensor = self._b("mean_tensor", VP, [VP])

        # Attention
        self.aftForward = self._b(
            "aft_forward", VP, [VP, VP, VP, VP, c_bool]
        )
        self.aftCrossForward = self._b(
            "aft_cross_forward", VP, [VP, VP, VP, VP]
        )

        # Concat
        self.concatTensors = self._b(
            "concat_tensors_gpu", VP, [VPP, c_int32]
        )
        self.concatTensorsAxis0 = self._b(
            "concat_tensors_axis0_gpu", VP, [VPP, c_int32]
        )

        # LayerNorm / embedding / loss
        self.layernormForward = self._b(
            "layernorm_forward", VP, [VP, VP, VP, c_float]
        )
        self.embeddingForward = self._b(
            "embedding_forward", VP, [IP, VP, VP, c_int32, c_int32]
        )
        self.crossEntropyLoss = self._b(
            "cross_entropy_loss", VP, [VP, IP, c_int32, c_int32]
        )

        # Optimizers
        adam_argtypes = [
            VP, VP, VP, c_int32,
            c_float, c_float, c_float, c_float,
        ]
        self.adamStep = self._b("adam_step", None, adam_argtypes)
        self.sdgStep = self._b("sdg_step", None, adam_argtypes)

        # Slice
        self.sliceTensor = self._b(
            "slice_tensor", VP, [VP, c_int32, c_int32]
        )

        # Init
        self.tensorXavierInit = self._b(
            "tensor_xavier_init", None, [VP, c_int32, c_int32, c_int32]
        )
        self.tensorZeroInit = self._b("tensor_zero_init", None, [VP])

        # Optional: L2 row-norm (named layerNorm in Dart).
        self.layerNorm = self._b("layer_norm_tensor", VP, [VP, c_float])

        # Hungarian / object detection helper
        self._computeCostMatrix = self._b(
            "compute_cost_matrix", None, [VP, VP, VP]
        )

        # Convolution helpers
        im2col_argtypes = [
            VP, c_int32, c_int32, c_int32,
            c_int32, c_int32, c_int32, c_int32,
            c_int32, c_int32, VP,
        ]
        self.im2col = self._b("im2col_cuda", None, im2col_argtypes)
        self.col2im = self._b("col2im_cuda", None, im2col_argtypes)

    # ------------------------------------------------------------------
    def computeCostMatrix(self, pb: int, gb: int, cm: int) -> None:
        self._computeCostMatrix(pb, gb, cm)


# Process-wide singleton, mirrors the `engine` final in cuda_engine.dart.
engine = CudaEngine()
