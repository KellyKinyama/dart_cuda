"""GPU Tensor with autograd, mirroring `lib/core/tensor/gpu_tensor.dart`.

The C++ engine owns all device memory. Python wrappers only carry an opaque
``void*`` handle and a shape; ``dispose()`` is the only way to free GPU
memory and must be called explicitly (same constraint as in Dart — the
C++ autograd graph captures raw pointers that Python's GC cannot track).
"""

from __future__ import annotations

import ctypes
import math
import random
from ctypes import POINTER, c_float, c_int32, c_void_p
from typing import Iterable, List, Sequence

from .cuda_engine import CudaEngine, engine

__all__ = ["Tensor", "engine", "CudaEngine"]


def _shape_rows_cols(shape: Sequence[int]) -> tuple[int, int]:
    """Same convention as the Dart constructor: rows = shape[0],
    cols = product of the rest (or 1 for rank-1 input)."""
    rows = shape[0]
    if len(shape) == 1:
        return rows, 1
    cols = 1
    for d in shape[1:]:
        cols *= d
    return rows, cols


def _alloc_float_buffer(values: Iterable[float]) -> tuple[ctypes.Array, int]:
    vals = list(values)
    n = len(vals)
    Buf = c_float * n
    return Buf(*[float(v) for v in vals]), n


def _alloc_int_buffer(values: Iterable[int]) -> ctypes.Array:
    vals = [int(v) for v in values]
    Buf = c_int32 * len(vals)
    return Buf(*vals)


class Tensor:
    """GPU-resident rank-N (effectively rank-2 logically) tensor."""

    # ----- construction -------------------------------------------------
    def __init__(self, handle: int, shape: Sequence[int], *, is_view: bool = False):
        self._handle = c_void_p(handle) if not isinstance(handle, c_void_p) else handle
        self.shape: List[int] = list(shape)
        self.length: int = 1
        for d in self.shape:
            self.length *= d
        self.is_view = is_view
        self._disposed = False

    @property
    def handle(self) -> c_void_p:
        return self._handle

    # ----- factories ---------------------------------------------------
    @classmethod
    def _raw(cls, handle, shape, is_view: bool = False) -> "Tensor":
        return cls(handle, shape, is_view=is_view)

    @classmethod
    def fill(cls, shape: Sequence[int], val: float) -> "Tensor":
        rows, cols = _shape_rows_cols(shape)
        n = rows * cols
        buf = (c_float * n)(*([float(val)] * n))
        h = engine.createTensor(rows, cols, buf)
        return cls(h, shape)

    @classmethod
    def from_list(cls, shape: Sequence[int], vals: Sequence[float]) -> "Tensor":
        rows, cols = _shape_rows_cols(shape)
        buf, _ = _alloc_float_buffer(vals)
        h = engine.createTensor(rows, cols, buf)
        return cls(h, shape)

    fromList = from_list  # Dart-name alias

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> "Tensor":
        return cls.fill(shape, 0.0)

    @classmethod
    def random(cls, shape: Sequence[int], *, scale: float = 0.005) -> "Tensor":
        rows, cols = _shape_rows_cols(shape)
        n_in = shape[0]
        n_out = shape[-1]
        limit = math.sqrt(6.0 / (n_in + n_out))
        n = rows * cols
        rng = random.Random()
        buf = (c_float * n)(*[(rng.random() * 2 - 1) * limit for _ in range(n)])
        h = engine.createTensor(rows, cols, buf)
        return cls(h, shape)

    # ----- host <-> device --------------------------------------------
    @property
    def data(self) -> List[float]:
        buf = (c_float * self.length)()
        engine.getTensorData(self._handle, buf)
        return list(buf)

    @data.setter
    def data(self, new_data: Sequence[float]) -> None:
        if len(new_data) != self.length:
            raise ValueError(
                f"Data length {len(new_data)} mismatch with Tensor length {self.length}"
            )
        buf = (c_float * self.length)(*[float(v) for v in new_data])
        engine.setTensorData(self._handle, buf)

    @property
    def grad(self) -> List[float]:
        buf = (c_float * self.length)()
        engine.getTensorGrad(self._handle, buf)
        return list(buf)

    def fetch_data(self) -> List[float]:
        buf = (c_float * self.length)()
        engine.tensorToHost(self._handle, buf)
        return list(buf)

    fetchData = fetch_data

    def fetch_row(self, row: int) -> List[float]:
        cols = self.shape[1]
        all_data = self.fetch_data()
        return all_data[row * cols : (row + 1) * cols]

    fetchRow = fetch_row

    # ----- autograd controls ------------------------------------------
    def zero_grad(self) -> None:
        engine.zeroGrad(self._handle)

    zeroGrad = zero_grad

    def step(self, lr: float) -> None:
        engine.tensorStep(self._handle, float(lr))

    def backward(self) -> None:
        engine.backward(self._handle)

    # ----- reductions --------------------------------------------------
    def sum(self) -> "Tensor":
        return Tensor._raw(engine.sumTensor(self._handle), [1, 1])

    def mean(self) -> "Tensor":
        return Tensor._raw(engine.meanTensor(self._handle), [1, 1])

    # ----- linear algebra ----------------------------------------------
    def matmul(self, other: "Tensor") -> "Tensor":
        m = self.shape[-2]
        k1 = self.shape[-1]
        k2 = other.shape[0]
        n = other.shape[-1]
        _ = m, k1, k2  # shape checks left to native (mirrors Dart)
        out_shape = list(self.shape)
        out_shape[-1] = n
        return Tensor._raw(
            engine.matmulTensors(self._handle, other._handle), out_shape
        )

    # ----- unary activations -------------------------------------------
    def gelu(self) -> "Tensor":
        return Tensor._raw(engine.geluTensor(self._handle), self.shape)

    def relu(self) -> "Tensor":
        return Tensor._raw(engine.reluTensor(self._handle), self.shape)

    def sigmoid(self) -> "Tensor":
        return Tensor._raw(engine.sigmoidTensor(self._handle), self.shape)

    def tanh(self) -> "Tensor":
        return Tensor._raw(engine.tanhTensor(self._handle), self.shape)

    def log(self) -> "Tensor":
        return Tensor._raw(engine.logTensor(self._handle), self.shape)

    def abs(self) -> "Tensor":
        return Tensor._raw(engine.abs_tensor(self._handle), self.shape)

    def softmax(self) -> "Tensor":
        return Tensor._raw(engine.softmax_forward(self._handle), self.shape)

    def pow(self, e: float) -> "Tensor":
        return Tensor._raw(engine.powTensor(self._handle, float(e)), self.shape)

    # ----- elementwise binary -----------------------------------------
    def _broadcast_for(self, op_func):
        if op_func is engine.addTensors:
            return engine.addTensorScalar
        if op_func is engine.subTensors:
            return engine.subTensorScalar
        if op_func is engine.mulTensors:
            return engine.mulTensorScalar
        if op_func is engine.divTensors:
            return engine.divTensorScalar
        return None

    def _scalar_op(self, other, op_func):
        if isinstance(other, Tensor):
            exact = self.length == other.length
            if not exact and other.length == 1:
                fn = self._broadcast_for(op_func)
                if fn is not None:
                    return Tensor._raw(fn(self._handle, other._handle), self.shape)
            return Tensor._raw(op_func(self._handle, other._handle), self.shape)
        if isinstance(other, (int, float)):
            tmp = Tensor.fill(self.shape, float(other))
            try:
                h = op_func(self._handle, tmp._handle)
            finally:
                tmp.dispose()
            return Tensor._raw(h, self.shape)
        raise TypeError(f"Operation not supported for type {type(other).__name__}")

    def __add__(self, other):
        return self._scalar_op(other, engine.addTensors)

    def __radd__(self, other):
        return self._scalar_op(other, engine.addTensors)

    def __sub__(self, other):
        return self._scalar_op(other, engine.subTensors)

    def __rsub__(self, other):
        # other - self  ==  (-self) + other
        return (-self) + other

    def __neg__(self):
        zero = Tensor.fill(self.shape, 0.0)
        result = zero - self
        zero.dispose()
        return result

    def __mul__(self, other):
        return self._scalar_op(other, engine.mulTensors)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero.")
            return self * (1.0 / float(other))
        return self._scalar_op(other, engine.divTensors)

    # ----- static helpers for AFT / concat / embedding ----------------
    @staticmethod
    def aft(q: "Tensor", k: "Tensor", v: "Tensor", wb: "Tensor", masked: bool) -> "Tensor":
        return Tensor._raw(
            engine.aftForward(q._handle, k._handle, v._handle, wb._handle, masked),
            q.shape,
        )

    @staticmethod
    def aft_cross(q: "Tensor", k: "Tensor", v: "Tensor", wb: "Tensor") -> "Tensor":
        return Tensor._raw(
            engine.aftCrossForward(q._handle, k._handle, v._handle, wb._handle),
            q.shape,
        )

    aftCross = aft_cross

    @staticmethod
    def concat(tensors: Sequence["Tensor"]) -> "Tensor":
        n = len(tensors)
        Arr = c_void_p * n
        # ctypes pointer-array packing
        arr = Arr(*[t._handle.value or 0 for t in tensors])
        h = engine.concatTensors(ctypes.cast(arr, POINTER(c_void_p)), n)
        return Tensor._raw(h, [tensors[0].shape[0], tensors[0].shape[1] * n])

    @staticmethod
    def concat_axis0(tensors: Sequence["Tensor"]) -> "Tensor":
        if not tensors:
            raise ValueError("concat_axis0 needs at least one tensor")
        cols = tensors[0].shape[1]
        total_rows = 0
        for t in tensors:
            if len(t.shape) != 2 or t.shape[1] != cols:
                raise ValueError("concat_axis0 requires equal column counts")
            total_rows += t.shape[0]
        n = len(tensors)
        Arr = c_void_p * n
        arr = Arr(*[t._handle.value or 0 for t in tensors])
        h = engine.concatTensorsAxis0(ctypes.cast(arr, POINTER(c_void_p)), n)
        return Tensor._raw(h, [total_rows, cols])

    concatAxis0 = concat_axis0

    @staticmethod
    def layer_norm(x: "Tensor", gamma: "Tensor", beta: "Tensor", eps: float) -> "Tensor":
        return Tensor._raw(
            engine.layernormForward(x._handle, gamma._handle, beta._handle, float(eps)),
            x.shape,
        )

    layerNorm = layer_norm

    @staticmethod
    def embeddings(idx: Sequence[int], wte: "Tensor", wpe: "Tensor") -> "Tensor":
        T = len(idx)
        D = wte.shape[1]
        buf = _alloc_int_buffer(idx)
        handle = engine.embeddingForward(buf, wte._handle, wpe._handle, T, D)
        return Tensor._raw(handle, [T, D])

    # ----- loss --------------------------------------------------------
    def cross_entropy(self, targets: Sequence[int]) -> "Tensor":
        T, V = self.shape[0], self.shape[1]
        if len(targets) != T:
            raise ValueError(
                f"Target length {len(targets)} must match Logits T: {T}"
            )
        buf = _alloc_int_buffer(targets)
        handle = engine.crossEntropyLoss(self._handle, buf, T, V)
        return Tensor._raw(handle, [1, 1])

    crossEntropy = cross_entropy

    def mse_loss(self, target: "Tensor") -> "Tensor":
        diff = self - target
        squared = diff.pow(2.0)
        total = squared.sum()
        scalar_loss = total / float(self.length)
        return scalar_loss

    mseLoss = mse_loss

    # ----- slicing / views ---------------------------------------------
    def get_row(self, row: int) -> "Tensor":
        if row < 0 or row >= self.shape[0]:
            raise IndexError("Row index out of bounds")
        h = engine.sliceTensor(self._handle, row, 1)
        return Tensor._raw(h, [1, self.shape[1]])

    getRow = get_row

    def slice(self, start_row: int, row_count: int) -> "Tensor":
        if start_row < 0 or (start_row + row_count) > self.shape[0]:
            raise IndexError(f"Slice indices out of bounds for shape {self.shape}")
        h = engine.sliceTensor(self._handle, start_row, row_count)
        return Tensor._raw(h, [row_count, self.shape[1]])

    def reshape(self, new_shape: Sequence[int]) -> "Tensor":
        return Tensor(self._handle, list(new_shape), is_view=True)

    # ----- normalization helpers --------------------------------------
    @staticmethod
    def l2_normalize(input: "Tensor", tracker: List["Tensor"], *, eps: float = 1e-12) -> "Tensor":
        x_sq = input.pow(2.0)
        sum_sq = x_sq.sum()
        norm = sum_sq.pow(0.5) + eps
        result = input / norm
        tracker.extend([x_sq, sum_sq, norm, result])
        return result

    l2Normalize = l2_normalize

    def normalize(self, *, eps: float = 1e-12) -> "Tensor":
        h = engine.layerNorm(self._handle, float(eps))
        return Tensor._raw(h, self.shape)

    # ----- object detection helper -------------------------------------
    def compute_cost_matrix(self, gt_boxes: "Tensor") -> "Tensor":
        if self.shape[1] != 4 or gt_boxes.shape[1] != 4:
            raise ValueError("Both tensors must have 4 columns (x, y, w, h)")
        cost = Tensor.fill([self.shape[0], gt_boxes.shape[0]], 0.0)
        engine.computeCostMatrix(self._handle, gt_boxes._handle, cost._handle)
        return cost

    computeCostMatrix = compute_cost_matrix

    # ----- misc --------------------------------------------------------
    def detach(self) -> "Tensor":
        return Tensor.from_list(self.shape, self.fetch_data())

    def print_matrix(self) -> str:
        cols = self.shape[1] if len(self.shape) > 1 else 1
        rows = self.shape[0]
        data = self.data
        lines = [f"Rows: {rows}, Cols: {cols}"]
        for i in range(rows):
            row = data[i * cols : (i + 1) * cols]
            lines.append(" " + " ".join(f"{v:5.1f}" for v in row))
        return "\n".join(lines)

    printMatrix = print_matrix

    # ----- lifecycle ---------------------------------------------------
    def dispose(self) -> None:
        if self._disposed or self.is_view:
            return
        if self._handle and self._handle.value:
            engine.destroyTensor(self._handle)
        self._disposed = True

    def __repr__(self) -> str:  # pragma: no cover
        return f"Tensor(shape={self.shape}, handle=0x{self._handle.value or 0:x})"
