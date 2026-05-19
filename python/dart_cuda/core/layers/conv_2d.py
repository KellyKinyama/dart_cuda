"""Conv2d via im2col + matmul — mirrors `lib/core/layers/conv_2d.dart`.

The Dart class does **not** extend Module (just exposes parameters()),
so we keep the same surface here.
"""

from __future__ import annotations

from typing import List

from ..tensor.gpu_tensor import Tensor, engine


class Conv2d:
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        patch_size = in_channels * kernel_size * kernel_size
        init_w = [0.1] * (out_channels * patch_size)
        self.weight = Tensor.from_list([out_channels, patch_size], init_w)
        self.bias = Tensor.from_list([1, out_channels], [0.0] * out_channels)

    # Dart-style camelCase aliases
    @property
    def inChannels(self) -> int:
        return self.in_channels

    @property
    def outChannels(self) -> int:
        return self.out_channels

    @property
    def kernelSize(self) -> int:
        return self.kernel_size

    def forward(self, input: Tensor, tracker: List[Tensor]) -> Tensor:
        h_in = input.shape[1]
        w_in = input.shape[2]
        h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1

        patch_size = self.in_channels * self.kernel_size * self.kernel_size
        num_outputs = h_out * w_out

        col_buffer = Tensor.fill([patch_size, num_outputs], 0.0)
        tracker.append(col_buffer)

        engine.im2col(
            input.handle,
            self.in_channels,
            h_in,
            w_in,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.padding,
            self.stride,
            self.stride,
            col_buffer.handle,
        )

        res = self.weight.matmul(col_buffer)
        tracker.append(res)
        out = res + self.bias
        tracker.append(out)
        return out

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def dispose(self) -> None:
        self.weight.dispose()
        self.bias.dispose()
