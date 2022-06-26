from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        conv_fn: Optional[str] = "Conv2d",
        norm_fn: Optional[str] = "BatchNorm2d",
        act_fn: Optional[str] = "ReLU",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv = eval(f"nn.{conv_fn}")(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = eval(f"nn.{norm_fn}")(num_features=out_channels)
        self.activate = eval(f"nn.{act_fn}")(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activate(x)

        return x
