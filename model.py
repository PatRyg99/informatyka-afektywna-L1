from typing import Tuple

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block wrapper"""

    def __init__(self, module, residual_module=nn.Identity()):
        super().__init__()
        self.module = module
        self.residual_module = residual_module

    def forward(self, x):
        return self.module(x) + self.residual_module(x)


class MultiBranchBlock(nn.Module):
    """Multi branch block wrapper"""

    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list

    def forward(self, x):
        return torch.cat([module(x) for module in self.module_list], dim=1)


class TNet(nn.Module):
    """Transformation network regressing affine matrix to transform trajectory with"""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=2,
                bias=True,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=2,
                bias=True,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2, bias=True),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.GELU(),
            nn.Linear(hidden_channels // 2, in_channels ** 2, bias=True),
        )

    def forward(self, x):
        bs = x.shape[0]

        output = self.stem(x)
        output = output.max(dim=2)[0]
        output = self.head(output).reshape(bs, self.in_channels, self.in_channels)

        return output @ x


class PointNet(nn.Module):
    """PointNet architecture"""

    def __init__(
        self,
        dim: int,
        channels: Tuple[int],
        tnets: Tuple[int],
        classes: int,
        stride: int,
        main_kernel_size: int,
        branch_kernel_sizes: Tuple[int],
    ):
        super().__init__()

        self.dim = dim
        self.classes = classes

        channels = (dim,) + channels

        self.stem = nn.Sequential(
            *[
                self.block(
                    channels[i],
                    channels[i + 1],
                    stride,
                    main_kernel_size,
                    branch_kernel_sizes,
                    tnets[i],
                )
                for i in range(len(channels) - 2)
            ],
            nn.Conv1d(
                in_channels=channels[-2],
                out_channels=channels[-1],
                kernel_size=main_kernel_size,
                padding_mode="replicate",
                bias=True,
            ),
        )

        self.head = nn.Sequential(
            nn.Linear(channels[-1], 256, bias=True),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, self.classes, bias=True),
        )

    def block(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        main_kernel_size: int,
        branch_kernel_sizes: Tuple[int],
        tnet: bool = False,
    ):
        branch_channels = out_channels // len(branch_kernel_sizes)

        return nn.Sequential(
            TNet(in_channels=in_channels, hidden_channels=branch_channels)
            if tnet
            else nn.Identity(),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=main_kernel_size,
                stride=stride,
                padding_mode="replicate",
                bias=True,
            ),
            MultiBranchBlock(
                nn.ModuleList(
                    [
                        ResBlock(
                            nn.Sequential(
                                nn.Conv1d(
                                    in_channels=branch_channels,
                                    out_channels=branch_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - 1) // 2,
                                    padding_mode="replicate",
                                    bias=True,
                                ),
                                nn.BatchNorm1d(branch_channels),
                                nn.GELU(),
                                nn.Conv1d(
                                    in_channels=branch_channels,
                                    out_channels=branch_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - 1) // 2,
                                    padding_mode="replicate",
                                    bias=True,
                                ),
                                nn.BatchNorm1d(branch_channels),
                                nn.GELU(),
                            ),
                        )
                        for kernel_size in branch_kernel_sizes
                    ]
                )
            ),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        bs = x.shape[0]
        x = x.reshape(bs, self.dim, -1)

        output = self.stem(x)
        output = output.max(dim=2)[0]
        output = self.head(output)

        return output