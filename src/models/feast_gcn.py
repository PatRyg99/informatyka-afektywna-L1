from typing import List

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, FeaStConv, global_max_pool

from src.models.layers.res_block import ResBlock


class FeaStResBlock(ResBlock):
    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = FeaStConv

        super(FeaStResBlock, self).__init__(
            convolution, in_channels, out_channels, **kwargs
        )


class FeaStEncoderBlock(nn.Module):
    def __init__(self, channels: List[int], heads: int):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                FeaStResBlock(ch_in, ch_out, heads=heads)
                for ch_in, ch_out in zip(channels, channels[1:])
            ]
        )

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = conv(x, edge_index)

        return x, edge_index


class FeastGCN(torch.nn.Module):
    def __init__(
        self,
        block_channels: List[int],
        aggr_channels: List[int],
        head_channels: List[int],
        heads: int = 8,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [FeaStEncoderBlock(channels, heads=heads) for channels in block_channels]
        )

        self.aggr_mlp = nn.Linear(*aggr_channels)
        self.head_mlp = MLP(head_channels, dropout=0.5)

    def forward(self, x, edge_index, batch):

        xs = [x]

        for block in self.blocks:
            x, edge_index = block(xs[-1], edge_index)
            xs.append(x)

        out = self.aggr_mlp(torch.cat(xs[1:], dim=1))
        out = global_max_pool(out, batch)

        return self.head_mlp(out)
