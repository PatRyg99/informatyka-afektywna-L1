from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool, MLP

from src.models.layers.res_block import ResBlock

class SAGEResBlock(ResBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = SAGEConv

        super(SAGEResBlock, self).__init__(convolution, in_channels, out_channels, **kwargs)

class SAGEEncoderBlock(nn.Module):
    def __init__(self, channels: List[int]):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEResBlock(ch_in, ch_out) for ch_in, ch_out in zip(channels, channels[1:])
        ])

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = conv(x, edge_index)

        return x, edge_index


class SAGEGCN(torch.nn.Module):
    def __init__(
        self,
        block_channels: List[int],
        aggr_channels: List[int],
        head_channels: List[int],
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            SAGEEncoderBlock(channels) for channels in block_channels
        ])

        self.aggr_mlp = nn.Linear(*aggr_channels)
        self.head_mlp = MLP(head_channels, dropout=0.5)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]

        for block in self.blocks:
            x, edge_index = block(xs[-1], edge_index)
            xs.append(x)

        out = self.aggr_mlp(torch.cat(xs[1:], dim=1))
        out = global_max_pool(out, batch)

        return self.head_mlp(out)
