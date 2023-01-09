from typing import List

import torch
import torch.nn as nn
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from torch_geometric.nn import MLP, global_max_pool


class GemEncoderBlock(nn.Module):
    def __init__(self, channels: List[int], max_order: int = 2):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                GemResNetBlock(ch_in, ch_out, max_order, max_order)
                for ch_in, ch_out in zip(channels, channels[1:])
            ]
        )

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = conv(x, edge_index)

        return x, edge_index


class GemGCN(torch.nn.Module):
    def __init__(
        self,
        block_channels: List[int],
        aggr_channels: List[int],
        head_channels: List[int],
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [GemEncoderBlock(channels) for channels in block_channels]
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
