from typing import List

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.matrix_features_transform import MatrixFeaturesTransform
from gem_cnn.transform.simple_geometry import SimpleGeometry
from torch_geometric.data import Data
from torch_geometric.nn import MLP, global_max_pool


class GemEncoderBlock(nn.Module):
    def __init__(
        self, channels: List[int], n_rings: int, num_samples: int, max_order: int = 2
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                GemResNetBlock(
                    ch_in,
                    ch_out,
                    max_order,
                    max_order,
                    n_rings=n_rings,
                    num_samples=num_samples,
                )
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
        n_rings: int = 2,
        max_order: int = 2,
        num_samples: int = 7,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                GemEncoderBlock(channels, n_rings=n_rings, num_samples=num_samples)
                for channels in block_channels
            ]
        )

        self.aggr_mlp = nn.Linear(*aggr_channels)
        self.head_mlp = MLP(head_channels, dropout=0.5)

        self.pre_transform = T.Compose(
            (
                SimpleGeometry(),
                MatrixFeaturesTransform(),
                GemPrecomp(n_rings, max_order),
            )
        )

    def extract_features(self, data: Data):
        data = self.pre_transform(data)
        attr = (data.edge_index, data.precomp, data.connection)

        xs = [data.x]

        for block in self.blocks:
            x = block(xs[-1], attr)
            xs.append(x)

        # Aggregate collected results
        graph_out = self.aggr_mlp(torch.cat(xs[1:], dim=1))
        out = global_max_pool(graph_out, data.batch)

        return graph_out, out

    def classify(self, features: torch.Tensor):
        return self.head_mlp(features)

    def forward(self, data: Data):
        _, global_out = self.extract_features(data)
        return self.classify(global_out)
