from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


class DGCNN(nn.Module):
    def __init__(
        self,
        blocks_mlp: List[int],
        aggr_mlp: List[int],
        head_mlp: List[int],
        k: int = 20,
        aggr: str = "max",
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [DynamicEdgeConv(MLP(mlp), k, aggr) for mlp in blocks_mlp]
        )
        self.aggr_mlp = nn.Linear(*aggr_mlp)
        self.head_mlp = MLP(head_mlp, dropout=0.5)

    def extract_features(self, data: Data):

        # Iterate over blocks and collect interim results
        xs = [data.x]
        for block in self.blocks:
            x = block(xs[-1], data.batch)
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
