from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
class DGCNN(nn.Module):
    def __init__(
        self,
        blocks_mlp: List[int],
        aggr_mlp: List[int],
        head_mlp: List[int],
        k: int = 20,
        aggr: str = "max"
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            DynamicEdgeConv(MLP(mlp), k, aggr) for mlp in blocks_mlp
        ])
        self.aggr_mlp = nn.Linear(*aggr_mlp)
        self.head_mlp = MLP(head_mlp, dropout=0.5)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        # Iterate over blocks and collect interim results
        xs = [pos]
        for block in self.blocks:
            x = block(xs[-1], batch)
            xs.append(x)

        # Aggregate collected results
        out = self.aggr_mlp(torch.cat(xs[1:], dim=1))
        out = global_max_pool(out, batch)

        # Run through head to obtain classification results
        return self.head_mlp(out)
