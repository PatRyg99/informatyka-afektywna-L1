from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

class DGCNNblock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        self.k = k
        self.model = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.model(x)

        return x.max(dim=-1, keepdim=False)[0]

class DGCCNAggregationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.model(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        return torch.cat((x1, x2), 1)


class DGCNN(nn.Module):
    def __init__(self, channels: List[int], head_channels: List[int], num_classes: int, k: int):
        super(DGCNN, self).__init__()

        self.k = k
        self.backbone = nn.ModuleList([
            DGCNNblock(in_channels, out_channels, k) for in_channels, out_channels in zip(channels, channels[1:-1])
        ])
        self.agg = DGCCNAggregationBlock(sum(channels[1:-1]), channels[-1])

        head_channels = [2 * channels[-1]] + head_channels
        self.head = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Dropout(0.3),
                )
                for in_channels, out_channels in zip(head_channels, head_channels[1:])
            ],
            nn.Linear(head_channels[-1], num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        xs = [x]
        for block in self.backbone:
            xi = block(xs[-1])
            xs.append(xi)

        x = torch.cat(xs[1:], dim=1)
        x = self.agg(x)
        x = self.head(x)

        return x
