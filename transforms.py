import torch
from scipy.spatial.transform import Rotation


class NormalRandomOffsetTransform:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, pointcloud: torch.Tensor) -> torch.Tensor:
        return pointcloud + torch.randn_like(pointcloud) * self.std


class RandomRotation:
    def __call__(self, pointcloud: torch.Tensor) -> torch.Tensor:
        rotation_mat = Rotation.random().as_matrix()
        rotation_mat = torch.from_numpy(rotation_mat).to(pointcloud.device)
        return (rotation_mat @ pointcloud.T).T
