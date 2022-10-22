from typing import List
import numpy as np
import torch


class NormalRandomOffsetTransform:
    def __init__(self, std: float) -> None:
        self.std = std

    def __call__(self, pointcloud: torch.Tensor) -> torch.Tensor:
        return pointcloud + torch.randn_like(pointcloud) * self.std


class RandomRotation:
    def __init__(self, angles: List[float]):
        assert len(angles) == 3, "Array of angles should be of length 3, specyfing rotation angle for each axis."
        self.angles = angles

    def _rand_angles(self):
        rand_angles = [
            np.random.random() * angle * (-1) ** np.random.randint(1)
            for angle in self.angles
        ]
        return rand_angles

    def _init_rot_matrix(self, angle: float, axis: int):
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

        rot = np.roll(rot, shift=(axis, axis), axis=(0, 1))
        return rot

    def __call__(self, pointcloud: torch.Tensor) -> torch.Tensor:
        angles = self._rand_angles()
        rotation_mat = np.linalg.multi_dot([self._init_rot_matrix(angle, i) for i, angle in enumerate(angles)])
        rotation_mat = torch.from_numpy(rotation_mat).to(pointcloud)
        return (rotation_mat @ pointcloud.T).T
