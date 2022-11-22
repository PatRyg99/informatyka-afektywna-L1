from typing import List

import torch
from torch_geometric.data import Data

import numpy as np
from monai.transforms.transform import MapTransform, Randomizable
from monai.config import KeysCollection

class PointcloudToPyGData:
    """Convert pointcloud to pytorch geometric data object"""

    def __call__(self, data):
        pos, y = data["pointcloud"], data["label"]
        pyg_data = Data(y=y.long(), pos=pos.float())

        return pyg_data

class NormalizePointcloudd(MapTransform):
    """Normalizes pointcloud to [-0.5, 0.5] cube"""

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            points = d[key]
            points = (points - points.min()) / (points.max() - points.min()) - 0.5

            d[key] = points

        return d

class RandomNormalOffsetd(Randomizable, MapTransform):
    """Apply random normal offset of points"""

    def __init__(
        self,
        keys: KeysCollection,
        std: float = 0.01,
        allow_missing_keys: bool = False
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.std = std

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            points = d[key]
            offset = self.R.normal(0.0, self.std, size=(points.shape))
            points += offset

            d[key] = points

        return d


class RandomRotationd(Randomizable, MapTransform):
    """Apply random rotation specified by angles for each axis"""

    def __init__(
        self,
        keys: KeysCollection,
        angles: List[float],
        allow_missing_keys: bool = False
    ) -> None:

        assert len(angles) == 3, "Array of angles should be of length 3, specyfing rotation angle for each axis."

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.angles = angles

    def _rand_angles(self):
        rand_angles = [
            self.R.random() * angle * (-1) ** self.R.randint(1)
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

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            points = d[key]

            angles = self._rand_angles()
            rotation_mat = np.linalg.multi_dot([self._init_rot_matrix(angle, i) for i, angle in enumerate(angles)])
            rotation_mat = torch.from_numpy(rotation_mat).to(points)

            d[key] = (rotation_mat @ points.T).T

        return d
