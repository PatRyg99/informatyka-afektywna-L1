import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyvista as pv
import robust_laplacian
import scipy
import scipy.sparse.linalg as sla
import torch
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, Randomizable
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


class LoadSampled(MapTransform):
    """Loads affectNet sample"""

    def __init__(
        self,
        keys: KeysCollection,
        labels_path: str,
        label_map: Dict[str, int],
        allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.labels = pd.read_csv(labels_path)
        self.label_map = label_map

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            vtk_path = d[key]

            poly_data: pv.PolyData = pv.read(str(vtk_path))
            points: torch.Tensor = torch.from_numpy(poly_data.points)
            edges: torch.Tensor = torch.from_numpy(
                poly_data.lines.reshape(-1, 3)[:, 1:]
            )

            jpg_rel_path = os.path.join(
                *str(Path(vtk_path).with_suffix(".jpg")).split("/")[-2:]
            )
            label: int = self.label_map[
                self.labels[self.labels["pth"] == jpg_rel_path]["label"].item()
            ]

            d["points"] = points
            d["edges"] = edges
            d["label"] = torch.tensor(label)

        return d


class GraphToPyGData:
    """Convert graph to pytorch geometric data object"""

    def __init__(self, x_key: str = None) -> None:
        self.x_key = x_key

    def __call__(self, data):

        if self.x_key is None:
            pos, edge_index, y = (
                data["points"],
                data["edges"],
                data["label"],
            )

            edge_index = to_undirected(edge_index.T.long())
            pyg_data = Data(y=y.long(), pos=pos.float(), edge_index=edge_index)

        else:
            x, pos, edge_index, y = (
                data[self.x_key],
                data["points"],
                data["edges"],
                data["label"],
            )

            edge_index = to_undirected(edge_index.T.long())
            pyg_data = Data(
                y=y.long(), x=x.float(), pos=pos.float(), edge_index=edge_index
            )

        return pyg_data


class ComputeHKSFeaturesd(MapTransform):
    """
    Computes heat kernel signatures based on eigendecomposition
    of a Laplace-Beltrami operator of a surface.
    """

    def __init__(
        self,
        keys: KeysCollection,
        hks_key: str,
        k_eig: int,
        num_features: int,
        eps: float = 1e-8,
        allow_missing_keys: bool = False,
    ) -> None:

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.hks_key = hks_key
        self.k_eig = k_eig
        self.num_features = num_features
        self.eps = eps

    def _compute_hks(self, points: np.array):

        # Build laplacian
        L, M = robust_laplacian.point_cloud_laplacian(points)
        massvec = M.diagonal()

        # Compute eigenbasis
        L_eigsh = (L + scipy.sparse.identity(L.shape[0]) * self.eps).tocsc()
        massvec_eigsh = massvec
        Mmat = scipy.sparse.diags(massvec_eigsh)

        evals, evecs = sla.eigsh(L_eigsh, k=self.k_eig, M=Mmat, sigma=self.eps)
        evals = np.clip(evals, a_min=0.0, a_max=float("inf"))

        # Compute hks
        scales = np.logspace(-2, 0.0, num=self.num_features)

        power_coefs = np.exp(-evals[None] * scales[..., None])[None]
        terms = power_coefs * (evecs * evecs)[:, None]
        out = np.sum(terms, axis=-1)

        return out

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            points = d[key]
            hks = self._compute_hks(points.numpy())
            d[self.hks_key] = torch.tensor(hks)

        return d


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
        self, keys: KeysCollection, std: float = 0.01, allow_missing_keys: bool = False
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
        allow_missing_keys: bool = False,
    ) -> None:

        assert (
            len(angles) == 3
        ), "Array of angles should be of length 3, specyfing rotation angle for each axis."

        MapTransform.__init__(self, keys, allow_missing_keys)
        self.angles = angles

    def _rand_angles(self):
        rand_angles = [
            self.R.random() * angle * (-1) ** self.R.randint(1) for angle in self.angles
        ]
        return rand_angles

    def _init_rot_matrix(self, angle: float, axis: int):
        rot = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)],
            ]
        )

        rot = np.roll(rot, shift=(axis, axis), axis=(0, 1))
        return rot

    def __call__(self, data):
        d = dict(data)

        for key in self.key_iterator(d):
            points = d[key]

            angles = self._rand_angles()
            rotation_mat = np.linalg.multi_dot(
                [self._init_rot_matrix(angle, i) for i, angle in enumerate(angles)]
            )
            rotation_mat = torch.from_numpy(rotation_mat).to(points)

            d[key] = (rotation_mat @ points.T).T

        return d
