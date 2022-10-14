from pathlib import Path
from typing import List, Tuple

import torch
import torch.utils.data
import numpy as np
import pyvista as pv


class SinglePersonDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: Path, person_name: str, num_additional_frames: float = 0.2):
        super().__init__()
        self.root_path = root_path
        self.person_name = person_name
        self.percentage_of_used_frames = num_additional_frames
        self.labels_clips_path = self.root_path / "label" / self.person_name
        self.pointclouds_clips_path = self.root_path / "pointcloud" / self.person_name
        self.paths = self.load_paths()

    def load_paths(self) -> List[Tuple[Path, Path]]:
        label_paths = list(self.labels_clips_path.glob("*/*.txt"))
        pointcloud_paths = [self.pointcloud_path_from_label_path(label_path) for label_path in label_paths]
        paths = list(zip(pointcloud_paths, label_paths))
        paths = self.insert_additional_paths(paths)
        return paths

    def pointcloud_path_from_label_path(self, label_path: Path) -> Path:
        pointcloud_path = self.pointclouds_clips_path / label_path.relative_to(self.labels_clips_path)
        pointcloud_path = pointcloud_path.with_stem(pointcloud_path.stem.rstrip("_emotion")).with_suffix(".vtk")
        return pointcloud_path

    def insert_additional_paths(self, paths: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        results: List[Tuple[Path, Path]] = []
        for original_pointcloud_path, original_label_path in paths:
            clip_name = original_label_path.parent.name
            used_image_indexes = self.get_pointclouds_indexes_from_label_path(original_label_path)
            for image_index in used_image_indexes:
                pointcloud_path = original_pointcloud_path.parent / f"{self.person_name}_{clip_name}_{image_index:08d}.vtk"
                results.append((pointcloud_path, original_label_path))
        return results

    def get_pointclouds_indexes_from_label_path(self, label_path: Path) -> List[int]:
        # splitting filenames that look like S005_001_00000011_emotion.txt
        original_image_index = int(label_path.stem.split("_")[2])
        min_used_image_index = int(original_image_index * (1 - self.percentage_of_used_frames))
        used_image_indexes = list(range(min_used_image_index, original_image_index + 1))
        return used_image_indexes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        pointcloud_path, label_path = self.paths[index]
        return self.load_points(pointcloud_path), self.load_label(label_path)

    def load_label(self, label_path: List[Path]) -> torch.Tensor:
        return torch.from_numpy(np.loadtxt(label_path, dtype=np.float32))

    def load_points(self, pointcloud_path: List[Path]) -> List[torch.Tensor]:
        poly: pv.PolyData = pv.read(str(pointcloud_path))
        poly: torch.Tensor = torch.from_numpy(poly.points)
        poly = (poly - poly.min()) / (poly.max() - poly.min()) - 0.5
        return poly


def make_dataset(root_path: Path, people_names: List[str]) -> torch.utils.data.Dataset:
    return torch.utils.data.ConcatDataset(
        [
            SinglePersonDataset(root_path, person_name=person_name)
            for person_name in people_names
        ]
    )
