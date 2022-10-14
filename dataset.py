from pathlib import Path
from typing import List, Tuple, Callable

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pyvista as pv


class SinglePersonDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: Path, person_name: str, transforms: Callable, percentage_of_used_frames: float = 0.2, percentage_of_neutral_frames: float = 0.08):
        super().__init__()
        self.root_path = root_path
        self.person_name = person_name
        self.transforms = transforms or nn.Identity()
        self.percentage_of_used_frames = percentage_of_used_frames
        self.percentage_of_neutral_frames = percentage_of_neutral_frames

        self.labels_clips_path = self.root_path / "label" / self.person_name
        self.pointclouds_clips_path = self.root_path / "pointcloud" / self.person_name
        self.neutral_label_path = self.root_path / "neutral_emotion.txt"
        self.paths = self.load_paths()

    def load_paths(self) -> List[Tuple[Path, Path]]:
        label_paths = list(self.labels_clips_path.glob("*/*.txt"))
        paths = self.insert_additional_paths(label_paths)
        return paths

    def insert_additional_paths(self, label_paths: List[Path]) -> List[Tuple[Path, Path]]:
        results: List[Tuple[Path, Path]] = []
        for original_label_path in label_paths:
            for pointcloud_path in self.get_pointclouds_paths_from_label_path(original_label_path):
                results.append((pointcloud_path, original_label_path))
            for neutral_pointcloud_path in self.get_neutral_pointclouds_paths_from_label_path(original_label_path):
                results.append((neutral_pointcloud_path, self.neutral_label_path))
        return results

    def get_pointclouds_paths_from_label_path(self, original_label_path: Path) -> List[Path]:
        clip_name = original_label_path.parent.name
        # splitting filenames that look like S005_001_00000011_emotion.txt
        label_index = int(original_label_path.stem.split("_")[2])
        min_used_image_index = int(label_index * (1 - self.percentage_of_used_frames))
        return [
            self.pointclouds_clips_path / clip_name / f"{self.person_name}_{clip_name}_{image_index:08d}.vtk"
            for image_index
            in range(min_used_image_index, label_index + 1)
        ]

    def get_neutral_pointclouds_paths_from_label_path(self, original_label_path: Path) -> List[Path]:
        clip_name = original_label_path.parent.name
        # splitting filenames that look like S005_001_00000011_emotion.txt
        label_index = int(original_label_path.stem.split("_")[2])
        max_used_image_index = int(label_index * self.percentage_of_neutral_frames)
        return [
            self.pointclouds_clips_path / clip_name / f"{self.person_name}_{clip_name}_{image_index:08d}.vtk"
            for image_index
            in range(1, max_used_image_index + 1)
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pointcloud_path, label_path = self.paths[index]
        return self.load_points(pointcloud_path), self.load_label(label_path)

    def load_label(self, label_path: Path) -> torch.Tensor:
        return torch.from_numpy(np.loadtxt(label_path, dtype=np.float32))

    def load_points(self, pointcloud_path: Path) -> torch.Tensor:
        poly_data: pv.PolyData = pv.read(str(pointcloud_path))
        poly: torch.Tensor = torch.from_numpy(poly_data.points)
        poly = (poly - poly.min()) / (poly.max() - poly.min()) - 0.5
        return self.transforms(poly)


def make_dataset(root_path: Path, people_names: List[str], transforms: Callable = None) -> torch.utils.data.Dataset:
    return torch.utils.data.ConcatDataset(
        [
            SinglePersonDataset(root_path, person_name=person_name, transforms=transforms)
            for person_name in people_names
        ]
    )
