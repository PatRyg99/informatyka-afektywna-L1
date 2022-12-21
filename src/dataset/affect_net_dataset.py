import os
from pathlib import Path
from typing import List, Tuple, Callable

import torch
import torch.utils.data
import monai


def make_dataset(
    root_path: Path,
    transforms: Callable = None,
    num_workers: int = 6
) -> torch.utils.data.Dataset:

    data_dict = [
        {
            "sample_path": os.path.join(dir, file)
        }
        for dir, _, files in os.walk(root_path) for file in files if file.endswith(".vtk")
    ]

    return monai.data.CacheDataset(data=data_dict, cache_rate=1.0, transform=transforms, num_workers=num_workers)
