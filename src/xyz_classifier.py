import json
from pathlib import Path

import numpy as np
from monai.transforms import Compose

from src.base_classifier import BaseClassifier
from src.dataset.ck_dataset import make_dataset
from src.dataset.transforms import (
    GraphToPyGData,
    NormalizePointcloudd,
    RandomNormalOffsetd,
    RandomRotationd,
)


class XYZClassifier(BaseClassifier):
    def __init__(
        self,
        dataset_path: Path,
        model_name: str,
        bs: int,
        lr: float,
        num_classes: int,
        in_channels: int = 3,
    ):
        super().__init__(dataset_path, model_name, bs, lr, num_classes, in_channels)

    def forward(self, data):
        x = data.pos
        edge_index, batch = data.edge_index, data.batch

        features = self.model.extract_features(x, edge_index, batch)
        output = self.model.classify(features)

        return features, output

    def prepare_data(self):

        with Path("split.json").open() as file:
            split_dict = json.load(file)

        self.train_ds = make_dataset(
            root_path=self.dataset_path,
            people_names=split_dict["train"],
            transforms=Compose(
                [
                    NormalizePointcloudd(["points"]),
                    RandomNormalOffsetd(["points"], 0.005),
                    RandomRotationd(["points"], [np.pi / 6, np.pi / 6, np.pi / 6]),
                    GraphToPyGData(),
                ]
            ),
        )

        self.val_ds = make_dataset(
            root_path=self.dataset_path,
            people_names=split_dict["val"],
            transforms=Compose([NormalizePointcloudd(["points"]), GraphToPyGData()]),
        )
