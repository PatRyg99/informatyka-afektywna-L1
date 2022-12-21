import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torchmetrics
from monai.transforms import Compose
from torch.nn import CrossEntropyLoss

from src.dataset.ck_dataset import make_dataset
from src.dataset.dataloader import TransformsDataLoader
from src.dataset.dl_transforms import InterpolateGraphs
from src.dataset.transforms import (
    ComputeHKSFeaturesd,
    GraphToPyGData,
    NormalizePointcloudd,
    RandomNormalOffsetd,
    RandomRotationd,
)
from src.models.dgcnn import DGCNN
from src.models.feast_gcn import FeastGCN
from src.models.sage_gcn import SAGEGCN


class Classifier(pl.LightningModule):
    def __init__(
        self,
        dataset_path: Path,
        model_name: str,
        features: str,
        bs: int,
        lr: float,
        num_classes: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.features = features
        self.dataset_path = dataset_path
        self.bs = bs
        self.lr = lr
        self.num_classes = num_classes

        # Define loss and metrics
        self.loss = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=num_classes, average="macro")
        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, average="macro")

        # Define features
        in_channels = 3 if self.features == "xyz" else 16

        # Define model
        if model_name == "dgcnn":
            self.model = DGCNN(
                blocks_mlp=[[2 * in_channels, 64, 64, 64], [2 * 64, 64, 128]],
                aggr_mlp=[128 + 64, 256],
                head_mlp=[256, 128, num_classes],
            )

        elif model_name == "feast":
            self.model = FeastGCN(
                block_channels=[[in_channels, 16, 32, 64], [64, 64, 64, 128]],
                aggr_channels=[128 + 64, 256],
                head_channels=[256, 128, num_classes],
            )

        elif model_name == "sage":
            self.model = SAGEGCN(
                block_channels=[[in_channels, 16, 32, 64], [64, 64, 64, 128]],
                aggr_channels=[128 + 64, 256],
                head_channels=[256, 128, num_classes],
            )

        else:
            raise NotImplementedError(
                f"Model '{model_name}' is not implemented. Choose from: ['dgcnn', 'feast', 'sage']."
            )

    def forward(self, data):
        x = data.pos if self.features == "xyz" else data.x
        edge_index, batch = data.edge_index, data.batch

        return self.model(x, edge_index, batch)

    def remap_labels(self, labels):
        uniques = np.unique(labels)

        for unique in uniques:
            labels[labels == unique] = np.argwhere(uniques == unique).flatten()

        return labels

    def prepare_data(self):

        with Path("split.json").open() as file:
            split_dict = json.load(file)

        self.train_ds = make_dataset(
            root_path=self.dataset_path,
            people_names=split_dict["train"],
            transforms=Compose(
                [
                    NormalizePointcloudd(["points"]),
                    ComputeHKSFeaturesd(["points"], "hks", 128, 16),
                    RandomNormalOffsetd(["points"], 0.005),
                    RandomRotationd(["points"], [np.pi / 6, np.pi / 6, np.pi / 6]),
                    GraphToPyGData(x="hks"),
                ]
            ),
        )

        self.val_ds = make_dataset(
            root_path=self.dataset_path,
            people_names=split_dict["val"],
            transforms=Compose(
                [
                    NormalizePointcloudd(["points"]),
                    ComputeHKSFeaturesd(["points"], "hks", 128, 16),
                    GraphToPyGData(x="hks"),
                ]
            ),
        )

    def train_dataloader(self):
        transforms = InterpolateGraphs()

        return TransformsDataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=8,
            drop_last=True,
            transforms=transforms,
        )

    def val_dataloader(self):
        return TransformsDataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=8,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def shared_step(self, batch, key):
        preds = self.forward(batch)

        loss = self.loss(preds, batch.y.long())

        log_preds = F.log_softmax(preds, dim=1)
        acc = self.acc(log_preds, batch.y.int())
        f1_score = self.f1_score(log_preds, batch.y.int())

        metrics = {f"{key}_loss": loss, f"{key}_acc": acc, f"{key}_f1_score": f1_score}
        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=self.bs)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
