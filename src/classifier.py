import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

import torch
import torch.cuda
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader

import torchmetrics
from torchvision import transforms

from src.dataset.dataset import make_dataset
from src.models.dgcnn import DGCNN
from src.dataset.transforms import PointcloudToPyGData, NormalizePointcloudd, RandomNormalOffsetd, RandomRotationd
from src.dataset.dataloader import InterpolateDataLoader


class Classifier(pl.LightningModule):
    def __init__(self, dataset_path: Path, bs: int, lr: float, num_classes: int):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.bs = bs
        self.lr = lr
        self.num_classes = num_classes

        self.loss = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=num_classes, average="macro")
        self.f1_score = torchmetrics.F1Score(
            num_classes=num_classes, average="macro"
        )

        self.model = DGCNN(
            blocks_mlp=[
                [2 * 3, 64, 64, 64],
                [2 * 64, 64, 128]
            ],
            aggr_mlp=[128 + 64, 256],
            head_mlp=[256, 128, num_classes],
        )

    def forward(self, x):
        return self.model(x)

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
            transforms=transforms.Compose([
                NormalizePointcloudd(["pointcloud"]),
                RandomNormalOffsetd(["pointcloud"], 0.005),
                RandomRotationd(["pointcloud"], [np.pi / 6, np.pi / 6, np.pi / 6]),
                PointcloudToPyGData()
            ])
        )

        self.val_ds = make_dataset(
            root_path=self.dataset_path,
            people_names=split_dict["val"],
            transforms=transforms.Compose([
                NormalizePointcloudd(["pointcloud"]),
                PointcloudToPyGData()
            ])
        )

    def train_dataloader(self):
        return InterpolateDataLoader(
            self.train_ds,
            batch_size=self.bs,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.bs,
            shuffle=False,
            num_workers=8,
            drop_last=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        return optimizer

    def shared_step(self, batch):
        preds = self.forward(batch)

        loss = self.loss(preds, batch.y.long())

        log_preds = F.log_softmax(preds, dim=1)
        acc = self.acc(log_preds, batch.y.int())
        f1_score = self.f1_score(log_preds, batch.y.int())

        return loss, acc, f1_score

    def training_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size=self.bs)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False, logger=True, batch_size=self.bs)
        self.log("train_f1_score", f1_score, on_epoch=True, prog_bar=False, logger=True, batch_size=self.bs)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)

        metrics = {"val_loss": loss, "val_acc": acc, "val_f1_score": f1_score}
        self.log_dict(metrics, on_epoch=True, batch_size=self.bs)
        return metrics
