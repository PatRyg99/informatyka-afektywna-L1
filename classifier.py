import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchmetrics
from torchvision import transforms

from dataset import make_dataset
from models.dgcnn import DGCNN
from models.pointnet import PointNet
from transforms import NormalRandomOffsetTransform, RandomRotation


class PointNetClassifier(pl.LightningModule):
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
            channels=[3, 64, 128, 256],
            head_channels=[256, 128],
            num_classes=num_classes,
            k=20
        )
        # self.model = PointNet(
        #     dim=3,
        #     channels=(8, 16, 32, 64, 128, 256),
        #     tnets=(True, True, False, False, False),
        #     classes=num_classes,
        #     stride=1,
        #     main_kernel_size=1,
        #     branch_kernel_sizes=(1, 3)
        # )

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
            transforms=transforms.Compose([NormalRandomOffsetTransform(0.005)])
        )
        self.val_ds = make_dataset(root_path=self.dataset_path, people_names=split_dict["val"])

    def train_dataloader(self):
        return DataLoader(
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
        pointclouds, labels = batch
        preds = self.forward(pointclouds.float())

        loss = self.loss(preds, labels.long())

        log_preds = F.log_softmax(preds)
        acc = self.acc(log_preds, labels.int())
        f1_score = self.f1_score(log_preds, labels.int())

        return loss, acc, f1_score

    def training_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_f1_score", f1_score, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1_score = self.shared_step(batch)

        metrics = {"val_loss": loss, "val_acc": acc, "val_f1_score": f1_score}
        self.log_dict(metrics, on_epoch=True)
        return metrics
