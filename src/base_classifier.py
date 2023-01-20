from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torchmetrics
from torch.nn import CrossEntropyLoss

from src.dataset.dataloader import TransformsDataLoader
from src.dataset.dl_transforms import InterpolateGraphs
from src.models.dgcnn import DGCNN
from src.models.feast_gcn import FeastGCN
from src.models.sage_gcn import SAGEGCN


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        dataset_path: Path,
        model_name: str,
        bs: int,
        lr: float,
        num_classes: int,
        in_channels: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.bs = bs
        self.lr = lr
        self.num_classes = num_classes

        # Define loss and metrics
        self.loss = CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(num_classes=num_classes, average="macro")
        self.f1_score = torchmetrics.F1Score(num_classes=num_classes, average="macro")

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
        raise NotImplementedError()

    def prepare_data(self):
        raise NotImplementedError()

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
        _, preds = self.forward(batch)

        loss = self.loss(preds, batch.y.long())

        log_preds = F.log_softmax(preds, dim=1)
        acc = self.acc(log_preds, batch.y.int())
        f1_score = self.f1_score(log_preds, batch.y.int())

        metrics = {f"{key}_loss": loss, f"{key}_acc": acc, f"{key}_f1_score": f1_score}
        self.log_dict(
            metrics, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.bs
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")
