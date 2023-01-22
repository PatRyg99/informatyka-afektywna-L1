import os
from pathlib import Path

import pytorch_lightning
import typer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.hks_classifier import HKSClassifier
from src.xyz_classifier import XYZClassifier

app = typer.Typer()


@app.command()
def train(
    in_path: Path = typer.Option("data/CK-dataset", "-i", "--in-path"),
    out_path: Path = typer.Option("output", "-o", "--out-path"),
    model_name: str = typer.Option("dgcnn", "-m", "--model-name"),
    features: str = typer.Option("xyz", "-f", "--features"),
    bs: int = typer.Option(16, "-b", "--batch-size"),
    lr: float = typer.Option(0.001, "-l", "--learning-rate"),
    epochs: int = typer.Option(100, "-e", "--epochs"),
) -> Path:

    # Init out directory
    model_dir = f"{model_name}-{features}-rot"
    logs_path = os.path.join(out_path, model_dir, "logs")
    checkpoint_path = os.path.join(out_path, model_dir, "checkpoint")

    # Init the LightningModule
    if features == "xyz":
        net = XYZClassifier(Path(in_path), model_name, bs=bs, lr=lr, num_classes=8)
    elif features == "hks":
        net = HKSClassifier(Path(in_path), model_name, bs=bs, lr=lr, num_classes=8)
    else:
        raise NotImplementedError(
            f"Model for feature type '{features}' is not defined, choose from: [hks, xyz]."
        )

    # Set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_f1_score", mode="max", save_last=True,
    )

    # Init Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=epochs,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    trainer.fit(net)

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model path: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    app()
