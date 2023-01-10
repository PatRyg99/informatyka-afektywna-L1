import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning
import typer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.classifier import Classifier

app = typer.Typer()


@app.command()
def train(
    in_path: Path = typer.Option("data/CK-dataset", "-i", "--in_path"),
    out_path: Path = typer.Option("output", "-o", "--out_path"),
    model_name: str = typer.Option("dgcnn", "-m", "--model-name"),
    features: str = typer.Option("xyz", "-f", "--features"),
    auto_lr: bool = typer.Option(False, "-al", "--auto_lr"),
) -> Path:

    # Init out directory
    ct = datetime.datetime.now().strftime("%m-%d-%Y.%H:%M:%S")
    logs_path = os.path.join(out_path, ct, "../logs")
    checkpoint_path = os.path.join(out_path, ct, "checkpoint")

    # Init the LightningModule
    net = Classifier(
        Path(in_path), model_name, features, bs=16, lr=0.001, num_classes=8
    )

    # Set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_f1_score", mode="max", save_last=True,
    )

    # Init Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=100,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(net)
        fig = lr_finder.plot(suggest=True)
        fig.suptitle(f"Suggested lr: {lr_finder.suggestion()}", fontsize=16)

        lr_output_path = Path("./data/lr_finder/lr_plot.png")
        lr_output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(lr_output_path)

        print("Suggested lr: ", lr_finder.suggestion())

    else:
        trainer.fit(net)

        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        return best_model_path


if __name__ == "__main__":
    app()
