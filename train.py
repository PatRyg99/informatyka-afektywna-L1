from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning
import typer

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from classifier import PointNetClassifier

app = typer.Typer()


@app.command()
def train(
    checkpoint_path: Path = typer.Option(".output/checkpoints", "-c", "--checkpoint_path"),
    logs_path: Path = typer.Option(".output/logs", "-l", "--logs_path"),
    auto_lr: bool = typer.Option(False, "-al", "--auto_lr"),
) -> Path:

    # initialise the LightningModule
    net = PointNetClassifier(Path(".data/"), 16, 0.001, 8)

    # set up loggers and checkpoints
    tb_logger = TensorBoardLogger(save_dir=str(logs_path))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path, monitor="val_loss", mode="min", save_last=True,
    )

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpus=[0],
        max_epochs=500,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1,
    )

    if auto_lr:
        lr_finder = trainer.tuner.lr_find(net)
        fig = lr_finder.plot(suggest=True)
        fig.suptitle(f"Suggested lr: {lr_finder.suggestion()}", fontsize=16)

        lr_output_path = Path(".data/lr_finder/lr_plot.png")
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