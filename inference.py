import json
import os
from pathlib import Path

import pandas as pd
import torch
import typer
from monai.transforms import Compose
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.classifier import Classifier
from src.dataset.ck_dataset import make_dataset
from src.dataset.transforms import (
    ComputeHKSFeaturesd,
    GraphToPyGData,
    NormalizePointcloudd,
)

app = typer.Typer()


@app.command()
def inference(
    data_path: Path = typer.Option("./data/CK-dataset", "-d", "--data_path"),
    input_path: Path = typer.Option("output/feast-hks", "-i", "--in_path"),
) -> Path:

    # Load model
    checkpoint_path = os.path.join(input_path, "checkpoint")
    model_path = [
        filename for filename in os.listdir(checkpoint_path) if "last" not in filename
    ][0]

    classifier = Classifier.load_from_checkpoint(
        os.path.join(checkpoint_path, model_path)
    ).eval()
    classifier.cuda()

    # Load split
    with Path("split.json").open() as file:
        json.load(file)

    modes = ["val"]

    for mode in tqdm(modes):
        ds = make_dataset(
            root_path=data_path,
            transforms=Compose(
                [
                    NormalizePointcloudd(["points"]),
                    ComputeHKSFeaturesd(["points"], "hks", 128, 16),
                    GraphToPyGData(x="hks"),
                ]
            ),
        )
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=10)
        preds = []

        for data in tqdm(dl, total=len(dl)):
            pred = classifier(data.cuda())[0]
            predicted_class = torch.argmax(pred).item()

            preds.append({"predicted": predicted_class, "true": data.y.int().item()})

        pd.DataFrame(preds).to_csv(
            os.path.join(input_path, f"{mode}_predictions_affect.csv"), index=None
        )


if __name__ == "__main__":
    app()
