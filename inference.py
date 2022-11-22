import os
import json
from pathlib import Path
import typer
from tqdm import tqdm
import pandas as pd

import torch
from torchvision import transforms

from src.dataset.dataset import make_dataset
from src.dataset.transforms import NormalizePointcloudd
from src.classifier import Classifier

app = typer.Typer()


@app.command()
def inference(
    data_path: Path = typer.Option("./data/", "-d", "--data_path"),
    input_path: Path = typer.Option("output/10-22-2022.12:46:06-no-neutral", "-i", "--in_path"),
) -> Path:

    # Load model
    checkpoint_path = os.path.join(input_path, "checkpoint")
    model_path = [
        filename for filename in os.listdir(checkpoint_path)
        if "last" not in filename
    ][0]

    classifier = Classifier.load_from_checkpoint(os.path.join(checkpoint_path, model_path)).eval()
    classifier.cuda()

    # Load split
    with Path("split.json").open() as file:
        split_dict = json.load(file)

    modes = ["train", "val"]

    for mode in tqdm(modes):
        ds = make_dataset(
            root_path=data_path,
            people_names=split_dict[mode],
            transforms=transforms.Compose([
                NormalizePointcloudd(["pointcloud"]),
            ])
        )
        preds = []

        for data_dict in tqdm(ds, total=len(ds)):
            x, y = data_dict["pointcloud"], data_dict["label"]

            pred = torch.sigmoid(classifier(x[None, ...].cuda().float())[0])
            predicted_class = torch.argmax(pred).item()

            preds.append({"predicted": predicted_class, "true": y.int().item()})

        pd.DataFrame(preds).to_csv(os.path.join(input_path, f"{mode}_predictions.csv"), index=None)


if __name__ == "__main__":
    app()
