import os
import json
from pathlib import Path
import typer
from tqdm import tqdm
import torch
import pandas as pd

from src.dataset.dataset import make_dataset
from src.classifier import Classifier

app = typer.Typer()


@app.command()
def inference(
    data_path: Path = typer.Option("./data/", "-d", "--data_path"),
    input_path: Path = typer.Option("output/10-22-2022.11:50:10", "-i", "--in_path"),
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

    # Inference train data
    train_ds = make_dataset(root_path=data_path, people_names=split_dict["train"])
    train_preds = []

    for x, y in tqdm(list(train_ds)):
        pred = torch.sigmoid(classifier(x[None, ...].cuda().float())[0])
        predicted_class = torch.argmax(pred).item()

        train_preds.append({"predicted": predicted_class, "true": y.int().item()})

    pd.DataFrame(train_preds).to_csv(os.path.join(input_path, "train_predictions.csv"), index=None)

    # Inference valid data
    val_ds = make_dataset(root_path=data_path, people_names=split_dict["val"])
    val_preds = []

    for x, y in tqdm(list(val_ds)):
        pred = torch.sigmoid(classifier(x[None, ...].cuda().float())[0])
        predicted_class = torch.argmax(pred).item()

        val_preds.append({"predicted": predicted_class, "true": y.int().item()})

    pd.DataFrame(val_preds).to_csv(os.path.join(input_path, "val_predictions.csv"), index=None)


if __name__ == "__main__":
    app()
