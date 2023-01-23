import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import typer
from monai.transforms import Compose
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.dataset.ck_dataset import make_dataset
from src.dataset.transforms import (
    ComputeHKSFeaturesd,
    ComputeNormalsd,
    GraphToPyGData,
    NormalizePointcloudd,
)
from src.hks_classifier import HKSClassifier
from src.xyz_classifier import XYZClassifier

app = typer.Typer()


@app.command()
def inference(
    data_path: Path = typer.Option("./data/CK-dataset", "-d", "--data_path"),
    input_path: Path = typer.Option("output/dgcnn-xyz", "-i", "--in_path"),
    features: str = typer.Option("xyz", "-f", "--features"),
) -> Path:

    # Load model
    checkpoint_path = os.path.join(input_path, "checkpoint")
    model_path = [
        filename for filename in os.listdir(checkpoint_path) if "last" not in filename
    ][0]

    if features == "xyz":
        classifier = XYZClassifier.load_from_checkpoint(
            os.path.join(checkpoint_path, model_path)
        )
    elif features == "hks":
        classifier = HKSClassifier.load_from_checkpoint(
            os.path.join(checkpoint_path, model_path)
        )
    else:
        raise NotImplementedError(
            f"Model for feature type '{features}' is not defined, choose from: [hks, xyz]."
        )

    classifier.eval()
    classifier.cuda()

    # Load split
    with Path("split.json").open() as file:
        split = json.load(file)

    modes = ["val"]

    for mode in tqdm(modes):
        ds = make_dataset(
            root_path=data_path,
            transforms=Compose(
                [
                    NormalizePointcloudd(["points"]),
                    ComputeNormalsd(["points"]),
                    # RandomRotationd(["points"], [2 * np.pi, 2 * np.pi, 2 * np.pi]),
                    ComputeHKSFeaturesd(["points"], "hks", 128, 16),
                    GraphToPyGData(x_key="hks"),
                ]
            ),
            people_names=split[mode],
        )
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=10)

        representations = []
        predictions = []

        for i, data in tqdm(enumerate(dl), total=len(dl)):
            mesh_act, representation, prediction = classifier(data.cuda())
            representation, prediction = (
                representation[0],
                prediction[0],
            )
            predicted_class = torch.argmax(prediction).item()

            representations.append(representation.detach().cpu().numpy())
            predictions.append(
                {"predicted": predicted_class, "true": data.y.int().item()}
            )

            # Create mesh & save
            edges = np.c_[
                2 * np.ones((len(data.edge_index.T), 1)),
                data.edge_index.T.detach().cpu().numpy(),
            ].flatten()

            mesh = pv.PolyData(
                data.pos.detach().cpu().numpy(), lines=edges.astype(int),
            )
            mesh["activation"] = mesh_act.detach().cpu().numpy()

            mesh_dir = os.path.join(input_path, f"{mode}_meshes")
            os.makedirs(mesh_dir, exist_ok=True)
            mesh.save(os.path.join(mesh_dir, f"mesh_{i}.vtk"))

        # Save model outputs
        np.save(
            os.path.join(input_path, f"{mode}_representations.npy"),
            np.array(representations),
        )
        pd.DataFrame(predictions).to_csv(
            os.path.join(input_path, f"{mode}_predictions.csv"), index=None
        )


if __name__ == "__main__":
    app()
