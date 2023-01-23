import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import typer
from pqdm.threads import pqdm
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

app = typer.Typer()


def project_vertice_features(features: np.array):
    pca = PCA(n_components=3)
    return pca.fit_transform(features)


@app.command()
def extract_cams(
    in_path: Path = typer.Option("output/dgcnn-hks", "-i", "--in-path"),
) -> Path:

    # Load labels and mesh paths
    labels = pd.read_csv(os.path.join(in_path, "val_predictions.csv"))[
        "true"
    ].to_numpy()

    meshes = np.array(
        [
            pv.read(os.path.join(in_path, "val_meshes", filename))
            for filename in sorted(os.listdir(os.path.join(in_path, "val_meshes")))
        ]
    )

    # Prepare emotion mapping and mesh list
    emotion_mapper = {
        1: "anger",
        2: "contempt",
        3: "disgust",
        4: "fear",
        5: "happy",
        6: "sadness",
        7: "surprise",
    }

    out_path = os.path.join(in_path, "emotion_meshes")
    os.makedirs(out_path, exist_ok=True)

    # Project activations per vertice
    all_acts = np.array([mesh["activation"] for mesh in meshes])
    all_projected_acts = np.array(
        pqdm(all_acts.transpose(1, 0, 2), project_vertice_features, n_jobs=10)
    )

    # Aggregate meshes for emotions
    for emotion_id, emotion_name in tqdm(emotion_mapper.items()):
        emotion_mask = labels == emotion_id
        emotion_meshes = meshes[emotion_mask]

        coords = np.array([mesh.points for mesh in emotion_meshes]).mean(axis=0)
        acts = all_projected_acts[:, emotion_mask].mean(axis=1)

        lines = emotion_meshes[0].lines
        faces = emotion_meshes[0].faces

        emotion_mesh = pv.PolyData(coords, lines=lines, faces=faces)
        emotion_mesh["activation"] = acts

        emotion_mesh.save(os.path.join(out_path, f"{emotion_name}_mesh.vtk"))


if __name__ == "__main__":
    app()
