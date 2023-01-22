from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.metrics import f1_score


def compute_PCA(x: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    return pca.fit_transform(x)


def compute_MDS(x: np.ndarray) -> np.ndarray:
    mds = MDS(n_components=2)
    return mds.fit_transform(x)


def compute_Isomap(x: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
    return isomap.fit_transform(x)


def compute_tSNE(x: np.ndarray, perplexity: int = 50) -> np.ndarray:
    tsne = TSNE(
        n_components=2, perplexity=perplexity, learning_rate="auto", init="random"
    )
    return tsne.fit_transform(x)


def plot_representation_projection(
    representations: np.array,
    predictions: pd.DataFrame,
    emotions_labels: List[str],
    model_name: str,
):
    """Plotting 2D projections of learnt representation space"""

    methods = [
        ("MDS", compute_MDS),
        ("Isomap", compute_Isomap),
        ("t-SNE", compute_tSNE),
    ]

    f1_macro = f1_score(predictions["true"], predictions["predicted"], average="macro")

    fig, axes = plt.subplots(ncols=len(methods), figsize=(20, 6), sharey=True)
    fig.suptitle(
        f"Representation for {model_name}: F1-macro={f1_macro:.2f}", fontsize=20
    )

    markers = ["o", "^"]
    cmap = cm.get_cmap("tab10", len(emotions_labels))

    for i, (ax, (name, compute_fn)) in enumerate(zip(axes.flatten(), methods)):
        z = compute_fn(x=representations)
        correct_mask = predictions["true"] == predictions["predicted"]

        for mask_flag, marker in zip([True, False], markers):
            mask = correct_mask == mask_flag

            ax.scatter(
                z[:, 0][mask],
                z[:, 1][mask],
                c=predictions["true"].to_numpy()[mask],
                marker=marker,
                cmap=cmap,
                s=45,
                linewidths=1.0,
                edgecolors=(0, 0, 0),
            )

        ax.set_title(name)
        ax.grid(zorder=-1)

        # Prepare legend
        if i == 0:

            def f(m, c):
                plt.plot([], [], marker=m, color=c, ls="none")[0]

            handles = [f("s", color) for color in cmap.colors]
            handles += [f(marker, "k") for marker in markers]

            labels = emotions_labels + ["correct", "incorrect"]

            ax.legend(
                handles=handles, labels=labels, title="emotions",
            )

    fig.tight_layout()
    plt.show()


def plot_rotation_ablation(model_name: str):
    """Plot ablation of invariance"""

    emotions_labels = [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "sadness",
        "surprise",
    ]

    representations = np.load(f"output/{model_name}/val_representations.npy")
    representations_rot = np.load(f"output/{model_name}/val_representations_rot.npy")

    predictions = pd.read_csv(f"output/{model_name}/val_predictions.csv")
    predictions_rot = pd.read_csv(f"output/{model_name}/val_predictions_rot.csv")

    plot_representation_projection(
        representations, predictions, emotions_labels, model_name=model_name
    )
    plot_representation_projection(
        representations_rot,
        predictions_rot,
        emotions_labels,
        model_name=model_name + " + rotations",
    )
