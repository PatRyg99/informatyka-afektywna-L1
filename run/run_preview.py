from pathlib import Path

import typer

from src.preview.camera_preview import CameraEmotionPreview
from src.preview.twitch_preview import TwitchEmotionPreview

app = typer.Typer()


@app.command()
def twitch(
    url: str = typer.Option("https://www.twitch.tv/grandpoobear", "-u", "--url"),
    model_path: str = typer.Option("./pretrained_models/dgcnn.ckpt", "-m", "--model"),
) -> None:
    log_path = Path("../logs") / f"{url.split('/')[-1]}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, mode="a") as log_file:
        preview = TwitchEmotionPreview(url, model_path, log_file=log_file)
        preview.run_gui()


@app.command()
def camera(
    model_path: str = typer.Option("./pretrained_models/dgcnn.ckpt", "-m", "--model"),
) -> None:
    log_path = Path("../logs") / "camera.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, mode="a") as log_file:
        preview = CameraEmotionPreview(model_path, log_file=log_file)
        preview.run_gui()


if __name__ == "__main__":
    app()
