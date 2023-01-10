import typer

from previews import TwitchEmotionPreview, LiveEmotionPreview

app = typer.Typer()


@app.command()
def twitch(
    url: str = typer.Option("https://www.twitch.tv/grandpoobear", "-u", "--url"),
    model_path: str = typer.Option("./pretrained_models/dgcnn_but_it_works.ckpt", "-m", "--model"),
) -> None:
    preview = TwitchEmotionPreview(url, model_path)
    preview.run_gui()


@app.command()
def camera(
    model_path: str = typer.Option("./pretrained_models/dgcnn_but_it_works.ckpt", "-m", "--model"),
) -> None:
    preview = LiveEmotionPreview(model_path)
    preview.run_gui()


if __name__ == "__main__":
    app()
