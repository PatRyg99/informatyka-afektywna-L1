from typing import TextIO

from simple_twitch_stream_receiver import SimpleTwitchStreamReceiver

from src.preview.emotion_preview import EmotionPreview


class TwitchEmotionPreview(EmotionPreview):
    def __init__(self, url: str, model_path: str, log_file: TextIO):
        super().__init__(model_path, log_file=log_file)
        self.receiver = SimpleTwitchStreamReceiver(url, quality="best")

    def frame_generator(self):
        yield from self.receiver
