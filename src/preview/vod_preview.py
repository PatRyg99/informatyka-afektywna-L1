from typing import TextIO

import cv2

from src.preview.emotion_preview import EmotionPreview


class VODEmotionPreview(EmotionPreview):
    def __init__(self, model_path: str, log_file: TextIO, vod_path: str, skip_first_n: int, read_interval: int, default_left=0, default_right=-1, default_top=0, default_bottom=-1):
        super().__init__(model_path, log_file=log_file, default_left=default_left, default_right=default_right, default_top=default_top, default_bottom=default_bottom)
        self.vod_path = vod_path
        self.skip_first_n = skip_first_n
        self.read_interval = read_interval

    def frame_generator(self):
        cap = cv2.VideoCapture(self.vod_path)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        num_read_frames = 0
        while True:
            for _ in range(self.read_interval):
                ret, frame = cap.read()
                num_read_frames += 1
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                if cv2.waitKey(1) == ord('q'):
                    break
            if num_read_frames < self.skip_first_n:
                continue
            yield frame
        cap.release()
        cv2.destroyAllWindows()
