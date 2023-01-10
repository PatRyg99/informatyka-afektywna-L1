from typing import TextIO

import cv2

from src.preview.emotion_preview import EmotionPreview


class CameraEmotionPreview(EmotionPreview):
    def __init__(self, model_path: str, log_file: TextIO):
        super().__init__(model_path, log_file=log_file, default_left=0, default_right=-1, default_top=0, default_bottom=-1)

    def frame_generator(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if cv2.waitKey(1) == ord('q'):
                break
            yield frame
        cap.release()
        cv2.destroyAllWindows()
