import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from src.classifier import Classifier
from export_meshes import face_to_mesh, NoFacesFoundException

emotion_map: dict[int, str] = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise",
}


def main():
    net = Classifier.load_from_checkpoint("output/01-09-2023.16:55:41/checkpoint/last.ckpt").eval().cuda()
    for frame in camera_frame_generator():
        process_frame(net, frame)


def process_frame(net: nn.Module, frame: np.ndarray):
    try:
        face_mesh, preview = face_to_mesh(face_img=frame)
    except NoFacesFoundException:
        frame = write_on_image(frame, "no faces found")
        cv2.imshow('face_landmarks', frame)
        return

    face_points = face_mesh.points
    face_points = (face_points - face_points.min()) / (face_points.max() - face_points.min()) - 0.5
    vertices: torch.Tensor = torch.from_numpy(face_points).to(dtype=torch.float32)[None, ...].cuda()
    pred = torch.sigmoid(net(vertices)[0]).cpu()
    predicted_class = torch.argmax(pred)
    predicted_label = emotion_map[predicted_class.item()]
    preview = write_on_image(preview, predicted_label)
    cv2.imshow('face_landmarks', preview)

    fig = plt.figure()
    xs = list(range(len(emotion_map)))
    ys = pred.tolist()
    label = list(emotion_map.values())
    plt.bar(xs, ys, tick_label=label)
    fig.canvas.draw()
    plot_array = np.fromstring(
        fig.canvas.tostring_rgb(), dtype=np.uint8,
        sep=''
    )
    plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow('probs', plot_array)
    plt.cla()


def write_on_image(img, text: str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 100)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2
    img = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
    return img


def camera_frame_generator():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if cv2.waitKey(1) == ord('q'):
            break
        yield frame
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
