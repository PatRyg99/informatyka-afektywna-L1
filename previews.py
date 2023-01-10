from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import typer
from matplotlib import pyplot as plt
from torch_geometric.utils import to_undirected
from tkinter import Tk, Frame, Label, IntVar, Text, END, StringVar, Entry
from PIL import ImageTk, Image
import cv2

from src.classifier import Classifier
from export_meshes import face_to_mesh, NoFacesFoundException
from torch_geometric.data import Data
from simple_twitch_stream_receiver import SimpleTwitchStreamReceiver


class EmotionPreview:
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

    def __init__(self, model_path: str):
        self.target_resolution = (640, 360)

        self.net = Classifier.load_from_checkpoint(model_path).eval().cuda()

        self.root = Tk()
        self.app = Frame(self.root)

        self.inputs_frame = Frame(self.app)
        self.inputs_frame.grid(column=0, row=0)

        self.images_frame = Frame(self.app)
        self.images_frame.grid(column=1, row=0)

        self.left_input_var = IntVar(self.inputs_frame, value=0)
        self.right_input_var = IntVar(self.inputs_frame, value=450)
        self.top_input_var = IntVar(self.inputs_frame, value=750)
        self.bottom_input_var = IntVar(self.inputs_frame, value=1080)
        self.output_var = StringVar(self.inputs_frame, value="No predictions made yet")

        Label(self.inputs_frame, text="left_margin").grid(column=0, row=0)
        Entry(self.inputs_frame, textvariable=self.left_input_var).grid(column=1, row=0)

        Label(self.inputs_frame, text="right_margin").grid(column=0, row=1)
        Entry(self.inputs_frame, textvariable=self.right_input_var).grid(column=1, row=1)

        Label(self.inputs_frame, text="top_margin").grid(column=0, row=2)
        Entry(self.inputs_frame, textvariable=self.top_input_var).grid(column=1, row=2)

        Label(self.inputs_frame, text="bottom_margin").grid(column=0, row=3)
        Entry(self.inputs_frame, textvariable=self.bottom_input_var).grid(column=1, row=3)

        Label(self.inputs_frame, textvariable=self.output_var).grid(column=0, row=4)

        num_displayed_images = 4
        self.tk_image_frames = [Label(self.images_frame) for _ in range(num_displayed_images)]
        for image_index, tk_image_frame in enumerate(self.tk_image_frames):
            tk_image_frame.grid(column=image_index // 2, row=image_index % 2)

        self.app.pack()

        self.frame_iterator = iter(self.frame_generator())

    def run_gui(self):
        self.update_stream()
        self.root.mainloop()

    def update_stream(self):
        frame = next(self.frame_iterator)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            left, right, top, bottom = self.left_input_var.get(), self.right_input_var.get(), self.top_input_var.get(), self.bottom_input_var.get()
        except:
            self.output_var.set("Please input correct crop margins")
            self.app.after(1, self.update_stream)
            return

        frame_with_rect = cv2.rectangle(frame.copy(), (left, bottom), (right, top), (255, 0, 0), 10)
        self.display_image(frame_with_rect, 0)

        cropped_frame: np.ndarray = frame[top: bottom, left: right]
        if np.prod(cropped_frame.shape) == 0:
            self.output_var.set("Incorrect crop margins!")
            self.app.after(1, self.update_stream)
            return

        # getting face mesh
        try:
            face_mesh, preview = face_to_mesh(face_img=cropped_frame)
            preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.display_image(preview, 1)
        except NoFacesFoundException:
            self.output_var.set("No faces found!")
            self.display_image(cropped_frame, 1)
            self.app.after(1, self.update_stream)
            return

        # predicting emotion from model
        points: torch.Tensor = torch.from_numpy(face_mesh.points).cuda()
        edges: torch.Tensor = torch.from_numpy(face_mesh.lines.reshape(-1, 3)[:, 1:]).cuda()
        edge_index = to_undirected(edges.T.long())

        data = Data(pos=points.float(), edge_index=edge_index)
        pred = self.net(data)
        pred = torch.sigmoid(pred[0]).cpu()

        predicted_class = torch.argmax(pred)
        predicted_label = self.emotion_map[predicted_class.item()]
        self.output_var.set(f"predicted: {predicted_label}")

        # drawimg emotions histogram
        fig = plt.figure()
        xs = list(range(len(self.emotion_map)))
        ys = pred.tolist()
        label = list(self.emotion_map.values())
        plt.bar(xs, ys, tick_label=label)
        fig.canvas.draw()
        plot_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.display_image(plot_array, 2)

        self.app.after(1, self.update_stream)

    def display_image(self, frame: np.ndarray, image_index: int):
        pill_frame = Image.fromarray(frame)
        pill_frame = pill_frame.resize(self.target_resolution)
        tk_frame = ImageTk.PhotoImage(image=pill_frame)

        self.tk_image_frames[image_index].imgtk = tk_frame
        self.tk_image_frames[image_index].configure(image=tk_frame)

    @abstractmethod
    def frame_generator(self):
        pass


class TwitchEmotionPreview(EmotionPreview):
    def __init__(self, url: str, model_path: str):
        super().__init__(model_path)
        self.receiver = SimpleTwitchStreamReceiver(url, quality="best")

    def frame_generator(self):
        yield from self.receiver


class LiveEmotionPreview(EmotionPreview):
    def frame_generator(self):
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
            if cv2.waitKey(1) == ord('q'):
                break
            yield frame
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
