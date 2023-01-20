from abc import abstractmethod
from datetime import datetime
from tkinter import Tk, Frame, Label, IntVar, StringVar, Entry, BooleanVar, Checkbutton
from typing import TextIO

import cv2
import numpy as np
import torch
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from run.export_meshes import face_to_mesh, NoFacesFoundException
from src.classifier import Classifier
from src.dataset.transforms import NormalizePointcloudd


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

    def __init__(
        self,
        model_path: str,
        log_file: TextIO,
        default_left: int = 0,
        default_right: int = 450,
        default_top: int = 750,
        default_bottom: int = 1080
    ):
        self.log_file = log_file
        self.target_resolution = (640, 360)

        self.net = Classifier.load_from_checkpoint(model_path).eval().cuda()
        self.transforms = NormalizePointcloudd(["points"])

        self.root = Tk()
        self.app = Frame(self.root)

        self.inputs_frame = Frame(self.app)
        self.inputs_frame.grid(column=0, row=0)

        self.images_frame = Frame(self.app)
        self.images_frame.grid(column=1, row=0)

        self.left_input_var = IntVar(self.inputs_frame, value=default_left)
        self.right_input_var = IntVar(self.inputs_frame, value=default_right)
        self.top_input_var = IntVar(self.inputs_frame, value=default_top)
        self.bottom_input_var = IntVar(self.inputs_frame, value=default_bottom)
        self.show_preview_var = BooleanVar(self.inputs_frame, value=True)
        self.output_var = StringVar(self.inputs_frame, value="No predictions made yet")

        Label(self.inputs_frame, text="left_margin").grid(column=0, row=0)
        Entry(self.inputs_frame, textvariable=self.left_input_var).grid(column=1, row=0)

        Label(self.inputs_frame, text="right_margin").grid(column=0, row=1)
        Entry(self.inputs_frame, textvariable=self.right_input_var).grid(column=1, row=1)

        Label(self.inputs_frame, text="top_margin").grid(column=0, row=2)
        Entry(self.inputs_frame, textvariable=self.top_input_var).grid(column=1, row=2)

        Label(self.inputs_frame, text="bottom_margin").grid(column=0, row=3)
        Entry(self.inputs_frame, textvariable=self.bottom_input_var).grid(column=1, row=3)

        Checkbutton(self.inputs_frame, variable=self.show_preview_var, text="Show preview:").grid(column=0, row=4)
        Label(self.inputs_frame, width=22, textvariable=self.output_var).grid(column=0, row=5)

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
        points: torch.Tensor = torch.from_numpy(face_mesh.points).float().cuda()
        edges: torch.Tensor = torch.from_numpy(face_mesh.lines.reshape(-1, 3)[:, 1:]).cuda()
        edge_index = to_undirected(edges.T.long())

        data_dict = {
            "points": points,
            "edges": edge_index
        }
        data_dict = self.transforms(data_dict)
        data = Data(pos=data_dict["points"], edge_index=data_dict["edges"])
        with torch.no_grad():
            pred = self.net(data)
            pred = torch.sigmoid(pred[0]).cpu()

        predicted_class = torch.argmax(pred)
        predicted_label = self.emotion_map[predicted_class.item()]
        self.output_var.set(f"predicted: {predicted_label}")

        # logging results to file
        self.log_file.write(",".join([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), *map(str, pred.tolist())]) + "\n")

        # drawing emotions histogram
        fig = plt.figure()
        xs = list(range(len(self.emotion_map)))
        ys = pred.tolist()
        label = list(self.emotion_map.values())
        plt.bar(xs, ys, tick_label=label)
        fig.canvas.draw()
        plot_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.display_image(plot_array, 2)
        fig.clear()
        del fig
        plt.close()

        self.app.after(1, self.update_stream)

    def display_image(self, frame: np.ndarray, image_index: int):
        if self.show_preview_var.get():
            pill_frame = Image.fromarray(frame)
            pill_frame = pill_frame.resize(self.target_resolution)
            tk_frame = ImageTk.PhotoImage(image=pill_frame)

            self.tk_image_frames[image_index].imgtk = tk_frame
            self.tk_image_frames[image_index].configure(image=tk_frame)

    @abstractmethod
    def frame_generator(self):
        pass
