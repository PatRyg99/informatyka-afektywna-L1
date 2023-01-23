from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv
from pqdm.processes import pqdm


def main():
    data_path = Path("./data/CK-dataset")
    images_root_path = data_path / "./image"
    mesh_root_path = data_path / "./mesh"
    preview_root_path = data_path / "./preview"
    paths = list(data_path.glob("**/*.png"))

    args = [
        (
            image_path,
            mesh_root_path
            / image_path.relative_to(images_root_path).with_suffix(".vtk"),
            preview_root_path / image_path.relative_to(images_root_path),
        )
        for image_path in paths
    ]
    pqdm(args, save_face_mesh, argument_type="args", n_jobs=8)


def save_face_mesh(
    face_path: Path, output_path: Path, preview_export_path: Optional[Path]
) -> None:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    face_img = cv2.imread(str(face_path))
    mesh, preview_img = face_to_mesh(face_img=face_img)

    if preview_export_path:
        preview_export_path.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(preview_export_path), preview_img)

    mesh.save(output_path)


def face_to_mesh(face_img: np.ndarray) -> Tuple[pv.PolyData, np.ndarray]:
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise NoFacesFoundException()
        [face_landmarks] = results.multi_face_landmarks
    preview: np.ndarray = generate_preview(landmarks=face_landmarks, face_img=face_img)
    nodes = np.stack(
        [[landmark.x, landmark.y, landmark.z] for landmark in face_landmarks.landmark]
    )

    def faces_from_lines(edges: np.array):
        """Construct properly oriented faces from the list of edges"""
        faces = []
        cycles = []

        for i, (u1, v1) in enumerate(edges):
            mask = edges[:, 0] == v1

            next_idx = np.argwhere(mask).flatten()
            next_edges = edges[mask]

            for j, (u2, v2) in zip(next_idx, next_edges):
                mask = (edges[:, 0] == v2) & (edges[:, 1] == u1)

                if mask.sum() == 1:
                    last_idx = np.argwhere(mask).item()

                    if {i, j, last_idx} not in cycles:
                        faces.append((u1, v1, v2))
                        cycles.append({i, j, last_idx})

        return np.array(faces)

    edges = np.array(list(mp.solutions.face_mesh.FACEMESH_TESSELATION))
    faces = faces_from_lines(edges)

    edges = np.c_[2 * np.ones(len(edges))[:, None], edges].flatten().astype(int)
    faces = np.c_[3 * np.ones(len(faces))[:, None], faces].flatten().astype(int)

    poly = pv.PolyData(nodes, faces=faces, lines=edges)
    return poly, preview


def generate_preview(landmarks, face_img: np.ndarray) -> np.ndarray:
    annotated_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
        thickness=1, circle_radius=1, color=(0, 0, 255)
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    return annotated_image


class NoFacesFoundException(Exception):
    pass


if __name__ == "__main__":
    main()
