from pathlib import Path
from typing import Optional
import numpy as np
import pyvista as pv
import mediapipe as mp
import cv2
from pqdm.processes import pqdm


def main():
    data_path = Path("./.data")
    images_root_path = data_path / "./image"
    pointclouds_root_path = data_path / "./pointcloud"
    preview_root_path = data_path / "./preview"
    paths = list(data_path.glob("**/*.png"))

    args = [
        (
            image_path,
            pointclouds_root_path / image_path.relative_to(images_root_path).with_suffix(".vtk"),
            preview_root_path / image_path.relative_to(images_root_path),
        ) for image_path in paths
    ]
    pqdm(args, save_face_mesh, argument_type="args", n_jobs=8)


def save_face_mesh(face_path: Path, output_path: Path, preview_export_path: Optional[Path]) -> None:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    face_img = cv2.imread(str(face_path))
    mesh: pv.PolyData = face_to_mesh(face_img=face_img, preview_export_path=preview_export_path)
    mesh.save(output_path)


def face_to_mesh(face_img: np.ndarray, preview_export_path: Optional[Path]) -> pv.PolyData:
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            print("no detection!")
        [face_landmarks] = results.multi_face_landmarks
        export_preview(face_landmarks, face_img=face_img, preview_export_path=preview_export_path)

    nodes = np.stack(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in face_landmarks.landmark
        ]
    )
    poly = pv.PolyData(nodes)
    return poly


def export_preview(landmarks, face_img: np.ndarray, preview_export_path: Optional[Path]) -> None:
    if preview_export_path is None:
        return
    preview_export_path.parent.mkdir(exist_ok=True, parents=True)
    annotated_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 0, 255))
    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    cv2.imwrite(str(preview_export_path), annotated_image)


if __name__ == "__main__":
    main()
