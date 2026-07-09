from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.face_detector import FaceDetector
from app.face_embedder import FaceEmbedder
from app.face_utils import crop_face, save_face_image


BASE_DIR = Path(__file__).resolve().parent.parent
FACES_DIR = BASE_DIR / "data" / "faces"
FACES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_user_dir(username: str) -> Path:
    user_dir = FACES_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def detect_largest_face(frame_bgr: np.ndarray, detector: FaceDetector) -> tuple[int, int, int, int] | None:
    faces = detector.detect(frame_bgr)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda b: b[2] * b[3])


def crop_largest_face(frame_bgr: np.ndarray, detector: FaceDetector, margin: float = 0.2) -> np.ndarray | None:
    bbox = detect_largest_face(frame_bgr, detector)
    if bbox is None:
        return None
    return crop_face(frame_bgr, bbox, margin=margin)


def save_face_sample(
    username: str,
    face_bgr: np.ndarray,
    embedder: FaceEmbedder,
    sample_index: int,
) -> tuple[str, np.ndarray] | None:
    user_dir = ensure_user_dir(username)
    save_path = user_dir / f"{sample_index:03d}.jpg"

    save_face_image(face_bgr, save_path, size=(160, 160))

    saved_face = cv2.imread(str(save_path))
    if saved_face is None:
        return None

    embedding, _ = embedder.get_embedding(saved_face)
    if embedding is None:
        return None

    return str(save_path), embedding


def process_uploaded_files(
    username: str,
    file_paths: list[str],
    detector: FaceDetector,
    embedder: FaceEmbedder,
    start_index: int,
    limit: int,
) -> list[tuple[str, np.ndarray]]:
    results: list[tuple[str, np.ndarray]] = []

    for path in file_paths:
        if len(results) >= limit:
            break

        image = cv2.imread(path)
        if image is None:
            continue

        face = crop_largest_face(image, detector=detector, margin=0.2)
        if face is None:
            continue

        saved = save_face_sample(
            username=username,
            face_bgr=face,
            embedder=embedder,
            sample_index=start_index + len(results),
        )
        if saved is None:
            continue

        results.append(saved)

    return results


def build_embedding_from_frame(frame_bgr: np.ndarray, detector: FaceDetector, embedder: FaceEmbedder) -> np.ndarray | None:
    face = crop_largest_face(frame_bgr, detector=detector, margin=0.2)
    if face is None:
        return None

    embedding, _ = embedder.get_embedding(face)
    return embedding