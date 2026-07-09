from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np


def resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        return Path(cast(str, getattr(sys, "_MEIPASS"))) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (60, 60),
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

        cascade_path = resource_path("models/haarcascade_frontalface_default.xml")

        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"Не удалось загрузить Haar cascade: {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        if faces is None or len(faces) == 0:
            return []

        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]