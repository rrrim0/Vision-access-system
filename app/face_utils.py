from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def crop_face(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.2) -> np.ndarray:
    x, y, w, h = bbox

    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame_bgr.shape[1], x + w + mx)
    y2 = min(frame_bgr.shape[0], y + h + my)

    return frame_bgr[y1:y2, x1:x2].copy()


def save_face_image(face_bgr: np.ndarray, out_path: str | Path, size: tuple[int, int] = (160, 160)) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resized = cv2.resize(face_bgr, size)
    ok = cv2.imwrite(str(out_path), resized)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить изображение: {out_path}")