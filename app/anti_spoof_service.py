from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np
from facenet_pytorch import MTCNN

from app.anti_spoof_model import AntiSpoofModel


class AntiSpoofService:
    def __init__(
        self,
        enabled: bool = True,
        window_size: int = 10,
        min_confidence: float = 0.80,
        margin: int = 20,
    ):
        self.enabled = enabled
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.margin = margin

        self.model: AntiSpoofModel | None = None
        self.mtcnn: MTCNN | None = None
        self.initialization_error: str | None = None
        self.predictions: deque[tuple[str, float]] = deque(maxlen=self.window_size)

        if self.enabled:
            try:
                self.model = AntiSpoofModel()
                self.mtcnn = MTCNN(
                    keep_all=True,
                    device=self.model.device,
                )
            except Exception as exc:
                self.initialization_error = str(exc)
                self.enabled = False
                raise RuntimeError(f"Anti-spoof не загрузился: {exc}")

    @staticmethod
    def _get_largest_face_box(boxes: np.ndarray | None) -> tuple[int, int, int, int] | None:
        if boxes is None or len(boxes) == 0:
            return None

        largest_box: tuple[int, int, int, int] | None = None
        largest_area = 0.0

        for box in boxes:
            x1, y1, x2, y2 = box
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            area = w * h

            if area > largest_area:
                largest_area = area
                largest_box = (int(x1), int(y1), int(x2), int(y2))

        return largest_box

    def _clamp_box(
        self,
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box

        x1 = max(0, x1 - self.margin)
        y1 = max(0, y1 - self.margin)
        x2 = min(frame_width, x2 + self.margin)
        y2 = min(frame_height, y2 + self.margin)

        return x1, y1, x2, y2

    def reset(self) -> None:
        self.predictions.clear()

    def check_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "ok": True,
                "is_live": True,
                "score": None,
                "label": None,
                "raw_label": None,
                "raw_score": None,
                "reason": "anti_spoof_disabled",
                "message": self.initialization_error or "Anti-spoof отключен",
            }

        if self.model is None or self.mtcnn is None:
            return {
                "enabled": True,
                "ok": False,
                "is_live": False,
                "score": None,
                "label": None,
                "raw_label": None,
                "raw_score": None,
                "reason": "anti_spoof_model_not_loaded",
                "message": "Модель anti-spoof или MTCNN не загружены",
            }

        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detect_result = self.mtcnn.detect(frame_rgb)
            boxes = detect_result[0]

            if boxes is None or len(boxes) == 0:
                self.reset()
                return {
                    "enabled": True,
                    "ok": False,
                    "is_live": False,
                    "score": None,
                    "label": None,
                    "raw_label": None,
                    "raw_score": None,
                    "reason": "face_not_found_for_antispoof",
                    "message": "Лицо для anti-spoof не найдено",
                }

            largest_box = self._get_largest_face_box(boxes)
            if largest_box is None:
                self.reset()
                return {
                    "enabled": True,
                    "ok": False,
                    "is_live": False,
                    "score": None,
                    "label": None,
                    "raw_label": None,
                    "raw_score": None,
                    "reason": "face_not_found_for_antispoof",
                    "message": "Не удалось выбрать лицо для anti-spoof",
                }

            frame_h, frame_w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = self._clamp_box(largest_box, frame_w, frame_h)

            if x2 <= x1 or y2 <= y1:
                self.reset()
                return {
                    "enabled": True,
                    "ok": False,
                    "is_live": False,
                    "score": None,
                    "label": None,
                    "raw_label": None,
                    "raw_score": None,
                    "reason": "invalid_face_box",
                    "message": "Некорректная область лица для anti-spoof",
                }

            face_crop = frame_bgr[y1:y2, x1:x2]
            cv2.imwrite("debug_crop.jpg", face_crop)
            if face_crop.size == 0:
                self.reset()
                return {
                    "enabled": True,
                    "ok": False,
                    "is_live": False,
                    "score": None,
                    "label": None,
                    "raw_label": None,
                    "raw_score": None,
                    "reason": "empty_face_crop",
                    "message": "Пустой crop лица для anti-spoof",
                }

            raw_label, raw_confidence = self.model.predict(face_crop)
            raw_label_norm = raw_label.strip().lower()
            raw_confidence = float(raw_confidence)

            self.predictions.append((raw_label_norm, raw_confidence))

            real_count = sum(
                1
                for lbl, conf in self.predictions
                if self.model.is_live_label(lbl) and conf >= self.min_confidence
            )
            spoof_count = len(self.predictions) - real_count
            avg_conf = sum(conf for _, conf in self.predictions) / len(self.predictions)

            is_live = real_count >= spoof_count
            final_label = "real" if is_live else "spoof"

            return {
                "enabled": True,
                "ok": is_live,
                "is_live": is_live,
                "score": float(avg_conf),
                "label": final_label,
                "raw_label": raw_label_norm,
                "raw_score": raw_confidence,
                "reason": "live" if is_live else "spoof_detected",
                "message": (
                    f"window={len(self.predictions)}/{self.window_size}, "
                    f"real_count={real_count}, spoof_count={spoof_count}, "
                    f"raw={raw_label_norm}:{raw_confidence:.4f}, avg={avg_conf:.4f}"
                ),
            }

        except Exception as exc:
            self.reset()
            return {
                "enabled": True,
                "ok": False,
                "is_live": False,
                "score": None,
                "label": None,
                "raw_label": None,
                "raw_score": None,
                "reason": "anti_spoof_inference_error",
                "message": str(exc),
            }