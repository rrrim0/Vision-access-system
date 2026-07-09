from collections import deque
from typing import Any, Optional

import cv2
import numpy as np
from facenet_pytorch import MTCNN

from app.anti_spoof.inference import AntiSpoofModel


WINDOW_SIZE = 10
MIN_CONFIDENCE = 0.80


def get_largest_face_box(boxes: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if boxes is None or len(boxes) == 0:
        return None

    largest_box: Optional[tuple[int, int, int, int]] = None
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


def clamp_box(
    box: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin: int = 20,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(frame_width, x2 + margin)
    y2 = min(frame_height, y2 + margin)

    return x1, y1, x2, y2


def main() -> None:
    anti_spoof = AntiSpoofModel()

    mtcnn = MTCNN(
        keep_all=True,
        device=anti_spoof.device,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть веб-камеру")

    predictions: deque[tuple[str, float]] = deque(maxlen=WINDOW_SIZE)

    print("Нажми 'q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр с камеры.")
            break

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detect_result: Any = mtcnn.detect(frame_rgb)
        boxes = detect_result[0]
        probs = detect_result[1] if len(detect_result) > 1 else None

        if boxes is not None and len(boxes) > 0:
            largest_box = get_largest_face_box(boxes)

            if largest_box is not None:
                frame_h, frame_w = frame.shape[:2]
                x1, y1, x2, y2 = clamp_box(largest_box, frame_w, frame_h, margin=20)

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size != 0:
                    label, confidence = anti_spoof.predict(face_crop)
                    predictions.append((label.lower(), confidence))

                    real_count = sum(
                        1
                        for lbl, conf in predictions
                        if lbl == "real" and conf >= MIN_CONFIDENCE
                    )
                    spoof_count = len(predictions) - real_count
                    avg_conf = sum(conf for _, conf in predictions) / len(predictions)

                    if real_count >= spoof_count:
                        final_label = "REAL"
                        color = (0, 255, 0)
                    else:
                        final_label = "SPOOF"
                        color = (0, 0, 255)

                    text = f"{final_label} {avg_conf:.2f}"

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        display_frame,
                        text,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        else:
            cv2.putText(
                display_frame,
                "Face not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Anti-Spoof Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()