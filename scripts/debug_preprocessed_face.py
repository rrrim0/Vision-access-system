from pathlib import Path

import cv2
import numpy as np
import torch

from app.camera import Camera
from app.face_embedder import FaceEmbedder


def tensor_to_bgr_image(face_tensor: torch.Tensor) -> np.ndarray:
    """
    Преобразует tensor лица формата (3, H, W) в обычное BGR-изображение OpenCV.
    """
    img = face_tensor.detach().cpu().permute(1, 2, 0).numpy()  # HWC, RGB
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def apply_preprocessing_steps(frame_bgr: np.ndarray):
    """
    Возвращает:
      - blurred_bgr
      - clahe_bgr
      - clahe_rgb
    """
    # 1. Устранение шумов
    blurred_bgr = cv2.GaussianBlur(frame_bgr, (3, 3), 0)

    # 2. Нормализация освещения через CLAHE по каналу яркости LAB
    lab = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab_equalized = cv2.merge((l_channel, a_channel, b_channel))
    clahe_bgr = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    # 3. Преобразование цветового пространства
    clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

    return blurred_bgr, clahe_bgr, clahe_rgb


def get_face_160(embedder: FaceEmbedder, rgb_image: np.ndarray):
    """
    Возвращает:
      - preprocessed_face_bgr (160x160) или None
      - bbox или None
    """
    det = embedder.mtcnn.detect(rgb_image, landmarks=True)
    if det is None:
        return None, None

    if isinstance(det, tuple) and len(det) == 2:
        boxes, _probs = det
    else:
        boxes, _probs, _landmarks = det

    if boxes is None or len(boxes) == 0:
        return None, None

    # Берем самое крупное лицо
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    box = boxes[idx]

    faces = embedder.mtcnn.extract(rgb_image, [box], save_path=None)
    if faces is None or len(faces) == 0:
        return None, None

    face_bgr = tensor_to_bgr_image(faces[0])
    x1, y1, x2, y2 = map(int, box)
    return face_bgr, (x1, y1, x2, y2)


def make_preview_row(images, labels):
    """
    Склеивает несколько изображений в одну строку с подписями.
    """
    prepared = []
    target_h = 240

    for img, label in zip(images, labels):
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized = cv2.resize(img, (new_w, target_h))

        cv2.putText(
            resized,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )
        prepared.append(resized)

    if not prepared:
        return None

    return cv2.hconcat(prepared)


def main():
    out_dir = Path("debug_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera()
    embedder = FaceEmbedder()

    saved_count = 0

    print("Управление:")
    print("  S - сохранить текущий коллаж")
    print("  ESC - выход")

    try:
        while True:
            frame = camera.read()
            frame_vis = frame.copy()

            blurred_bgr, clahe_bgr, clahe_rgb = apply_preprocessing_steps(frame)

            face_160_bgr, bbox = get_face_160(embedder, clahe_rgb)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(clahe_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Для итогового лица делаем тот же размер по высоте, чтобы был виден в коллаже
            face_panel = None
            if face_160_bgr is not None:
                face_panel = cv2.resize(face_160_bgr, (240, 240))
                cv2.putText(
                    face_panel,
                    "Face 160x160",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

            preview = make_preview_row(
                [frame_vis, blurred_bgr, clahe_bgr, face_panel],
                ["Original", "GaussianBlur", "CLAHE", "Result"]
            )

            if preview is not None:
                cv2.imshow("Preprocessing debug", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key in (ord("s"), ord("S")):
                if preview is None:
                    print("Нечего сохранять.")
                    continue

                out_path = out_dir / f"preprocessing_preview_{saved_count:03d}.png"
                cv2.imwrite(str(out_path), preview)
                saved_count += 1
                print(f"Сохранено: {out_path}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()