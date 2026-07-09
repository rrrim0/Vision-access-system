from pathlib import Path
import cv2

from app.face_detector import FaceDetector
from app.face_utils import crop_face, save_face_image


VIDEO_DIR = Path("data") / "twins_videos_raw"
OUT_DIR = Path("data") / "id_images" / "twins"

VIDEO_MAP = {
    "twinA": VIDEO_DIR / "twinA.mp4",
    "twinB": VIDEO_DIR / "twinB.mp4",
}

# Настройки извлечения
EVERY_N_FRAMES = 5          # сохранять лицо с каждого 5-го кадра
MAX_IMAGES_PER_PERSON = 250 # максимум изображений на одного человека
MIN_FACE_SIZE = 80          # минимальный размер лица по ширине/высоте


def process_video(person_name: str, video_path: Path, detector: FaceDetector):
    if not video_path.exists():
        print(f"[ERROR] Видео не найдено: {video_path}")
        return 0

    out_dir = OUT_DIR / person_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        return 0

    saved = 0
    frame_idx = 0

    print(f"\n[{person_name}] Обработка видео: {video_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        if frame_idx % EVERY_N_FRAMES != 0:
            continue

        faces = detector.detect(frame)
        if len(faces) == 0:
            continue

        # Берём самое крупное лицо
        bbox = max(faces, key=lambda b: b[2] * b[3])
        x, y, w, h = bbox

        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue

        face = crop_face(frame, bbox, margin=0.2)

        filename = f"{saved:04d}.jpg"
        save_face_image(face, out_dir / filename, size=(160, 160))
        saved += 1

        if saved % 25 == 0:
            print(f"[{person_name}] Сохранено: {saved}")

        if saved >= MAX_IMAGES_PER_PERSON:
            print(f"[{person_name}] Достигнут лимит {MAX_IMAGES_PER_PERSON}")
            break

    cap.release()
    print(f"[{person_name}] Готово. Всего сохранено: {saved}")
    return saved


def main():
    detector = FaceDetector()

    total = 0
    for person_name, video_path in VIDEO_MAP.items():
        total += process_video(person_name, video_path, detector)

    print(f"\nИтог: всего сохранено {total} изображений.")
    print(f"Папка с результатами: {OUT_DIR}")


if __name__ == "__main__":
    main()