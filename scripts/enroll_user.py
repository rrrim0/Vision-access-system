from pathlib import Path
import time
import cv2

from app.camera import Camera
from app.face_detector import FaceDetector
from app.face_utils import crop_face, save_face_image

def main():
    person_name = input("Введите имя пользователя (латиницей или без пробелов): ").strip()
    if not person_name:
        raise ValueError("Имя пустое")

    out_dir = Path("data") / "faces" / person_name
    out_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera()
    detector = FaceDetector()

    saved = 0
    last_save_time = 0.0

    print("Управление: S — сохранить лицо, ESC — выход")
    try:
        while True:
            frame = camera.read()
            faces = detector.detect(frame)

            # Берём самое крупное лицо (обычно основное)
            face_bbox = None
            if len(faces) > 0:
                face_bbox = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = face_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"Saved: {saved}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Enroll User (S=save, ESC=exit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key in (ord('s'), ord('S')):
                # антидребезг: не чаще 2 раз в секунду
                now = time.time()
                if now - last_save_time < 0.5:
                    continue
                last_save_time = now

                if face_bbox is None:
                    print("Лицо не найдено — снимок не сохранён")
                    continue

                face = crop_face(frame, face_bbox, margin=0.2)
                filename = f"{saved:03d}.jpg"
                save_face_image(face, out_dir / filename, size=(160, 160))
                saved += 1
                print(f"Сохранено: {out_dir / filename}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Готово. Всего сохранено: {saved}. Папка: {out_dir}")

if __name__ == "__main__":
    main()