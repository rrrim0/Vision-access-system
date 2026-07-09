from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACES_DIR = Path("data") / "faces"


class FaceEngine:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=True,
            device=DEVICE
        )

        self.embedder = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(DEVICE)

    def get_embedding(
        self,
        frame_bgr: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        det = self.mtcnn.detect(rgb, landmarks=True)
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

        faces = self.mtcnn.extract(rgb, [box], save_path=None)
        if faces is None or len(faces) == 0:
            return None, None

        face_tensor = faces[0].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = self.embedder(face_tensor)

        emb_np = emb[0].cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(emb_np)
        if norm > 0:
            emb_np = emb_np / norm

        x1, y1, x2, y2 = map(int, box)
        return emb_np, (x1, y1, x2, y2)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def build_templates(engine: FaceEngine) -> dict[str, np.ndarray]:
    if not FACES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {FACES_DIR}")

    templates: dict[str, np.ndarray] = {}

    user_dirs = [p for p in FACES_DIR.iterdir() if p.is_dir()]
    if not user_dirs:
        raise RuntimeError("В data/faces нет папок пользователей.")

    for user_dir in user_dirs:
        user_name = user_dir.name
        embs = []

        for img_path in sorted(user_dir.glob("*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            emb, _ = engine.get_embedding(img)
            if emb is not None:
                embs.append(emb)

        if len(embs) == 0:
            print(f"[WARN] Не удалось извлечь embedding для {user_name}")
            continue

        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        mean_norm = np.linalg.norm(mean_emb)
        if mean_norm > 0:
            mean_emb = mean_emb / mean_norm

        templates[user_name] = mean_emb
        print(f"[OK] {user_name}: шаблон по {len(embs)} изображениям")

    if not templates:
        raise RuntimeError("Не удалось построить шаблоны.")

    return templates


def main():
    engine = FaceEngine()
    templates = build_templates(engine)

    print("\nЗарегистрированные пользователи:")
    print(sorted(templates.keys()))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть камеру.")

    print("\nНажми:")
    print("  S - зафиксировать текущий кадр и вывести similarity")
    print("  Q - выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emb, box = engine.get_embedding(frame)

        display = frame.copy()

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            display,
            "Press S to inspect similarity",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Faces Similarity Debug", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key in (ord("s"), ord("S")):
            print("\n==============================")
            print("SIMILARITY CHECK")
            print("==============================")

            if emb is None:
                print("Лицо не найдено.")
                continue

            scores = []
            for user_name, template_emb in templates.items():
                score = cosine_similarity(emb, template_emb)
                scores.append((user_name, score))

            scores.sort(key=lambda x: x[1], reverse=True)

            print("Top matches:")
            for user_name, score in scores[:10]:
                print(f"  {user_name}: {score:.4f}")

            rim_score = None
            if "rim" in templates:
                rim_score = cosine_similarity(emb, templates["rim"])
                print(f"\nScore for rim: {rim_score:.4f}")

            best_name, best_score = scores[0]
            print(f"\nBest match: {best_name} ({best_score:.4f})")

            print("\nРекомендация по порогу:")
            if best_score >= 0.70:
                print("  Порог можно ставить около 0.65-0.70")
            elif best_score >= 0.60:
                print("  Порог можно ставить около 0.55-0.60")
            elif best_score >= 0.50:
                print("  Порог можно ставить около 0.45-0.50")
            else:
                print("  Порог нужно опускать ниже 0.45 или переснять фото в data/faces")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()