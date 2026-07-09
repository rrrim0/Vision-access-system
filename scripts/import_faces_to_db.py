from pathlib import Path
import cv2

from app.db import init_db, add_user, add_face_template, get_user_by_username
from app.face_embedder import FaceEmbedder


FACES_DIR = Path("data") / "faces"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    if not FACES_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка: {FACES_DIR}")

    init_db()
    embedder = FaceEmbedder()

    user_dirs = [p for p in FACES_DIR.iterdir() if p.is_dir()]
    if not user_dirs:
        raise RuntimeError("В data/faces нет папок пользователей.")

    total_users = 0
    total_templates = 0

    for user_dir in user_dirs:
        username = user_dir.name

        existing_user = get_user_by_username(username)
        if existing_user is None:
            user_id = add_user(username=username, full_name=username, role="employee")
            print(f"[OK] Добавлен пользователь: {username} (id={user_id})")
            total_users += 1
        else:
            user_id = int(existing_user["id"])
            print(f"[INFO] Пользователь уже есть: {username} (id={user_id})")

        saved_for_user = 0

        for img_path in sorted(user_dir.iterdir()):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Не удалось прочитать изображение: {img_path}")
                continue

            emb, _bbox = embedder.get_embedding(img)
            if emb is None:
                print(f"[WARN] Не удалось извлечь embedding: {img_path.name}")
                continue

            add_face_template(
                user_id=user_id,
                embedding=emb,
                image_path=str(img_path)
            )
            saved_for_user += 1
            total_templates += 1

        print(f"[OK] {username}: сохранено шаблонов {saved_for_user}")

    print("\nИмпорт завершён.")
    print(f"Новых пользователей: {total_users}")
    print(f"Всего сохранено шаблонов: {total_templates}")


if __name__ == "__main__":
    main()