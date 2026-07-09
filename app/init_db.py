from pathlib import Path

from app.db import DB_PATH, get_admin_by_username, init_db


def main() -> None:
    print("Инициализация базы данных...")
    init_db()
    print(f"База данных готова: {Path(DB_PATH).resolve()}")

    admin = get_admin_by_username("admin")
    if admin is not None:
        print("Администратор создан или уже существует:")
        print("  login: admin")
        print("  password: admin")
    else:
        print("Не удалось проверить создание администратора.")


if __name__ == "__main__":
    main()