from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Any

from app.security import decrypt_bytes, decrypt_text, encrypt_bytes, encrypt_text, hash_password, verify_password


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "face_access.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                full_name_encrypted BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                photo_path_encrypted BLOB NOT NULL,
                embedding_encrypted BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

        conn.commit()

    ensure_default_admin()


def ensure_default_admin() -> None:
    if get_admin_by_username("admin") is None:
        add_admin("admin", "admin")


def add_admin(username: str, password: str) -> int:
    salt_b64, hash_b64 = hash_password(password)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO admins (username, password_salt, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, salt_b64, hash_b64),
        )
        conn.commit()

        admin_id = cursor.lastrowid
        if admin_id is None:
            raise RuntimeError("Не удалось получить id администратора.")
        return admin_id


def get_admin_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, created_at FROM admins WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def verify_admin_credentials(username: str, password: str) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT password_salt, password_hash FROM admins WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row is None:
            return False
        return verify_password(password, row["password_salt"], row["password_hash"])


def add_user(username: str, full_name: str) -> int:
    encrypted_name = encrypt_text(full_name)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, full_name_encrypted) VALUES (?, ?)",
            (username, encrypted_name),
        )
        conn.commit()

        user_id = cursor.lastrowid
        if user_id is None:
            raise RuntimeError("Не удалось получить id пользователя.")
        return user_id


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, full_name_encrypted, created_at FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "full_name": decrypt_text(row["full_name_encrypted"]),
            "created_at": row["created_at"],
        }


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, full_name_encrypted, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "full_name": decrypt_text(row["full_name_encrypted"]),
            "created_at": row["created_at"],
        }


def update_user_full_name(user_id: int, new_full_name: str) -> None:
    encrypted_name = encrypt_text(new_full_name)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET full_name_encrypted = ? WHERE id = ?",
            (encrypted_name, user_id),
        )
        conn.commit()


def get_all_users() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name_encrypted, created_at FROM users ORDER BY id ASC")
        rows = cursor.fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            result.append({
                "id": row["id"],
                "username": row["username"],
                "full_name": decrypt_text(row["full_name_encrypted"]),
                "created_at": row["created_at"],
            })
        return result


def add_face_template(user_id: int, photo_path: str, embedding: Any) -> int:
    encrypted_path = encrypt_text(photo_path)
    embedding_blob = pickle.dumps(embedding)
    encrypted_embedding = encrypt_bytes(embedding_blob)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO face_templates (user_id, photo_path_encrypted, embedding_encrypted)
            VALUES (?, ?, ?)
            """,
            (user_id, encrypted_path, encrypted_embedding),
        )
        conn.commit()

        template_id = cursor.lastrowid
        if template_id is None:
            raise RuntimeError("Не удалось получить id шаблона.")
        return template_id


def get_templates_database() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                u.id AS user_id,
                u.username,
                u.full_name_encrypted,
                ft.id AS template_id,
                ft.photo_path_encrypted,
                ft.embedding_encrypted,
                ft.created_at
            FROM users u
            JOIN face_templates ft ON u.id = ft.user_id
            ORDER BY u.id ASC, ft.id ASC
        """)
        rows = cursor.fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            embedding_blob = decrypt_bytes(row["embedding_encrypted"])
            embedding = pickle.loads(embedding_blob)

            result.append({
                "user_id": row["user_id"],
                "username": row["username"],
                "full_name": decrypt_text(row["full_name_encrypted"]),
                "template_id": row["template_id"],
                "photo_path": decrypt_text(row["photo_path_encrypted"]),
                "embedding": embedding,
                "created_at": row["created_at"],
            })
        return result


def get_face_template_paths_by_user_id(user_id: int) -> list[str]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT photo_path_encrypted FROM face_templates WHERE user_id = ?",
            (user_id,),
        )
        rows = cursor.fetchall()
        return [decrypt_text(row["photo_path_encrypted"]) for row in rows]


def delete_user(user_id: int) -> None:
    photo_paths = get_face_template_paths_by_user_id(user_id)

    for path_str in photo_paths:
        path = Path(path_str)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()