from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.fernet import Fernet


BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_DIR = BASE_DIR / "secrets"
SECRET_DIR.mkdir(parents=True, exist_ok=True)

FERNET_KEY_PATH = SECRET_DIR / "fernet.key"


def get_or_create_fernet_key() -> bytes:
    if FERNET_KEY_PATH.exists():
        return FERNET_KEY_PATH.read_bytes()

    key = Fernet.generate_key()
    FERNET_KEY_PATH.write_bytes(key)
    return key


def get_fernet() -> Fernet:
    return Fernet(get_or_create_fernet_key())


def encrypt_bytes(data: bytes) -> bytes:
    return get_fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    return get_fernet().decrypt(data)


def encrypt_text(text: str) -> bytes:
    return encrypt_bytes(text.encode("utf-8"))


def decrypt_text(data: bytes) -> str:
    return decrypt_bytes(data).decode("utf-8")


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt is None:
        salt = os.urandom(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120_000,
    )

    salt_b64 = base64.b64encode(salt).decode("utf-8")
    hash_b64 = base64.b64encode(password_hash).decode("utf-8")
    return salt_b64, hash_b64


def verify_password(password: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    expected_hash = base64.b64decode(hash_b64.encode("utf-8"))

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120_000,
    )

    return actual_hash == expected_hash