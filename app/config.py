from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Камера
CAMERA_INDEX = 0
USE_DSHOW = True

# Логи
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Данные/БД (пока заглушка, позже подключим SQLite)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = DATA_DIR / "database.db"
