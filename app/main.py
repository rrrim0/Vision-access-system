from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from app.db import init_db
from app.ui_terminal_mode import TerminalWindow


def resource_path(relative_path: str) -> Path:
    if getattr(sys, "frozen", False):
        base_path = Path(cast(str, getattr(sys, "_MEIPASS")))
    else:
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path


def main() -> int:
    init_db()

    app = QApplication(sys.argv)

    icon_path = resource_path("assets/app_icon.ico")
    app.setWindowIcon(QIcon(str(icon_path)))

    window = TerminalWindow()
    window.setWindowIcon(QIcon(str(icon_path)))
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())