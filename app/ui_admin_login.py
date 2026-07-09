from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QPushButton, QWidget

try:
    from app.db import verify_admin_credentials
except Exception as exc:
    verify_admin_credentials = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from app.ui_theme import (
    AUTH_FIELD_H, AUTH_FIELD_W, AUTH_H, AUTH_W,
    TitleBarButton,
    CLOSEBTN_HOVER, CLOSEBTN_PRESSED,
    ICON_CLOSE,
    show_error, show_warning,
)


class AdminAuthDialog(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("adminAuthDialog")
        self.setFixedSize(AUTH_W, AUTH_H)

        self.title_label = QLabel("Вход администратора", self)
        self.title_label.setObjectName("authTitle")
        self.title_label.setGeometry(0, 34, AUTH_W, 45)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.login_label = QLabel("Логин", self)
        self.login_label.setObjectName("authLabel")
        self.login_label.setGeometry(32, 105, AUTH_FIELD_W, 22)

        self.login_input = QLineEdit(self)
        self.login_input.setObjectName("authInput")
        self.login_input.setGeometry(32, 132, AUTH_FIELD_W, AUTH_FIELD_H)

        self.password_label = QLabel("Пароль", self)
        self.password_label.setObjectName("authLabel")
        self.password_label.setGeometry(32, 200, AUTH_FIELD_W, 22)

        self.password_input = QLineEdit(self)
        self.password_input.setObjectName("authInput")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setGeometry(32, 227, AUTH_FIELD_W, AUTH_FIELD_H)

        self.login_btn = QPushButton("Войти", self)
        self.login_btn.setObjectName("authLoginButton")
        self.login_btn.setGeometry(32, 322, AUTH_FIELD_W, AUTH_FIELD_H)

        self.cancel_btn = QPushButton("Отмена", self)
        self.cancel_btn.setObjectName("authCancelButton")
        self.cancel_btn.setGeometry(32, 385, AUTH_FIELD_W, AUTH_FIELD_H)

        self.cancel_btn.clicked.connect(self.hide)
        self.login_btn.clicked.connect(self.check_auth)
        self.password_input.returnPressed.connect(self.check_auth)
        self.login_input.returnPressed.connect(self.password_input.setFocus)

        self.hide()

    def show_centered(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            x = (parent.width() - AUTH_W) // 2
            y = (parent.height() - AUTH_H) // 2
            self.move(x, y)

        self.login_input.clear()
        self.password_input.clear()
        self.show()
        self.raise_()
        self.login_input.setFocus()

    def check_auth(self) -> None:
        login = self.login_input.text().strip()
        password = self.password_input.text()

        if verify_admin_credentials is None:
            show_error(self, "Ошибка", f"Модуль авторизации не загружен:\n{IMPORT_ERROR}")
            return

        if verify_admin_credentials(login, password):
            self.hide()
            terminal = self.window()
            open_admin_mode = getattr(terminal, "open_admin_mode", None)
            if callable(open_admin_mode):
                open_admin_mode()
            return

        self.password_input.clear()
        self.password_input.setFocus()
        show_warning(self, "Ошибка входа", "Неверный логин или пароль.")