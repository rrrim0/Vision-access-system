from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QLineEdit, QPushButton, QWidget

try:
    from app.db import update_user_full_name
except Exception as exc:
    update_user_full_name = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from app.ui_theme import (
    AUTH_INPUT_BG, AUTH_LOGIN_BTN, AUTH_LOGIN_BTN_HOVER,
    BTN_BG, BTN_HOVER,
    CLOSEBTN_HOVER, CLOSEBTN_PRESSED,
    ICON_CLOSE, LINE_BG, TEXT_WHITE, WINDOW_BG,
    SUCCESS_BOTTOM_H, SUCCESS_BOTTOM_W, SUCCESS_CLOSE_BTN_H, SUCCESS_CLOSE_BTN_W,
    SUCCESS_CONTINUE_H, SUCCESS_DIALOG_H, SUCCESS_DIALOG_W,
    SUCCESS_HEADER_H, SUCCESS_HEADER_W,
    TitleBarButton,
    show_error, show_warning,
)


class EditUserDialog(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("editUserDialog")
        self.setFixedSize(SUCCESS_DIALOG_W, SUCCESS_DIALOG_H)

        self.user_id: int | None = None
        self.on_saved: Callable[[], None] | None = None

        self.header = QFrame(self)
        self.header.setObjectName("editUserHeader")
        self.header.setGeometry(1, 0, SUCCESS_HEADER_W, SUCCESS_HEADER_H)

        self.close_btn = TitleBarButton(
            ICON_CLOSE,
            hover_bg=CLOSEBTN_HOVER,
            pressed_bg=CLOSEBTN_PRESSED,
            parent=self.header,
        )
        self.close_btn.setGeometry(
            SUCCESS_HEADER_W - SUCCESS_CLOSE_BTN_W, 0,
            SUCCESS_CLOSE_BTN_W, SUCCESS_CLOSE_BTN_H,
        )

        self.title_label = QLabel("Изменение данных", self)
        self.title_label.setObjectName("editUserTitle")
        self.title_label.setGeometry(28, 56, SUCCESS_DIALOG_W - 56, 30)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.username_label = QLabel("Пользователь: —", self)
        self.username_label.setObjectName("editUserSmallText")
        self.username_label.setGeometry(28, 86, SUCCESS_DIALOG_W - 56, 24)
        self.username_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.fullname_input = QLineEdit(self)
        self.fullname_input.setObjectName("editUserInput")
        self.fullname_input.setGeometry(28, 124, SUCCESS_DIALOG_W - 56, 43)
        self.fullname_input.setPlaceholderText("Введите новое ФИО")

        self.bottom_panel = QFrame(self)
        self.bottom_panel.setObjectName("editUserBottom")
        self.bottom_panel.setGeometry(1, SUCCESS_DIALOG_H - SUCCESS_BOTTOM_H, SUCCESS_BOTTOM_W, SUCCESS_BOTTOM_H)

        self.cancel_btn = QPushButton("Отмена", self.bottom_panel)
        self.cancel_btn.setObjectName("editUserCancelButton")
        self.cancel_btn.setGeometry(SUCCESS_BOTTOM_W - 252, 12, 105, SUCCESS_CONTINUE_H)

        self.save_btn = QPushButton("Сохранить", self.bottom_panel)
        self.save_btn.setObjectName("editUserSaveButton")
        self.save_btn.setGeometry(SUCCESS_BOTTOM_W - 135, 12, 117, SUCCESS_CONTINUE_H)

        self.close_btn.clicked.connect(self.hide)
        self.cancel_btn.clicked.connect(self.hide)
        self.save_btn.clicked.connect(self.save_changes)
        self.fullname_input.returnPressed.connect(self.save_changes)

        self.setStyleSheet(
            f"""
            #editUserDialog {{
                background: {WINDOW_BG};
                border: 1px solid {LINE_BG};
                border-radius: 0px;
            }}
            #editUserHeader {{
                background: {WINDOW_BG};
                border-bottom: 1px solid {LINE_BG};
            }}
            #editUserBottom {{
                background: {WINDOW_BG};
                border-top: 1px solid {LINE_BG};
            }}
            #editUserTitle {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 20px;
                font-weight: 600;
            }}
            #editUserSmallText {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 400;
            }}
            #editUserInput {{
                background: {AUTH_INPUT_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 16px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: {AUTH_LOGIN_BTN};
            }}
            #editUserInput:focus {{ border: 1px solid {AUTH_LOGIN_BTN}; }}
            #editUserCancelButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #editUserCancelButton:hover {{ background: {BTN_HOVER}; }}
            #editUserSaveButton {{
                background: {AUTH_LOGIN_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #editUserSaveButton:hover {{ background: {AUTH_LOGIN_BTN_HOVER}; }}
            """
        )

        self.hide()

    def show_for_user(
        self,
        user_id: int,
        username: str,
        full_name: str,
        on_saved: Callable[[], None] | None = None,
    ) -> None:
        self.user_id = user_id
        self.on_saved = on_saved
        self.username_label.setText(f"Пользователь: {username}")
        self.fullname_input.setText(full_name)
        self.fullname_input.selectAll()
        self.show_centered()
        self.fullname_input.setFocus()

    def show_centered(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            x = (parent.width() - SUCCESS_DIALOG_W) // 2
            y = (parent.height() - SUCCESS_DIALOG_H) // 2
            self.move(x, y)
        self.show()
        self.raise_()

    def save_changes(self) -> None:
        if self.user_id is None:
            show_warning(self, "Ошибка", "Пользователь не выбран.")
            return
        new_full_name = self.fullname_input.text().strip()
        if not new_full_name:
            show_warning(self, "Ошибка", "Введите ФИО.")
            return
        if update_user_full_name is None:
            show_error(self, "Ошибка", "Функция изменения пользователя не загружена.")
            return
        try:
            update_user_full_name(self.user_id, new_full_name)
            if self.on_saved is not None:
                self.on_saved()
            self.hide()
        except Exception as exc:
            show_error(self, "Ошибка", f"Не удалось изменить пользователя:\n{exc}")