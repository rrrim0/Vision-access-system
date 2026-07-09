from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QFrame, QLabel, QMainWindow, QPushButton, QWidget

from app.ui_theme import (
    ADMIN_BLUE, ADMIN_BTN_H, ADMIN_BTN_W, ADMIN_H, ADMIN_HEADER_BTN_H,
    ADMIN_HEADER_BTN_W, ADMIN_HEADER_H, ADMIN_W,
    AUTH_LOGIN_BTN_HOVER,
    BTN_BG, BTN_HOVER,
    CLOSEBTN_HOVER, CLOSEBTN_PRESSED,
    HEADER_BG, ICON_CLOSE, ICON_MAX, ICON_MIN,
    LINE_BG, TEXT_WHITE, WINDOW_BG,
    TitleBarButton,
    get_available_screen_geometry,
)


class AdminModeWindow(QMainWindow):
    def __init__(self, terminal_window: Any = None) -> None:
        super().__init__()
        self.terminal_window = terminal_window
        self.register_window = None
        self.database_window = None

        self.setWindowTitle("Режим администратора")
        self.setFixedSize(ADMIN_W, ADMIN_H)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)

        self._drag_pos: QPoint | None = None

        root = QWidget(self)
        root.setObjectName("adminRoot")
        self.setCentralWidget(root)

        self.title_bar = QFrame(root)
        self.title_bar.setObjectName("adminTitleBar")
        self.title_bar.setGeometry(0, 0, ADMIN_W, ADMIN_HEADER_H)

        self.header_line = QFrame(root)
        self.header_line.setObjectName("adminHeaderLine")
        self.header_line.setGeometry(0, ADMIN_HEADER_H, ADMIN_W, 1)

        self.min_btn = TitleBarButton(ICON_MIN, parent=self.title_bar)
        self.min_btn.setGeometry(ADMIN_W - ADMIN_HEADER_BTN_W * 3, 0, ADMIN_HEADER_BTN_W, ADMIN_HEADER_BTN_H)

        self.max_btn = TitleBarButton(ICON_MAX, parent=self.title_bar)
        self.max_btn.setGeometry(ADMIN_W - ADMIN_HEADER_BTN_W * 2, 0, ADMIN_HEADER_BTN_W, ADMIN_HEADER_BTN_H)

        self.close_btn = TitleBarButton(
            ICON_CLOSE,
            hover_bg=CLOSEBTN_HOVER,
            pressed_bg=CLOSEBTN_PRESSED,
            parent=self.title_bar,
        )
        self.close_btn.setGeometry(ADMIN_W - ADMIN_HEADER_BTN_W, 0, ADMIN_HEADER_BTN_W, ADMIN_HEADER_BTN_H)

        self.title_label = QLabel("Режим администратора", root)
        self.title_label.setObjectName("adminModeTitle")
        self.title_label.setGeometry(0, 92, ADMIN_W, 45)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_x = 40

        self.register_btn = QPushButton("Регистрация пользователя", root)
        self.register_btn.setObjectName("adminModeButton")
        self.register_btn.setGeometry(btn_x, 168, ADMIN_BTN_W, ADMIN_BTN_H)

        self.database_btn = QPushButton("Управление базой данных", root)
        self.database_btn.setObjectName("adminModeButton")
        self.database_btn.setGeometry(btn_x, 236, ADMIN_BTN_W, ADMIN_BTN_H)

        self.terminal_btn = QPushButton("Терминальный режим", root)
        self.terminal_btn.setObjectName("adminModeBlueButton")
        self.terminal_btn.setGeometry(btn_x, 337, ADMIN_BTN_W, ADMIN_BTN_H)

        self.min_btn.clicked.connect(self.showMinimized)
        self.close_btn.clicked.connect(self.return_to_terminal)
        self.terminal_btn.clicked.connect(self.return_to_terminal)
        self.register_btn.clicked.connect(self.open_register_window)
        self.database_btn.clicked.connect(self.open_database_window)

        self.setStyleSheet(
            f"""
            #adminRoot {{
                background: {WINDOW_BG};
                border-radius: 0px;
            }}
            #adminTitleBar {{
                background: {HEADER_BG};
            }}
            #adminHeaderLine {{
                background: {LINE_BG};
            }}
            #adminModeTitle {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 24px;
                font-weight: 600;
            }}
            #adminModeButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 16px;
                font-weight: 500;
            }}
            #adminModeButton:hover {{ background: {BTN_HOVER}; }}
            #adminModeBlueButton {{
                background: {ADMIN_BLUE};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 16px;
                font-weight: 500;
            }}
            #adminModeBlueButton:hover {{ background: {AUTH_LOGIN_BTN_HOVER}; }}
            """
        )

    def open_register_window(self) -> None:
        from app.ui_register_user import RegisterUserWindow
        self.register_window = RegisterUserWindow(self)
        self.register_window.move(0, 0)
        self.register_window.show()
        self.hide()

    def open_database_window(self) -> None:
        from app.ui_user_manager import DatabaseManagementWindow
        self.database_window = DatabaseManagementWindow(self)
        self.database_window.move(0, 0)
        self.database_window.show()
        self.hide()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= ADMIN_HEADER_H:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def return_to_terminal(self) -> None:
        terminal = self.terminal_window
        if terminal is not None:
            terminal.show()
            camera = getattr(terminal, "camera", None)
            if camera is not None:
                camera.start_camera()
        self.close()