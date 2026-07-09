from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

try:
    from app.db import delete_user, get_all_users
except Exception as exc:
    delete_user = get_all_users = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from app.ui_theme import (
    AUTH_LOGIN_BTN,
    AUTH_LOGIN_BTN_HOVER,
    BORDER_COLOR,
    BTN_BG,
    BTN_DANGER,
    BTN_DANGER_HOVER,
    BTN_HOVER,
    CLOSEBTN_HOVER,
    CLOSEBTN_PRESSED,
    DB_BTN_H,
    DB_BTN_W,
    DB_BTN_Y,
    DB_H,
    DB_HEADER_BTN_H,
    DB_HEADER_BTN_W,
    DB_HEADER_H,
    DB_SCROLL_BG,
    DB_SCROLL_BTN,
    DB_SCROLL_W,
    DB_TABLE_BG,
    DB_TABLE_HEADER_BG,
    DB_TABLE_H,
    DB_TABLE_ROW_BG,
    DB_TABLE_TEXT,
    DB_TABLE_W,
    DB_TABLE_X,
    DB_TABLE_Y,
    DB_TITLE_H,
    DB_TITLE_W,
    DB_TITLE_X,
    DB_TITLE_Y,
    DB_W,
    HEADER_BG,
    ICON_CLOSE,
    ICON_MAX,
    ICON_MIN,
    ICON_SCROLL_DOWN,
    ICON_SCROLL_UP,
    LINE_BG,
    TEXT_WHITE,
    WINDOW_BG,
    ScalableWindowMixin,
    TitleBarButton,
    ask_delete_confirmation,
    show_error,
    show_warning,
)
from app.ui_edit_user import EditUserDialog


class DatabaseManagementWindow(QMainWindow, ScalableWindowMixin):
    def __init__(self, admin_window: Any = None) -> None:
        super().__init__()
        self.admin_window = admin_window

        self.setWindowTitle("Управление базой данных")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setup_scaled_window(DB_W, DB_H)

        self._drag_pos: QPoint | None = None
        self.all_users: list[Any] = []

        root = QWidget(self)
        root.setObjectName("dbRoot")
        self.setCentralWidget(root)

        self.title_bar = QFrame(root)
        self.title_bar.setObjectName("dbTitleBar")
        self.title_bar.setGeometry(0, 0, DB_W, DB_HEADER_H)

        self.header_line = QFrame(root)
        self.header_line.setObjectName("dbHeaderLine")
        self.header_line.setGeometry(0, DB_HEADER_H, DB_W, 1)

        self.min_btn = TitleBarButton(ICON_MIN, parent=self.title_bar)
        self.min_btn.setGeometry(
            DB_W - DB_HEADER_BTN_W * 3,
            0,
            DB_HEADER_BTN_W,
            DB_HEADER_BTN_H,
        )

        self.max_btn = TitleBarButton(ICON_MAX, parent=self.title_bar)
        self.max_btn.setGeometry(
            DB_W - DB_HEADER_BTN_W * 2,
            0,
            DB_HEADER_BTN_W,
            DB_HEADER_BTN_H,
        )

        self.close_btn = TitleBarButton(
            ICON_CLOSE,
            hover_bg=CLOSEBTN_HOVER,
            pressed_bg=CLOSEBTN_PRESSED,
            parent=self.title_bar,
        )
        self.close_btn.setGeometry(
            DB_W - DB_HEADER_BTN_W,
            0,
            DB_HEADER_BTN_W,
            DB_HEADER_BTN_H,
        )

        self.title_label = QLabel("Пользователи  системы", root)
        self.title_label.setObjectName("dbTitle")
        self.title_label.setGeometry(DB_TITLE_X, DB_TITLE_Y, DB_TITLE_W, DB_TITLE_H)

        self.search_input = QLineEdit(root)
        self.search_input.setObjectName("dbSearchInput")
        self.search_input.setPlaceholderText("Поиск")
        self.search_input.setGeometry(1230, DB_TITLE_Y, 606, DB_TITLE_H)

        self.table = QTableWidget(root)
        self.table.setObjectName("dbTable")
        self.table.setGeometry(DB_TABLE_X, DB_TABLE_Y, DB_TABLE_W, DB_TABLE_H)

        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "Никнейм", "ФИО", "Создан"])
        self.table.setRowCount(0)

        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(1, 450)
        self.table.setColumnWidth(2, 600)
        self.table.setColumnWidth(3, 505)

        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setFixedHeight(60)
        self.table.verticalHeader().setDefaultSectionSize(52)

        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)

        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.verticalScrollBar().setFixedWidth(DB_SCROLL_W)

        self.refresh_btn = QPushButton("Обновить", root)
        self.refresh_btn.setObjectName("dbBlueButton")
        self.refresh_btn.setGeometry(105, DB_BTN_Y, DB_BTN_W, DB_BTN_H)

        self.edit_btn = QPushButton("Изменить", root)
        self.edit_btn.setObjectName("dbButton")
        self.edit_btn.setGeometry(472, DB_BTN_Y, DB_BTN_W, DB_BTN_H)

        self.delete_btn = QPushButton("Удалить", root)
        self.delete_btn.setObjectName("dbDangerButton")
        self.delete_btn.setGeometry(838, DB_BTN_Y, DB_BTN_W, DB_BTN_H)

        self.close_window_btn = QPushButton("Закрыть", root)
        self.close_window_btn.setObjectName("dbButton")
        self.close_window_btn.setGeometry(1489, DB_BTN_Y, DB_BTN_W, DB_BTN_H)

        self.edit_dialog = EditUserDialog(root)

        self.min_btn.clicked.connect(self.showMinimized)
        self.max_btn.clicked.connect(self.toggle_scaled_maximize)
        self.close_btn.clicked.connect(self.return_to_admin)
        self.close_window_btn.clicked.connect(self.return_to_admin)
        self.refresh_btn.clicked.connect(self.load_users)
        self.edit_btn.clicked.connect(self.edit_selected_user)
        self.delete_btn.clicked.connect(self.delete_selected_user)
        self.search_input.textChanged.connect(self.filter_users)

        self.apply_scaled_geometry()
        self._apply_styles()
        self.load_users()

    def apply_scaled_geometry(self) -> None:
        self.title_bar.setGeometry(self.sr(0, 0, DB_W, DB_HEADER_H))
        self.header_line.setGeometry(self.sr(0, DB_HEADER_H, DB_W, 1))

        self.min_btn.setGeometry(
            self.sr(
                DB_W - DB_HEADER_BTN_W * 3,
                0,
                DB_HEADER_BTN_W,
                DB_HEADER_BTN_H,
            )
        )
        self.max_btn.setGeometry(
            self.sr(
                DB_W - DB_HEADER_BTN_W * 2,
                0,
                DB_HEADER_BTN_W,
                DB_HEADER_BTN_H,
            )
        )
        self.close_btn.setGeometry(
            self.sr(
                DB_W - DB_HEADER_BTN_W,
                0,
                DB_HEADER_BTN_W,
                DB_HEADER_BTN_H,
            )
        )

        self.title_label.setGeometry(
            self.sr(DB_TITLE_X, DB_TITLE_Y, DB_TITLE_W, DB_TITLE_H)
        )

        self.search_input.setGeometry(
            self.sr(1230, DB_TITLE_Y, 606, DB_TITLE_H)
        )

        self.table.setGeometry(
            self.sr(DB_TABLE_X, DB_TABLE_Y, DB_TABLE_W, DB_TABLE_H)
        )

        self.refresh_btn.setGeometry(self.sr(105, DB_BTN_Y, DB_BTN_W, DB_BTN_H))
        self.edit_btn.setGeometry(self.sr(472, DB_BTN_Y, DB_BTN_W, DB_BTN_H))
        self.delete_btn.setGeometry(self.sr(838, DB_BTN_Y, DB_BTN_W, DB_BTN_H))
        self.close_window_btn.setGeometry(
            self.sr(1489, DB_BTN_Y, DB_BTN_W, DB_BTN_H)
        )

        self.table.setColumnWidth(0, self.sx(150))
        self.table.setColumnWidth(1, self.sx(450))
        self.table.setColumnWidth(2, self.sx(600))
        self.table.setColumnWidth(3, self.sx(505))
        self.table.horizontalHeader().setFixedHeight(self.sy(60))
        self.table.verticalHeader().setDefaultSectionSize(self.sy(52))
        self.table.verticalScrollBar().setFixedWidth(max(self.sx(DB_SCROLL_W), 16))

    def resizeEvent(self, event) -> None:
        if not hasattr(self, "title_bar"):
            super().resizeEvent(event)
            return

        self.scale_x = max(self.width(), 1) / DB_W
        self.scale_y = max(self.height(), 1) / DB_H
        self.apply_scaled_geometry()

        if self.edit_dialog.isVisible():
            self.edit_dialog.show_centered()

        super().resizeEvent(event)

    def load_users(self) -> None:
        if get_all_users is None:
            self.all_users = []
            self.table.setRowCount(0)
            return

        self.all_users = list(get_all_users())
        self.filter_users()

    def filter_users(self) -> None:
        query = self.search_input.text().strip().lower()

        if not query:
            self._fill_table(self.all_users)
            return

        filtered_users: list[Any] = []

        for user in self.all_users:
            values = [
                str(user["id"]),
                str(user["username"]),
                str(user["full_name"]),
                str(user["created_at"]),
            ]

            if any(query in value.lower() for value in values):
                filtered_users.append(user)

        self._fill_table(filtered_users)

    def _fill_table(self, users: list[Any]) -> None:
        self.table.setRowCount(0)

        for row_idx, user in enumerate(users):
            self.table.insertRow(row_idx)

            values = [
                str(user["id"]),
                str(user["username"]),
                str(user["full_name"]),
                str(user["created_at"]),
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setForeground(QBrush(QColor(DB_TABLE_TEXT)))
                item.setBackground(QBrush(QColor(DB_TABLE_ROW_BG)))
                self.table.setItem(row_idx, col, item)

    def _selected_user_id(self) -> int | None:
        selected = self.table.selectedItems()

        if not selected:
            show_warning(self, "Внимание", "Выберите пользователя.")
            return None

        row = selected[0].row()
        item = self.table.item(row, 0)

        return int(item.text()) if item is not None else None

    def edit_selected_user(self) -> None:
        user_id = self._selected_user_id()

        if user_id is None:
            return

        row = self.table.currentRow()

        username_item = self.table.item(row, 1)
        full_name_item = self.table.item(row, 2)

        username = username_item.text() if username_item is not None else "—"
        full_name = full_name_item.text() if full_name_item is not None else ""

        self.edit_dialog.show_for_user(
            user_id=user_id,
            username=username,
            full_name=full_name,
            on_saved=self.load_users,
        )

    def delete_selected_user(self) -> None:
        user_id = self._selected_user_id()

        if user_id is None:
            return

        confirmed = ask_delete_confirmation(
            self,
            "Удаление пользователя",
            "Удалить выбранного пользователя? Будут удалены его шаблоны, фотографии и связанные записи.",
        )

        if not confirmed:
            return

        try:
            if delete_user is None:
                raise RuntimeError("Функция delete_user не загружена.")

            delete_user(user_id)
            self.load_users()

        except Exception as exc:
            show_error(self, "Ошибка", f"Не удалось удалить пользователя:\n{exc}")

    def _apply_styles(self) -> None:
        scroll_up_path = ICON_SCROLL_UP.as_posix()
        scroll_down_path = ICON_SCROLL_DOWN.as_posix()

        self.setStyleSheet(
            f"""
            #dbRoot {{
                background: {WINDOW_BG};
            }}

            #dbTitleBar {{
                background: {HEADER_BG};
            }}

            #dbHeaderLine {{
                background: {LINE_BG};
            }}

            #dbTitle {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 40px;
                font-weight: 700;
            }}

            #dbSearchInput {{
                background: {DB_TABLE_BG};
                color: {DB_TABLE_TEXT};
                border: 1px solid {BORDER_COLOR};
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 28px;
                font-weight: 500;
                padding-left: 18px;
                padding-right: 18px;
                selection-background-color: {AUTH_LOGIN_BTN};
            }}

            #dbSearchInput:focus {{
                border: 1px solid {AUTH_LOGIN_BTN};
            }}

            #dbSearchInput::placeholder {{
                color: #6B6F76;
            }}

            #dbTable {{
                background: {DB_TABLE_BG};
                color: {DB_TABLE_TEXT};
                border: 1px solid {BORDER_COLOR};
                gridline-color: transparent;
                font-family: "Fira Code";
                font-size: 28px;
                font-weight: 500;
                selection-background-color: {DB_TABLE_ROW_BG};
                selection-color: {DB_TABLE_TEXT};
            }}

            QHeaderView::section {{
                background: {DB_TABLE_HEADER_BG};
                color: {DB_TABLE_TEXT};
                border: none;
                font-family: "Fira Code";
                font-size: 30px;
                font-weight: 700;
            }}

            QTableWidget::item {{
                border: none;
                padding: 0px;
            }}

            QScrollBar:vertical {{
                background: {DB_SCROLL_BG};
                width: {DB_SCROLL_W}px;
                margin: {DB_SCROLL_BTN}px 0px {DB_SCROLL_BTN}px 0px;
            }}

            QScrollBar::handle:vertical {{
                background: {DB_SCROLL_BG};
                min-height: 80px;
                border-radius: 0px;
            }}

            QScrollBar::sub-line:vertical {{
                background: {DB_SCROLL_BG};
                height: {DB_SCROLL_BTN}px;
                subcontrol-position: top;
                subcontrol-origin: margin;
                image: url({scroll_up_path});
            }}

            QScrollBar::add-line:vertical {{
                background: {DB_SCROLL_BG};
                height: {DB_SCROLL_BTN}px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
                image: url({scroll_down_path});
            }}

            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {{
                background: {DB_SCROLL_BG};
            }}

            #dbButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 20px;
                font-weight: 500;
            }}

            #dbButton:hover {{
                background: {BTN_HOVER};
            }}

            #dbBlueButton {{
                background: {AUTH_LOGIN_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 20px;
                font-weight: 500;
            }}

            #dbBlueButton:hover {{
                background: {AUTH_LOGIN_BTN_HOVER};
            }}

            #dbDangerButton {{
                background: {BTN_DANGER};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 20px;
                font-weight: 500;
            }}

            #dbDangerButton:hover {{
                background: {BTN_DANGER_HOVER};
            }}
            """
        )

    def mousePressEvent(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.position().y() <= DB_HEADER_H
        ):
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

    def return_to_admin(self) -> None:
        if self.admin_window is not None:
            self.admin_window.show()

        self.close()