from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QLabel,
    QPushButton,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "app" else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------
# NORMAL TERMINAL
# ----------------------------
WINDOW_W = 918
WINDOW_H = 831

HEADER_H = 40
HEADER_LINE_H = 1

VIDEO_W = 883
VIDEO_H = 724
VIDEO_X = 18
VIDEO_Y = 91

MENU_W = 294
MENU_H = 180
MENU_X = VIDEO_X
MENU_Y = VIDEO_Y

BTN_W = 261
BTN_H = 39
MENU_BTN_X = 17
MENU_BTN_Y1 = 18
MENU_BTN_GAP = 12

MENU_ICON_X = VIDEO_X
MENU_ICON_Y = 51
MENU_ICON_W = 34
MENU_ICON_H = 34

# ----------------------------
# EXPANDED TERMINAL
# ----------------------------
EXPANDED_W = 1920
EXPANDED_H = 1080
EXPANDED_HEADER_H = 47

EXPANDED_VIDEO_W = 1147
EXPANDED_VIDEO_H = 941
EXPANDED_VIDEO_X = 34
EXPANDED_VIDEO_Y = 71

EXPANDED_MENU_X = 1298
EXPANDED_MENU_Y = 120
EXPANDED_MENU_W = 550
EXPANDED_MENU_H = 270
EXPANDED_MENU_BTN_W = 550
EXPANDED_MENU_BTN_H = 60
EXPANDED_MENU_BTN_FONT = 24

EXPANDED_MENU_ICON_X = 1298
EXPANDED_MENU_ICON_Y = 71
EXPANDED_MENU_ICON_W = 550
EXPANDED_MENU_ICON_H = 60

RIGHT_INFO_X = 1298
RIGHT_INFO_Y = 868
RIGHT_INFO_W = 531
RIGHT_INFO_H = 150
RIGHT_INFO_FONT = 24
RIGHT_INFO_LINE_THICKNESS = 2

# ----------------------------
# AUTH DIALOG
# ----------------------------
AUTH_W = 549
AUTH_H = 458
AUTH_FIELD_W = 486
AUTH_FIELD_H = 49

# ----------------------------
# ADMIN WINDOW
# ----------------------------
ADMIN_W = 692
ADMIN_HEADER_H = 40
ADMIN_BODY_H = 432
ADMIN_H = ADMIN_HEADER_H + ADMIN_BODY_H
ADMIN_HEADER_BTN_W = 45
ADMIN_HEADER_BTN_H = 34
ADMIN_BTN_W = 612
ADMIN_BTN_H = 49

# ----------------------------
# REGISTER WINDOW
# ----------------------------
REGISTER_W = 1920
REGISTER_H = 1080
REGISTER_HEADER_H = 47
REGISTER_HEADER_BTN_W = 63
REGISTER_HEADER_BTN_H = 47

REGISTER_VIDEO_X = 75
REGISTER_VIDEO_Y = 92
REGISTER_VIDEO_W = 1147
REGISTER_VIDEO_H = 941

REGISTER_PANEL_X = 1345
REGISTER_PANEL_Y = 115
REGISTER_PANEL_W = 488
REGISTER_PANEL_H = 878

REGISTER_INPUT_W = 486
REGISTER_INPUT_H = 51
REGISTER_BTN_W = 486
REGISTER_BTN_H = 49
REGISTER_SAVE_BTN_H = 74
REGISTER_FONT = 24
REGISTER_LINE_W = 491
REGISTER_LINE_H = 2

# ----------------------------
# SUCCESS DIALOG
# ----------------------------
SUCCESS_DIALOG_W = 475
SUCCESS_DIALOG_H = 249
SUCCESS_HEADER_W = 473
SUCCESS_HEADER_H = 39
SUCCESS_BOTTOM_W = 473
SUCCESS_BOTTOM_H = 52
SUCCESS_CLOSE_BTN_W = 37
SUCCESS_CLOSE_BTN_H = 39
SUCCESS_CONTINUE_W = 123
SUCCESS_CONTINUE_H = 28
SUCCESS_TEXT_FONT = 20
SUCCESS_BTN_FONT = 16

# ----------------------------
# DATABASE WINDOW
# ----------------------------
DB_W = 1920
DB_H = 1080
DB_HEADER_H = 47
DB_HEADER_BTN_W = 63
DB_HEADER_BTN_H = 47

DB_TITLE_X = 80
DB_TITLE_Y = 98
DB_TITLE_W = 900
DB_TITLE_H = 54

DB_TABLE_X = 80
DB_TABLE_Y = 176
DB_TABLE_W = 1756
DB_TABLE_H = 800

DB_SCROLL_W = 26
DB_SCROLL_BTN = 26

DB_BTN_W = 304
DB_BTN_H = 49
DB_BTN_Y = 1004

# ----------------------------
# COLORS
# ----------------------------
WINDOW_BG = "#121314"
HEADER_BG = "#121314"
LINE_BG = "#2E2F31"
VIDEO_BG = "#0C0E12"
MENU_BG = "#121314"

BTN_BG = "#2E2F31"
BTN_DANGER = "#C93131"
BTN_HOVER = "#3A3E45"
BTN_DANGER_HOVER = "#AA2C2C"

BORDER_COLOR = "#676A70"
BORDER_WIDTH = 1
CORNER_RADIUS = 10

TITLEBTN_HOVER = "#1E2228"
TITLEBTN_PRESSED = "#171B20"
CLOSEBTN_HOVER = "#C93131"
CLOSEBTN_PRESSED = "#AA2C2C"

TEXT_WHITE = "#FFFFFF"
TEXT_GREEN = "#54BF48"

STATUS_OK = "#54BF48"
STATUS_FAIL = "#C93131"
STATUS_NONE = "#6B6F76"

MENU_ICON_BG_EXPANDED = "#1E1E1E"

AUTH_BG = "#0F1011"
AUTH_BORDER = "#2E2F31"
AUTH_INPUT_BG = "#676A70"
AUTH_LOGIN_BTN = "#3166C9"
AUTH_LOGIN_BTN_HOVER = "#3D73D8"
AUTH_CANCEL_BTN = "#2E2F31"
AUTH_RADIUS = 5

ADMIN_BLUE = AUTH_LOGIN_BTN

REGISTER_INPUT_BG = "#919398"
REGISTER_CAMERA_BTN_BG = "#1E1E1E"

DB_SCROLL_BG = "#585B61"
DB_TABLE_BG = "#1E1E1E"
DB_TABLE_HEADER_BG = "#2E2F31"
DB_TABLE_ROW_BG = "#272727"
DB_TABLE_TEXT = "#FFFFFF"

# ----------------------------
# ICONS
# ----------------------------
TITLE_SLOT_W = 52
TITLE_SLOT_H = 39
TITLE_ICON_SIZE = QSize(18, 18)
TITLE_ICON_OFFSET_Y = 4

MENU_ICON_SIZE_NORMAL = QSize(24, 24)
MENU_ICON_SIZE_EXPANDED = QSize(30, 30)

ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_MENU = ASSETS_DIR / "menu_mini.png"
ICON_MENU_MAXI = ASSETS_DIR / "menu_maxi.png"
ICON_MIN = ASSETS_DIR / "minimize.png"
ICON_MAX = ASSETS_DIR / "maximize.png"
ICON_RESTORE = ASSETS_DIR / "restore.png"
ICON_CLOSE = ASSETS_DIR / "close.png"
ICON_SUCCESS = ASSETS_DIR / "info.png"
ICON_SCROLL_UP = ASSETS_DIR / "up.png"
ICON_SCROLL_DOWN = ASSETS_DIR / "down.png"

DEFAULT_GREETING_TEXT = "Система готова к работе"
WELCOME_GREETING_TEXT = "Добро пожаловать!"
GREETING_FONT_SIZE = 17

MAX_SAMPLES = 10
MIN_SAMPLES = 3


def get_available_screen_geometry() -> QRect:
    screen = QApplication.primaryScreen()
    if screen is None:
        return QRect(0, 0, 1920, 1040)
    return screen.availableGeometry()


def get_haarcascade_path() -> str:
    cv2_dir = Path(cv2.__file__).resolve().parent
    cascade = cv2_dir / "data" / "haarcascade_frontalface_default.xml"
    return str(cascade)


def map_frame_rect_to_widget(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    target_w: int,
    target_h: int,
) -> QRect:
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]
    scale = max(target_w / frame_w, target_h / frame_h)
    scaled_w = frame_w * scale
    scaled_h = frame_h * scale
    offset_x = (scaled_w - target_w) / 2
    offset_y = (scaled_h - target_h) / 2
    return QRect(
        int(x * scale - offset_x),
        int(y * scale - offset_y),
        int(w * scale),
        int(h * scale),
    )


def menu_button_style(font_px: int, danger: bool = False) -> str:
    bg = BTN_DANGER if danger else BTN_BG
    hover = BTN_DANGER_HOVER if danger else BTN_HOVER
    return f"""
        QPushButton {{
            background: {bg};
            color: {TEXT_WHITE};
            border: none;
            border-radius: 5px;
            font-family: "Fira Code";
            font-size: {font_px}px;
            font-weight: 500;
            letter-spacing: -0.5px;
        }}
        QPushButton:hover {{
            background: {hover};
        }}
    """


def _fallback_icon(name: str) -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor(0, 0, 0, 0))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    white = QColor(TEXT_WHITE)
    green = QColor(STATUS_OK)

    pen = QPen(white, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if name in {"menu_mini", "menu_maxi"}:
        painter.drawLine(15, 18, 49, 18)
        painter.drawLine(15, 32, 49, 32)
        painter.drawLine(15, 46, 49, 46)
    elif name == "minimize":
        painter.drawLine(18, 39, 46, 39)
    elif name == "maximize":
        painter.drawRect(QRect(19, 19, 26, 26))
    elif name == "restore":
        painter.drawRect(QRect(16, 25, 24, 24))
        painter.drawLine(25, 16, 48, 16)
        painter.drawLine(48, 16, 48, 39)
        painter.drawLine(40, 39, 48, 39)
    elif name == "close":
        painter.drawLine(20, 20, 44, 44)
        painter.drawLine(44, 20, 20, 44)
    elif name == "info":
        painter.setPen(QPen(green, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawEllipse(QRect(11, 11, 42, 42))
        painter.drawLine(32, 30, 32, 45)
        painter.drawPoint(32, 21)
    elif name == "up":
        path = QPainterPath()
        path.moveTo(32, 18)
        path.lineTo(14, 42)
        path.lineTo(50, 42)
        path.closeSubpath()
        painter.setBrush(white)
        painter.drawPath(path)
    elif name == "down":
        path = QPainterPath()
        path.moveTo(14, 22)
        path.lineTo(50, 22)
        path.lineTo(32, 46)
        path.closeSubpath()
        painter.setBrush(white)
        painter.drawPath(path)

    painter.end()
    return QIcon(pixmap)


def load_icon(path: Path) -> QIcon:
    if path.exists():
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    return _fallback_icon(path.stem)


def show_error(parent: QWidget | None, title: str, message: str) -> None:
    dialog = StyledMessageDialog(
        parent=parent,
        title=title,
        message=message,
        dialog_type="error",
        confirm_text="Понятно",
        cancel_text=None,
    )
    dialog.exec()


def show_warning(parent: QWidget | None, title: str, message: str) -> None:
    dialog = StyledMessageDialog(
        parent=parent,
        title=title,
        message=message,
        dialog_type="warning",
        confirm_text="Понятно",
        cancel_text=None,
    )
    dialog.exec()


def ask_delete_confirmation(parent: QWidget | None, title: str, message: str) -> bool:
    dialog = StyledMessageDialog(
        parent=parent,
        title=title,
        message=message,
        dialog_type="danger",
        confirm_text="Удалить",
        cancel_text="Отмена",
    )
    return dialog.exec() == QDialog.DialogCode.Accepted


class ScalableWindowMixin:
    base_width: int = 1920
    base_height: int = 1080
    scale_x: float = 1.0
    scale_y: float = 1.0

    def setup_scaled_window(self, base_width: int, base_height: int) -> None:
        from typing import cast
        from PySide6.QtWidgets import QMainWindow
        self.base_width = base_width
        self.base_height = base_height

        window = cast(QMainWindow, self)
        available = get_available_screen_geometry()

        self.scale_x = available.width() / base_width
        self.scale_y = available.height() / base_height

        window.setGeometry(
            available.x(),
            available.y(),
            available.width(),
            available.height(),
        )
        window.setMinimumSize(1100, 700)

    def sx(self, value: int | float) -> int:
        return int(value * self.scale_x)

    def sy(self, value: int | float) -> int:
        return int(value * self.scale_y)

    def sr(self, x: int | float, y: int | float, w: int | float, h: int | float) -> QRect:
        return QRect(self.sx(x), self.sy(y), self.sx(w), self.sy(h))

    def toggle_scaled_maximize(self) -> None:
        from typing import cast
        from PySide6.QtWidgets import QMainWindow
        window = cast(QMainWindow, self)

        if window.isMaximized():
            window.showNormal()
            available = get_available_screen_geometry()
            window.setGeometry(
                available.x(),
                available.y(),
                available.width(),
                available.height(),
            )
        else:
            window.showMaximized()


class TitleBarButton(QPushButton):
    def __init__(
        self,
        icon_path: Path,
        normal_bg: str = "transparent",
        hover_bg: str = TITLEBTN_HOVER,
        pressed_bg: str = TITLEBTN_PRESSED,
        icon_size: QSize = TITLE_ICON_SIZE,
        icon_offset_y: int = TITLE_ICON_OFFSET_Y,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._icon = load_icon(icon_path)
        self._normal_bg = QColor(normal_bg) if normal_bg != "transparent" else None
        self._hover_bg = QColor(hover_bg)
        self._pressed_bg = QColor(pressed_bg)
        self._icon_size = icon_size
        self._icon_offset_y = icon_offset_y

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFlat(True)
        self.setStyleSheet("border:none; background:transparent; padding:0; margin:0;")

    def set_icon_path(self, icon_path: Path) -> None:
        self._icon = load_icon(icon_path)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self.isDown():
            painter.fillRect(self.rect(), self._pressed_bg)
        elif self.underMouse():
            painter.fillRect(self.rect(), self._hover_bg)
        elif self._normal_bg is not None:
            painter.fillRect(self.rect(), self._normal_bg)

        if not self._icon.isNull():
            pixmap = self._icon.pixmap(self._icon_size)
            x = (self.width() - self._icon_size.width()) // 2
            y = (self.height() - self._icon_size.height()) // 2 + self._icon_offset_y
            painter.drawPixmap(x, y, pixmap)


class StyledMessageDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        message: str,
        dialog_type: str = "warning",
        confirm_text: str = "Понятно",
        cancel_text: str | None = None,
    ) -> None:
        super().__init__(parent)

        self.dialog_type = dialog_type

        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(SUCCESS_DIALOG_W, SUCCESS_DIALOG_H)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setObjectName("styledMessageDialog")

        self.header = QFrame(self)
        self.header.setObjectName("styledMessageHeader")
        self.header.setGeometry(1, 0, SUCCESS_HEADER_W, SUCCESS_HEADER_H)

        self.close_btn = TitleBarButton(
            ICON_CLOSE,
            hover_bg=CLOSEBTN_HOVER,
            pressed_bg=CLOSEBTN_PRESSED,
            parent=self.header,
        )
        self.close_btn.setGeometry(
            SUCCESS_HEADER_W - SUCCESS_CLOSE_BTN_W,
            0,
            SUCCESS_CLOSE_BTN_W,
            SUCCESS_CLOSE_BTN_H,
        )

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("styledMessageTitle")
        self.title_label.setGeometry(28, 55, SUCCESS_DIALOG_W - 56, 32)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.message_label = QLabel(message, self)
        self.message_label.setObjectName("styledMessageText")
        self.message_label.setGeometry(28, 92, SUCCESS_DIALOG_W - 56, 82)
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        self.bottom_panel = QFrame(self)
        self.bottom_panel.setObjectName("styledMessageBottom")
        self.bottom_panel.setGeometry(
            1,
            SUCCESS_DIALOG_H - SUCCESS_BOTTOM_H,
            SUCCESS_BOTTOM_W,
            SUCCESS_BOTTOM_H,
        )

        if cancel_text is None:
            self.confirm_btn = QPushButton(confirm_text, self.bottom_panel)
            self.confirm_btn.setObjectName(self._confirm_object_name())
            self.confirm_btn.setGeometry(SUCCESS_BOTTOM_W - 140, 12, 122, SUCCESS_CONTINUE_H)
            self.cancel_btn = None
        else:
            self.cancel_btn = QPushButton(cancel_text, self.bottom_panel)
            self.cancel_btn.setObjectName("styledMessageCancelButton")
            self.cancel_btn.setGeometry(SUCCESS_BOTTOM_W - 252, 12, 105, SUCCESS_CONTINUE_H)

            self.confirm_btn = QPushButton(confirm_text, self.bottom_panel)
            self.confirm_btn.setObjectName(self._confirm_object_name())
            self.confirm_btn.setGeometry(SUCCESS_BOTTOM_W - 135, 12, 117, SUCCESS_CONTINUE_H)

            self.cancel_btn.clicked.connect(self.reject)

        self.close_btn.clicked.connect(self.reject)
        self.confirm_btn.clicked.connect(self.accept)

        self._apply_styles()
        self._center_to_parent()

    def _confirm_object_name(self) -> str:
        if self.dialog_type == "danger":
            return "styledMessageDangerButton"
        if self.dialog_type == "error":
            return "styledMessageErrorButton"
        return "styledMessageConfirmButton"

    def _center_to_parent(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            parent_window = parent.window()
            x = parent_window.x() + (parent_window.width() - SUCCESS_DIALOG_W) // 2
            y = parent_window.y() + (parent_window.height() - SUCCESS_DIALOG_H) // 2
            self.move(x, y)

    def _apply_styles(self) -> None:
        if self.dialog_type in ("danger", "error"):
            title_color = BTN_DANGER
        else:
            title_color = TEXT_WHITE

        self.setStyleSheet(
            f"""
            #styledMessageDialog {{
                background: {WINDOW_BG};
                border: 1px solid {LINE_BG};
                border-radius: 0px;
            }}
            #styledMessageHeader {{
                background: {WINDOW_BG};
                border-bottom: 1px solid {LINE_BG};
            }}
            #styledMessageBottom {{
                background: {WINDOW_BG};
                border-top: 1px solid {LINE_BG};
            }}
            #styledMessageTitle {{
                color: {title_color};
                font-family: "Fira Code";
                font-size: 20px;
                font-weight: 600;
            }}
            #styledMessageText {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 15px;
                font-weight: 400;
                line-height: 130%;
            }}
            #styledMessageConfirmButton {{
                background: {AUTH_LOGIN_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #styledMessageConfirmButton:hover {{ background: {AUTH_LOGIN_BTN_HOVER}; }}
            #styledMessageErrorButton {{
                background: {BTN_DANGER};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #styledMessageErrorButton:hover {{ background: {BTN_DANGER_HOVER}; }}
            #styledMessageDangerButton {{
                background: {BTN_DANGER};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #styledMessageDangerButton:hover {{ background: {BTN_DANGER_HOVER}; }}
            #styledMessageCancelButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: 14px;
                font-weight: 500;
            }}
            #styledMessageCancelButton:hover {{ background: {BTN_HOVER}; }}
            """
        )


class MenuIconButton(QPushButton):
    def __init__(self, icon_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._icon = load_icon(icon_path)
        self._icon_size = MENU_ICON_SIZE_NORMAL
        self._bg_color: str | None = None
        self._radius = 0
        self._left_padding = 0

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFlat(True)
        self.setStyleSheet("border:none; background:transparent; padding:0; margin:0;")

    def set_icon_path(self, icon_path: Path) -> None:
        self._icon = load_icon(icon_path)
        self.update()

    def set_icon_size(self, size: QSize) -> None:
        self._icon_size = size
        self.update()

    def set_background(self, color: str | None, radius: int = 0) -> None:
        self._bg_color = color
        self._radius = radius
        self.update()

    def set_left_padding(self, value: int) -> None:
        self._left_padding = value
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._bg_color:
            path = QPainterPath()
            path.addRoundedRect(self.rect(), self._radius, self._radius)
            painter.fillPath(path, QColor(self._bg_color))

        if not self._icon.isNull():
            pixmap = self._icon.pixmap(self._icon_size)
            x = self._left_padding
            y = (self.height() - self._icon_size.height()) // 2
            painter.drawPixmap(x, y, pixmap)


class TitleBar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setGeometry(0, 0, WINDOW_W, HEADER_H)
        self._drag_pos: QPoint | None = None

        self.min_btn = TitleBarButton(ICON_MIN, parent=self)
        self.max_btn = TitleBarButton(ICON_MAX, parent=self)
        self.close_btn = TitleBarButton(
            ICON_CLOSE,
            hover_bg=CLOSEBTN_HOVER,
            pressed_bg=CLOSEBTN_PRESSED,
            parent=self,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        window = self.window()
        if self._drag_pos is not None and window is not None:
            delta = event.globalPosition().toPoint() - self._drag_pos
            window.move(window.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)


class RoundedVideoLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._radius = CORNER_RADIUS - 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setVideoPixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect(), self._radius, self._radius)
        painter.setClipPath(path)
        painter.fillRect(self.rect(), QColor(VIDEO_BG))

        if self._pixmap is not None and not self._pixmap.isNull():
            painter.drawPixmap(self.rect(), self._pixmap)
        else:
            painter.setPen(QColor(TEXT_WHITE))
            painter.setFont(QFont("Fira Code", 15))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Камера не найдена")


class FaceOverlay(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.face_rect: QRect | None = None
        self.person_name = ""
        self.greeting = DEFAULT_GREETING_TEXT
        self.status = "Статус: Лицо не найдено"
        self.status_state = "none"
        self.show_status_text = True

    def set_face_rect(self, rect: QRect | None) -> None:
        self.face_rect = rect
        self.update()

    def set_status(self, text: str, state: str) -> None:
        self.status = text
        self.status_state = state
        self.update()

    def _indicator_color(self) -> QColor:
        if self.status_state == "ok":
            return QColor(STATUS_OK)
        if self.status_state == "fail":
            return QColor(STATUS_FAIL)
        return QColor(STATUS_NONE)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        greet_color = QColor(TEXT_GREEN)
        greet_color.setAlpha(191)

        status_color = QColor(TEXT_WHITE)
        status_color.setAlpha(191)

        painter.setPen(greet_color)
        greet_font = QFont("Fira Code", GREETING_FONT_SIZE)
        greet_font.setWeight(QFont.Weight.Medium)
        painter.setFont(greet_font)
        painter.drawText(
            QRect(0, 10, self.width(), 32),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            self.greeting,
        )

        if self.face_rect is not None:
            if self.status_state == "ok":
                face_box_color = QColor(STATUS_OK)
            else:
                face_box_color = QColor(STATUS_NONE)

            painter.setPen(QPen(face_box_color, 7))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.face_rect)

            painter.setPen(QColor(TEXT_WHITE))
            name_font = QFont("Fira Code", 17)
            name_font.setWeight(QFont.Weight.Medium)
            painter.setFont(name_font)
            painter.drawText(
                QRect(self.face_rect.x(), self.face_rect.bottom() + 8, self.face_rect.width(), 30),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                self.person_name,
            )

        if self.show_status_text:
            painter.setPen(status_color)
            status_font = QFont("Fira Code", 17)
            status_font.setWeight(QFont.Weight.Medium)
            painter.setFont(status_font)

            text_rect = QRect(14, self.height() - 38, self.width() - 28, 28)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self.status,
            )

            indicator_radius = 5
            indicator_x = text_rect.x() + 96
            indicator_y = text_rect.y() + text_rect.height() // 2

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._indicator_color())
            painter.drawEllipse(
                indicator_x,
                indicator_y - indicator_radius,
                indicator_radius * 2,
                indicator_radius * 2,
            )