from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QMainWindow,
    QPushButton,
    QWidget,
)

try:
    from app.db import init_db
    from app.face_detector import FaceDetector
    from app.face_embedder import FaceEmbedder
    from app.face_service import crop_largest_face, save_face_sample
    from app.matcher import match_face
    from app.anti_spoof_service import AntiSpoofService
    from app.arduino_controller import ArduinoController
except Exception as exc:
    FaceDetector = FaceEmbedder = AntiSpoofService = ArduinoController = None
    init_db = None
    crop_largest_face = save_face_sample = match_face = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from app.ui_theme import (
    BORDER_COLOR, BORDER_WIDTH, BTN_BG, BTN_DANGER, BTN_HOVER, BTN_DANGER_HOVER,
    CLOSEBTN_HOVER, CLOSEBTN_PRESSED, CORNER_RADIUS,
    DEFAULT_GREETING_TEXT, EXPANDED_HEADER_H, EXPANDED_MENU_BTN_FONT,
    EXPANDED_MENU_BTN_H, EXPANDED_MENU_BTN_W, EXPANDED_MENU_H,
    EXPANDED_MENU_ICON_H, EXPANDED_MENU_ICON_W, EXPANDED_MENU_ICON_X,
    EXPANDED_MENU_ICON_Y, EXPANDED_MENU_W, EXPANDED_MENU_X, EXPANDED_MENU_Y,
    EXPANDED_VIDEO_H, EXPANDED_VIDEO_W, EXPANDED_VIDEO_X, EXPANDED_VIDEO_Y,
    EXPANDED_W, EXPANDED_H, HEADER_BG, HEADER_H, HEADER_LINE_H,
    ICON_CLOSE, ICON_MAX, ICON_MENU, ICON_MENU_MAXI, ICON_MIN, ICON_RESTORE,
    LINE_BG, MENU_BG, MENU_ICON_BG_EXPANDED, MENU_ICON_H,
    MENU_ICON_W, MENU_ICON_X, MENU_ICON_Y, MENU_W, MENU_H, MENU_X, MENU_Y,
    MENU_BTN_X, MENU_BTN_GAP, MENU_BTN_Y1, BTN_W, BTN_H,
    RIGHT_INFO_FONT, RIGHT_INFO_H, RIGHT_INFO_LINE_THICKNESS,
    RIGHT_INFO_W, RIGHT_INFO_X, RIGHT_INFO_Y,
    STATUS_FAIL, STATUS_NONE, STATUS_OK,
    TEXT_WHITE, TITLE_SLOT_H, TITLE_SLOT_W,
    VIDEO_BG, VIDEO_H, VIDEO_W, VIDEO_X, VIDEO_Y,
    WELCOME_GREETING_TEXT, WINDOW_BG, WINDOW_W, WINDOW_H,
    AUTH_BG, AUTH_BORDER, AUTH_INPUT_BG, AUTH_LOGIN_BTN, AUTH_LOGIN_BTN_HOVER,
    AUTH_CANCEL_BTN, AUTH_RADIUS,
    FaceOverlay, MenuIconButton, RoundedVideoLabel, TitleBar, TitleBarButton,
    get_available_screen_geometry, map_frame_rect_to_widget, menu_button_style,
    PROJECT_ROOT, ADMIN_W, ADMIN_H,
)


class CameraViewport(QFrame):
    def __init__(self, x: int, y: int, parent: QWidget | None = None, arduino: Any = None, logger: Callable[[str], None] | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraViewport")
        self.setGeometry(x, y, VIDEO_W, VIDEO_H)
        self.arduino = arduino
        self.logger = logger

        inner = BORDER_WIDTH
        self.video_label = RoundedVideoLabel(self)
        self.video_label.setGeometry(inner, inner, VIDEO_W - inner * 2, VIDEO_H - inner * 2)

        self.overlay = FaceOverlay(self)
        self.overlay.setGeometry(inner, inner, VIDEO_W - inner * 2, VIDEO_H - inner * 2)

        self.capture: cv2.VideoCapture | None = None
        self.on_status_changed: Callable[[str, str], None] | None = None
        self.current_frame: Any = None

        self.detector = None
        self.embedder = None
        self.anti_spoof = None
        self.pipeline_error: str | None = None
        self.last_recognition_time = 0.0
        self.recognition_interval = 1.0
        self.last_access_ts = 0.0
        self.access_cooldown = 2.5
        self.busy_recognition = False

        self.smooth_face_box: tuple[int, int, int, int] | None = None
        self.last_raw_face_box: tuple[int, int, int, int] | None = None
        self.face_lost_frames = 0
        self.face_lost_limit = 5
        self.face_smooth_alpha = 0.84

        self.pending_user = None
        self.success_streak = 0
        self.required_successes = 3

        try:
            if IMPORT_ERROR is not None:
                raise RuntimeError(f"Ошибка импорта модулей проекта: {IMPORT_ERROR}")
            assert FaceDetector is not None
            assert FaceEmbedder is not None
            assert AntiSpoofService is not None
            self.detector = FaceDetector()
            self.embedder = FaceEmbedder()
            self.anti_spoof = AntiSpoofService(enabled=True, window_size=10, min_confidence=0.80)
        except Exception as exc:
            self.pipeline_error = str(exc)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.start_camera()

    def log(self, text: str) -> None:
        if callable(self.logger):
            self.logger(text)

    def _emit_status(self, text: str, state: str) -> None:
        self.overlay.set_status(text, state)
        if callable(self.on_status_changed):
            self.on_status_changed(text, state)

    def _safe_arduino(self, command: str) -> None:
        now = time.time()
        if now - self.last_access_ts < self.access_cooldown:
            return
        if self.arduino is not None:
            fn = getattr(self.arduino, command, None)
            if callable(fn):
                fn()
        self.last_access_ts = now

    def start_camera(self) -> None:
        self.release_camera_only()

        self.smooth_face_box = None
        self.last_raw_face_box = None
        self.face_lost_frames = 0
        self.overlay.set_face_rect(None)
        self.overlay.greeting = DEFAULT_GREETING_TEXT
        self.overlay.person_name = "—"

        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        try:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        except Exception:
            pass
        self.timer.start(30)
        if self.pipeline_error:
            self._emit_status("Статус: Ошибка пайплайна", "fail")
            self.log(f"Терминал: ошибка инициализации пайплайна: {self.pipeline_error}")
        else:
            self._emit_status("Статус: Ожидание лица", "none")
            self.log("Терминал: камера запущена")

    def release_camera_only(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        self.capture = None

    def release(self) -> None:
        self.timer.stop()
        self.release_camera_only()
        if self.anti_spoof is not None:
            self.anti_spoof.reset()

    def update_frame(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self.overlay.person_name = "—"
            self._emit_status("Статус: Камера не найдена", "none")
            self.overlay.set_face_rect(None)
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self.overlay.person_name = "—"
            self._emit_status("Статус: Нет сигнала", "none")
            self.overlay.set_face_rect(None)
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        self._show_frame(frame)
        self._draw_detected_face(frame)

        now = time.time()
        if self.pipeline_error is None and not self.busy_recognition and now - self.last_recognition_time >= self.recognition_interval:
            self.busy_recognition = True
            try:
                self.process_recognition(frame)
            finally:
                self.last_recognition_time = now
                self.busy_recognition = False

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setVideoPixmap(scaled)

    def _smooth_bbox(self, new_box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if new_box is None:
            self.face_lost_frames += 1
            if self.face_lost_frames >= self.face_lost_limit:
                self.smooth_face_box = None
                self.last_raw_face_box = None
            return self.smooth_face_box

        self.face_lost_frames = 0
        self.last_raw_face_box = new_box

        if self.smooth_face_box is None:
            self.smooth_face_box = new_box
            return new_box

        old_x, old_y, old_w, old_h = self.smooth_face_box
        new_x, new_y, new_w, new_h = new_box
        alpha = self.face_smooth_alpha

        self.smooth_face_box = (
            int(old_x * alpha + new_x * (1.0 - alpha)),
            int(old_y * alpha + new_y * (1.0 - alpha)),
            int(old_w * alpha + new_w * (1.0 - alpha)),
            int(old_h * alpha + new_h * (1.0 - alpha)),
        )
        return self.smooth_face_box

    def _set_smoothed_face_rect(self, raw_box: tuple[int, int, int, int] | None, frame) -> None:
        smoothed_box = self._smooth_bbox(raw_box)
        if smoothed_box is None:
            self.overlay.set_face_rect(None)
            return
        rect = map_frame_rect_to_widget(smoothed_box, frame.shape, self.overlay.width(), self.overlay.height())
        self.overlay.set_face_rect(rect)

    def _draw_detected_face(self, frame) -> tuple[int, int, int, int] | None:
        if self.detector is None:
            self.last_raw_face_box = None
            self._set_smoothed_face_rect(None, frame)
            return None

        faces = self.detector.detect(frame)
        if not faces:
            self.last_raw_face_box = None
            self._set_smoothed_face_rect(None, frame)
            return None

        bbox = max(faces, key=lambda f: f[2] * f[3])
        raw_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        self._set_smoothed_face_rect(raw_box, frame)
        return raw_box

    def process_recognition(self, frame) -> None:
        if self.detector is None or self.embedder is None or match_face is None:
            self.overlay.person_name = "—"
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self._emit_status("Статус: Пайплайн не загружен", "fail")
            return

        bbox = self.last_raw_face_box
        if bbox is None:
            self.pending_user = None
            self.success_streak = 0
            self.overlay.person_name = "—"
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self._emit_status("Статус: Лицо не найдено", "none")
            if self.arduino is not None:
                idle = getattr(self.arduino, "idle", None)
                if callable(idle):
                    idle()
            return

        if self.anti_spoof is not None:
            spoof_result = self.anti_spoof.check_frame(frame)
            self.log(
                "ANTI-SPOOF: "
                f"ok={spoof_result.get('ok')}, reason={spoof_result.get('reason')}, "
                f"label={spoof_result.get('label')}, score={spoof_result.get('score')}, "
                f"raw_label={spoof_result.get('raw_label')}, raw_score={spoof_result.get('raw_score')}"
            )
            if not spoof_result.get("ok"):
                reason = spoof_result.get("reason")
                if reason in {"face_not_found_for_antispoof", "invalid_face_box", "empty_face_crop"}:
                    self.pending_user = None
                    self.success_streak = 0
                    self.overlay.person_name = "—"
                    self.overlay.greeting = DEFAULT_GREETING_TEXT
                    self._emit_status("Статус: Проверка живости", "none")
                    return
                self.pending_user = None
                self.success_streak = 0
                self.overlay.person_name = "—"
                self.overlay.greeting = DEFAULT_GREETING_TEXT
                self._emit_status("Статус: Доступ запрещён", "fail")
                self.log(f"Терминал: отказ — anti-spoof: {spoof_result.get('message')}")
                if self.anti_spoof is not None:
                    self.anti_spoof.reset()
                self._safe_arduino("access_denied")
                return

        embedding, _face_box = self.embedder.get_embedding(frame)
        if embedding is None:
            self.pending_user = None
            self.success_streak = 0
            self.overlay.person_name = "—"
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self._emit_status("Статус: Доступ запрещён", "fail")
            self.log("Терминал: отказ — не удалось построить эмбеддинг")
            self._safe_arduino("access_denied")
            return

        result = match_face(embedding, threshold=0.75)
        if result is None:
            self.pending_user = None
            self.success_streak = 0
            self.overlay.person_name = "—"
            self.overlay.greeting = DEFAULT_GREETING_TEXT
            self._emit_status("Статус: Доступ запрещён", "fail")
            self.log("Терминал: отказ — пользователь не распознан")
            if self.anti_spoof is not None:
                self.anti_spoof.reset()
            self._safe_arduino("access_denied")
            return

        full_name = str(result["full_name"])
        username = str(result["username"])
        score = float(result["score"])

        # Тихое внутреннее подтверждение:
        # интерфейс НЕ показывает имя и "Доступ разрешён" после одного случайного совпадения.
        if self.pending_user == username:
            self.success_streak += 1
        else:
            self.pending_user = username
            self.success_streak = 1

        if self.success_streak < self.required_successes:
            self.log(
                f"Терминал: предварительное совпадение — "
                f"{full_name} ({username}), "
                f"score={score:.4f}, "
                f"confirm={self.success_streak}/{self.required_successes}"
            )
            return

        self.overlay.person_name = full_name
        self.overlay.greeting = WELCOME_GREETING_TEXT
        self._emit_status("Статус: Доступ разрешён", "ok")
        self.log(f"Терминал: доступ разрешён — {full_name} ({username}), score={score:.4f}")

        if self.anti_spoof is not None:
            self.anti_spoof.reset()

        self._safe_arduino("access_granted")

        self.pending_user = None
        self.success_streak = 0


class SideMenuCard(QFrame):
    def __init__(self, x: int, y: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setGeometry(x, y, MENU_W, MENU_H)

        y1 = MENU_BTN_Y1
        y2 = y1 + BTN_H + MENU_BTN_GAP
        y3 = y2 + BTN_H + MENU_BTN_GAP

        self.admin_btn = self._make_btn("Войти как администратор", MENU_BTN_X, y1, False)
        self.restart_btn = self._make_btn("Перезапуск камеры", MENU_BTN_X, y2, False)
        self.exit_btn = self._make_btn("Выход из программы", MENU_BTN_X, y3, True)

    def _make_btn(self, text: str, x: int, y: int, danger: bool) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setGeometry(x, y, BTN_W, BTN_H)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        btn.setStyleSheet(menu_button_style(16, danger))
        return btn


class RightInfoPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setGeometry(RIGHT_INFO_X, RIGHT_INFO_Y, RIGHT_INFO_W, RIGHT_INFO_H)
        self.ready_text = "Система готова к работе"
        self.status_text = "Статус: Доступ разрешён"
        self.status_state = "ok"

    def set_status(self, text: str, state: str) -> None:
        self.status_text = text
        self.status_state = state
        self.update()

    def _indicator_color(self) -> QColor:
        if self.status_state == "ok":
            return QColor(STATUS_OK)
        if self.status_state == "fail":
            return QColor(STATUS_FAIL)
        return QColor(STATUS_NONE)

    def paintEvent(self, event) -> None:
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QFont
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(QColor(LINE_BG), RIGHT_INFO_LINE_THICKNESS))
        painter.drawLine(0, 0, RIGHT_INFO_W, 0)

        from PySide6.QtGui import QFont
        font = QFont("Fira Code", RIGHT_INFO_FONT)
        font.setWeight(QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor(TEXT_WHITE))

        painter.drawText(
            QRect(0, 18, self.width(), 40),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.ready_text,
        )
        painter.drawText(
            QRect(0, 76, 120, 40),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "Статус:",
        )

        indicator_radius = 8
        indicator_x = 145
        indicator_y = 96

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._indicator_color())
        painter.drawEllipse(indicator_x, indicator_y - indicator_radius, indicator_radius * 2, indicator_radius * 2)

        painter.setPen(QColor(TEXT_WHITE))
        status_text_only = self.status_text.replace("Статус:", "").strip()
        painter.drawText(
            QRect(180, 76, self.width() - 180, 40),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            status_text_only,
        )


class TerminalWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Terminal")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)

        self.is_expanded_mode = False
        self._layout_ready = False

        available = get_available_screen_geometry()
        start_w = min(WINDOW_W, available.width())
        start_h = min(WINDOW_H, available.height())

        self.setMinimumSize(760, 540)
        self.setMaximumSize(16777215, 16777215)
        self.setGeometry(available.x(), available.y(), start_w, start_h)

        self.normal_geometry = self.geometry()
        self.admin_window = None
        self.log_dir = PROJECT_ROOT / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.log_dir / "events.log"
        self.arduino = None

        try:
            if ArduinoController is not None:
                self.arduino = ArduinoController(port="COM6", baudrate=9600, min_interval=2.5)
                self.arduino.connect()
        except Exception as exc:
            self.log(f"Arduino: не удалось подключиться: {exc}")

        root = QWidget(self)
        root.setObjectName("root")
        self.setCentralWidget(root)

        self.title_bar = TitleBar(root)

        self.header_line = QFrame(root)
        self.header_line.setObjectName("headerLine")

        self.menu_button = MenuIconButton(ICON_MENU, root)

        self.camera = CameraViewport(VIDEO_X, VIDEO_Y, root, arduino=self.arduino, logger=self.log)
        self.camera.setObjectName("cameraViewport")
        self.camera.on_status_changed = self.update_status_ui

        self.menu = SideMenuCard(MENU_X, MENU_Y, root)
        self.menu.hide()
        self.menu_visible = False

        self.right_info = RightInfoPanel(root)
        self.right_info.hide()

        # Import here to avoid circular imports
        from app.ui_admin_login import AdminAuthDialog
        self.auth_dialog = AdminAuthDialog(root)

        self.title_bar.min_btn.clicked.connect(self.showMinimized)
        self.title_bar.max_btn.clicked.connect(self.toggle_expanded_mode)
        self.title_bar.close_btn.clicked.connect(self.close)

        self.menu_button.clicked.connect(self.toggle_menu)
        self.menu.admin_btn.clicked.connect(self.show_admin_auth)
        self.menu.exit_btn.clicked.connect(self.close)
        self.menu.restart_btn.clicked.connect(self.camera.start_camera)

        self._apply_styles()
        self._layout_ready = True
        self.apply_normal_layout()

    def log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {text}"
        print(line)
        try:
            with self.event_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            f"""
            #root {{
                background: {WINDOW_BG};
                border-radius: 0px;
            }}
            TitleBar {{
                background: {HEADER_BG};
                border-radius: 0px;
            }}
            #headerLine {{
                background: {LINE_BG};
            }}
            #cameraViewport {{
                background: {VIDEO_BG};
                border: {BORDER_WIDTH}px solid {BORDER_COLOR};
                border-radius: {CORNER_RADIUS}px;
            }}
            #adminAuthDialog {{
                background: {AUTH_BG};
                border: 1px solid {AUTH_BORDER};
                border-radius: {AUTH_RADIUS}px;
            }}
            #authTitle {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 24px;
                font-weight: 600;
            }}
            #authLabel {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: 12px;
                font-weight: 500;
            }}
            #authInput {{
                background: {AUTH_INPUT_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: {AUTH_RADIUS}px;
                font-family: "Fira Code";
                font-size: 18px;
                padding-left: 12px;
                padding-right: 12px;
                selection-background-color: {AUTH_LOGIN_BTN};
            }}
            #authInput:focus {{ border: 1px solid {AUTH_LOGIN_BTN}; }}
            #authLoginButton {{
                background: {AUTH_LOGIN_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: {AUTH_RADIUS}px;
                font-family: "Fira Code";
                font-size: 16px;
                font-weight: 600;
            }}
            #authLoginButton:hover {{ background: {AUTH_LOGIN_BTN_HOVER}; }}
            #authCancelButton {{
                background: {AUTH_CANCEL_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: {AUTH_RADIUS}px;
                font-family: "Fira Code";
                font-size: 16px;
                font-weight: 600;
            }}
            #authCancelButton:hover {{ background: {BTN_HOVER}; }}
            """
        )

    def _layout_scale(self, base_w: int, base_h: int):
        sx = max(self.width(), 1) / base_w
        sy = max(self.height(), 1) / base_h

        def rx(value):
            return int(value * sx)

        def ry(value):
            return int(value * sy)

        def rr(x, y, w, h):
            return QRect(rx(x), ry(y), rx(w), ry(h))

        return rx, ry, rr

    def _set_camera_inner_geometry(self) -> None:
        inner = BORDER_WIDTH
        self.camera.video_label.setGeometry(
            inner, inner,
            max(1, self.camera.width() - inner * 2),
            max(1, self.camera.height() - inner * 2),
        )
        self.camera.overlay.setGeometry(
            inner, inner,
            max(1, self.camera.width() - inner * 2),
            max(1, self.camera.height() - inner * 2),
        )
        self.camera.overlay.update()

    def update_status_ui(self, text: str, state: str) -> None:
        self.right_info.set_status(text, state)

    def show_admin_auth(self) -> None:
        self.auth_dialog.show_centered()

    def open_admin_mode(self) -> None:
        self.camera.release()

        from app.ui_admin_panel import AdminModeWindow
        self.admin_window = AdminModeWindow(self)
        available = get_available_screen_geometry()
        self.admin_window.move(
            available.x() + max(0, (available.width() - ADMIN_W) // 2),
            available.y() + max(0, (available.height() - ADMIN_H) // 2),
        )
        self.admin_window.show()
        self.hide()

    def toggle_menu(self) -> None:
        self.menu_visible = not self.menu_visible
        self.menu.setVisible(self.menu_visible)

    def _apply_menu_styles(self, font_px: int) -> None:
        self.menu.admin_btn.setStyleSheet(menu_button_style(font_px, False))
        self.menu.restart_btn.setStyleSheet(menu_button_style(font_px, False))
        self.menu.exit_btn.setStyleSheet(menu_button_style(font_px, True))

    def apply_normal_layout(self) -> None:
        rx, ry, rr = self._layout_scale(WINDOW_W, WINDOW_H)

        self.title_bar.setGeometry(rr(0, 0, WINDOW_W, HEADER_H))
        self.header_line.setGeometry(rr(0, HEADER_H, WINDOW_W, HEADER_LINE_H))

        title_btn_w = max(34, rx(TITLE_SLOT_W))
        title_btn_h = max(28, ry(TITLE_SLOT_H))

        self.title_bar.min_btn.setGeometry(self.width() - title_btn_w * 3, 0, title_btn_w, title_btn_h)
        self.title_bar.max_btn.setGeometry(self.width() - title_btn_w * 2, 0, title_btn_w, title_btn_h)
        self.title_bar.close_btn.setGeometry(self.width() - title_btn_w, 0, title_btn_w, title_btn_h)

        self.menu_button.setGeometry(rr(MENU_ICON_X, MENU_ICON_Y, MENU_ICON_W, MENU_ICON_H))
        self.menu_button.set_icon_path(ICON_MENU)
        self.menu_button.set_icon_size(QSize(max(16, rx(24)), max(16, ry(24))))
        self.menu_button.set_background(None, 0)
        self.menu_button.set_left_padding(0)

        self.camera.setGeometry(rr(VIDEO_X, VIDEO_Y, VIDEO_W, VIDEO_H))
        self._set_camera_inner_geometry()
        self.camera.overlay.show_status_text = True

        self.menu.setGeometry(rr(MENU_X, MENU_Y, MENU_W, MENU_H))
        self.menu.setStyleSheet(
            f"background: {MENU_BG}; border: {BORDER_WIDTH}px solid {BORDER_COLOR}; border-radius: {CORNER_RADIUS}px;"
        )

        btn_x = rx(17)
        btn_gap = ry(12)
        btn_w = rx(BTN_W)
        btn_h = ry(BTN_H)
        btn1_y = ry(18)
        btn2_y = btn1_y + btn_h + btn_gap
        btn3_y = btn2_y + btn_h + btn_gap

        self.menu.admin_btn.setGeometry(btn_x, btn1_y, btn_w, btn_h)
        self.menu.restart_btn.setGeometry(btn_x, btn2_y, btn_w, btn_h)
        self.menu.exit_btn.setGeometry(btn_x, btn3_y, btn_w, btn_h)
        self._apply_menu_styles(max(12, rx(16)))

        self.right_info.hide()

        if self.auth_dialog.isVisible():
            self.auth_dialog.show_centered()

    def apply_fullscreen_layout(self) -> None:
        rx, ry, rr = self._layout_scale(EXPANDED_W, EXPANDED_H)

        self.title_bar.setGeometry(rr(0, 0, EXPANDED_W, EXPANDED_HEADER_H))
        self.header_line.setGeometry(rr(0, EXPANDED_HEADER_H, EXPANDED_W, 1))

        title_btn_w = max(34, rx(TITLE_SLOT_W))
        title_btn_h = max(28, ry(TITLE_SLOT_H))

        self.title_bar.min_btn.setGeometry(self.width() - title_btn_w * 3, 0, title_btn_w, title_btn_h)
        self.title_bar.max_btn.setGeometry(self.width() - title_btn_w * 2, 0, title_btn_w, title_btn_h)
        self.title_bar.close_btn.setGeometry(self.width() - title_btn_w, 0, title_btn_w, title_btn_h)

        self.menu_button.setGeometry(rr(EXPANDED_MENU_ICON_X, EXPANDED_MENU_ICON_Y, EXPANDED_MENU_ICON_W, EXPANDED_MENU_ICON_H))
        self.menu_button.set_icon_path(ICON_MENU_MAXI)
        self.menu_button.set_icon_size(QSize(max(18, rx(30)), max(18, ry(30))))
        self.menu_button.set_background(MENU_ICON_BG_EXPANDED, 5)
        self.menu_button.set_left_padding(rx(16))

        self.camera.setGeometry(rr(EXPANDED_VIDEO_X, EXPANDED_VIDEO_Y, EXPANDED_VIDEO_W, EXPANDED_VIDEO_H))
        self._set_camera_inner_geometry()
        self.camera.overlay.show_status_text = False

        self.menu.setGeometry(rr(EXPANDED_MENU_X, EXPANDED_MENU_Y, EXPANDED_MENU_W, EXPANDED_MENU_H))
        self.menu.setStyleSheet("background: transparent; border: none;")

        btn_w = rx(EXPANDED_MENU_BTN_W)
        btn_h = ry(EXPANDED_MENU_BTN_H)
        btn_gap = ry(25)
        btn1_y = ry(34)
        btn2_y = btn1_y + btn_h + btn_gap
        btn3_y = btn2_y + btn_h + btn_gap

        self.menu.admin_btn.setGeometry(0, btn1_y, btn_w, btn_h)
        self.menu.restart_btn.setGeometry(0, btn2_y, btn_w, btn_h)
        self.menu.exit_btn.setGeometry(0, btn3_y, btn_w, btn_h)
        self._apply_menu_styles(max(14, rx(EXPANDED_MENU_BTN_FONT)))

        self.right_info.setGeometry(rr(RIGHT_INFO_X, RIGHT_INFO_Y, RIGHT_INFO_W, RIGHT_INFO_H))
        self.right_info.show()

        if self.auth_dialog.isVisible():
            self.auth_dialog.show_centered()

    def toggle_expanded_mode(self) -> None:
        if not self.is_expanded_mode:
            self.normal_geometry = self.geometry()
            available = get_available_screen_geometry()
            self.setMinimumSize(760, 540)
            self.setMaximumSize(16777215, 16777215)
            self.setGeometry(available)
            self.apply_fullscreen_layout()
            self.title_bar.max_btn.set_icon_path(ICON_RESTORE)
            self.is_expanded_mode = True
        else:
            self.setMinimumSize(760, 540)
            self.setMaximumSize(16777215, 16777215)
            self.setGeometry(self.normal_geometry)
            self.apply_normal_layout()
            self.title_bar.max_btn.set_icon_path(ICON_MAX)
            self.is_expanded_mode = False

    def resizeEvent(self, event) -> None:
        if getattr(self, "_layout_ready", False):
            if self.is_expanded_mode:
                self.apply_fullscreen_layout()
            else:
                self.apply_normal_layout()
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        self.camera.release()
        try:
            if self.arduino is not None:
                self.arduino.idle(force=True)
                self.arduino.close()
        except Exception:
            pass
        super().closeEvent(event)