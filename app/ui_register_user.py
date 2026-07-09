from __future__ import annotations

from typing import Any

import cv2
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QWidget,
)

try:
    from app.db import add_face_template, add_user, get_user_by_username
    from app.face_detector import FaceDetector
    from app.face_embedder import FaceEmbedder
    from app.face_service import crop_largest_face, process_uploaded_files, save_face_sample
except Exception as exc:
    add_face_template = add_user = get_user_by_username = None
    FaceDetector = FaceEmbedder = None
    crop_largest_face = process_uploaded_files = save_face_sample = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

from app.ui_theme import (
    AUTH_LOGIN_BTN, AUTH_LOGIN_BTN_HOVER,
    BORDER_COLOR, BORDER_WIDTH, BTN_BG, BTN_DANGER, BTN_DANGER_HOVER, BTN_HOVER,
    CLOSEBTN_HOVER, CLOSEBTN_PRESSED, CORNER_RADIUS,
    HEADER_BG, ICON_CLOSE, ICON_MAX, ICON_MIN,
    LINE_BG, MAX_SAMPLES, MIN_SAMPLES,
    REGISTER_BTN_H, REGISTER_BTN_W, REGISTER_FONT, REGISTER_H,
    REGISTER_HEADER_BTN_H, REGISTER_HEADER_BTN_W, REGISTER_HEADER_H,
    REGISTER_INPUT_BG, REGISTER_INPUT_H, REGISTER_INPUT_W,
    REGISTER_LINE_H, REGISTER_LINE_W,
    REGISTER_PANEL_H, REGISTER_PANEL_W, REGISTER_PANEL_X, REGISTER_PANEL_Y,
    REGISTER_SAVE_BTN_H,
    REGISTER_VIDEO_H, REGISTER_VIDEO_W, REGISTER_VIDEO_X, REGISTER_VIDEO_Y,
    REGISTER_W,
    SUCCESS_BOTTOM_H, SUCCESS_BOTTOM_W, SUCCESS_BTN_FONT, SUCCESS_CLOSE_BTN_H,
    SUCCESS_CLOSE_BTN_W, SUCCESS_CONTINUE_H, SUCCESS_CONTINUE_W,
    SUCCESS_DIALOG_H, SUCCESS_DIALOG_W, SUCCESS_HEADER_H, SUCCESS_HEADER_W,
    SUCCESS_TEXT_FONT,
    TEXT_WHITE, VIDEO_BG, WINDOW_BG,
    FaceOverlay, RoundedVideoLabel, ScalableWindowMixin, TitleBarButton,
    get_haarcascade_path, show_error, show_warning,
    ICON_SUCCESS,
    REGISTER_CAMERA_BTN_BG,
)


class SuccessRegistrationDialog(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("successDialog")
        self.setFixedSize(SUCCESS_DIALOG_W, SUCCESS_DIALOG_H)

        self.header = QFrame(self)
        self.header.setObjectName("successHeader")
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

        self.icon_label = QLabel(self)
        self.icon_label.setGeometry(32, 92, 48, 48)

        icon = QPixmap(str(ICON_SUCCESS))
        if not icon.isNull():
            self.icon_label.setPixmap(
                icon.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

        self.message_label = QLabel("Пользователь зарегистрирован", self)
        self.message_label.setObjectName("successMessage")
        self.message_label.setGeometry(95, 92, 345, 48)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.bottom_panel = QFrame(self)
        self.bottom_panel.setObjectName("successBottom")
        self.bottom_panel.setGeometry(1, SUCCESS_DIALOG_H - SUCCESS_BOTTOM_H, SUCCESS_BOTTOM_W, SUCCESS_BOTTOM_H)

        self.continue_btn = QPushButton("Продолжить", self.bottom_panel)
        self.continue_btn.setObjectName("successContinueButton")
        self.continue_btn.setGeometry(
            SUCCESS_BOTTOM_W - SUCCESS_CONTINUE_W - 18, 12,
            SUCCESS_CONTINUE_W, SUCCESS_CONTINUE_H,
        )

        self.close_btn.clicked.connect(self.hide)
        self.continue_btn.clicked.connect(self.hide)

        self.hide()

    def show_centered(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            x = (parent.width() - SUCCESS_DIALOG_W) // 2
            y = (parent.height() - SUCCESS_DIALOG_H) // 2
            self.move(x, y)
        self.show()
        self.raise_()


class RegisterUserWindow(QMainWindow, ScalableWindowMixin):
    def __init__(self, admin_window: Any = None) -> None:
        super().__init__()
        self.admin_window = admin_window

        self.setWindowTitle("Регистрация пользователя")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setup_scaled_window(REGISTER_W, REGISTER_H)

        self.camera_running = False
        self.capture = None
        self.face_cascade = cv2.CascadeClassifier(get_haarcascade_path())
        self.samples_count = 0
        self.samples: list[tuple[str, Any]] = []
        self.current_frame: Any = None
        self.detector = None
        self.embedder = None
        self.pipeline_error: str | None = None

        try:
            if IMPORT_ERROR is not None:
                raise RuntimeError(f"Ошибка импорта модулей проекта: {IMPORT_ERROR}")
            assert FaceDetector is not None
            assert FaceEmbedder is not None
            self.detector = FaceDetector()
            self.embedder = FaceEmbedder()
        except Exception as exc:
            self.pipeline_error = str(exc)

        self._drag_pos: QPoint | None = None

        root = QWidget(self)
        root.setObjectName("registerRoot")
        self.setCentralWidget(root)

        self.title_bar = QFrame(root)
        self.title_bar.setObjectName("registerTitleBar")
        self.title_bar.setGeometry(0, 0, REGISTER_W, REGISTER_HEADER_H)

        self.header_line = QFrame(root)
        self.header_line.setObjectName("registerHeaderLine")
        self.header_line.setGeometry(0, REGISTER_HEADER_H, REGISTER_W, 1)

        self.min_btn = TitleBarButton(ICON_MIN, parent=self.title_bar)
        self.min_btn.setGeometry(REGISTER_W - REGISTER_HEADER_BTN_W * 3, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H)

        self.max_btn = TitleBarButton(ICON_MAX, parent=self.title_bar)
        self.max_btn.setGeometry(REGISTER_W - REGISTER_HEADER_BTN_W * 2, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H)

        self.close_btn = TitleBarButton(
            ICON_CLOSE, hover_bg=CLOSEBTN_HOVER, pressed_bg=CLOSEBTN_PRESSED, parent=self.title_bar,
        )
        self.close_btn.setGeometry(REGISTER_W - REGISTER_HEADER_BTN_W, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H)

        self.video_frame = QFrame(root)
        self.video_frame.setObjectName("registerVideoFrame")
        self.video_frame.setGeometry(REGISTER_VIDEO_X, REGISTER_VIDEO_Y, REGISTER_VIDEO_W, REGISTER_VIDEO_H)

        self.video_label = RoundedVideoLabel(self.video_frame)
        self.video_label.setGeometry(BORDER_WIDTH, BORDER_WIDTH, REGISTER_VIDEO_W - 2, REGISTER_VIDEO_H - 2)

        self.overlay = FaceOverlay(self.video_frame)
        self.overlay.setGeometry(BORDER_WIDTH, BORDER_WIDTH, REGISTER_VIDEO_W - 2, REGISTER_VIDEO_H - 2)
        self.overlay.greeting = ""
        self.overlay.show_status_text = False

        self.panel = QWidget(root)
        self.panel.setGeometry(REGISTER_PANEL_X, REGISTER_PANEL_Y, REGISTER_PANEL_W, REGISTER_PANEL_H)

        self.nickname_label = QLabel("Никнейм", self.panel)
        self.nickname_label.setObjectName("registerLabel")
        self.nickname_label.setGeometry(0, 0, REGISTER_INPUT_W, 34)

        self.nickname_input = QLineEdit(self.panel)
        self.nickname_input.setObjectName("registerInput")
        self.nickname_input.setGeometry(0, 38, REGISTER_INPUT_W, REGISTER_INPUT_H)

        self.fullname_label = QLabel("ФИО", self.panel)
        self.fullname_label.setObjectName("registerLabel")
        self.fullname_label.setGeometry(0, 105, REGISTER_INPUT_W, 34)

        self.fullname_input = QLineEdit(self.panel)
        self.fullname_input.setObjectName("registerInput")
        self.fullname_input.setGeometry(0, 143, REGISTER_INPUT_W, REGISTER_INPUT_H)

        self.line_1 = QFrame(self.panel)
        self.line_1.setObjectName("registerLine")
        self.line_1.setGeometry(0, 238, REGISTER_LINE_W, REGISTER_LINE_H)

        self.camera_status_label = QLabel("Камера: не запущена", self.panel)
        self.camera_status_label.setObjectName("registerText")
        self.camera_status_label.setGeometry(0, 266, REGISTER_INPUT_W, 34)

        self.samples_label = QLabel("Образцов: 0/10", self.panel)
        self.samples_label.setObjectName("registerText")
        self.samples_label.setGeometry(0, 316, REGISTER_INPUT_W, 34)

        self.progress_frame = QFrame(self.panel)
        self.progress_frame.setObjectName("registerProgressFrame")
        self.progress_frame.setGeometry(0, 360, 210, 14)

        self.progress_fill = QFrame(self.progress_frame)
        self.progress_fill.setObjectName("registerProgressFill")
        self.progress_fill.setGeometry(0, 0, 0, 14)

        self.camera_btn = QPushButton("Запустить камеру", self.panel)
        self.camera_btn.setObjectName("registerCameraButton")
        self.camera_btn.setGeometry(0, 398, REGISTER_BTN_W, REGISTER_BTN_H)

        self.snapshot_btn = QPushButton("Сделать снимок", self.panel)
        self.snapshot_btn.setObjectName("registerButton")
        self.snapshot_btn.setGeometry(0, 468, REGISTER_BTN_W, REGISTER_BTN_H)

        self.upload_btn = QPushButton("Загрузить фото", self.panel)
        self.upload_btn.setObjectName("registerButton")
        self.upload_btn.setGeometry(0, 538, REGISTER_BTN_W, REGISTER_BTN_H)

        self.line_2 = QFrame(self.panel)
        self.line_2.setObjectName("registerLine")
        self.line_2.setGeometry(0, 620, REGISTER_LINE_W, REGISTER_LINE_H)

        self.save_btn = QPushButton("Сохранить пользователя", self.panel)
        self.save_btn.setObjectName("registerSaveButton")
        self.save_btn.setGeometry(0, 657, REGISTER_BTN_W, REGISTER_SAVE_BTN_H)

        self.clear_btn = QPushButton("Очистить образцы", self.panel)
        self.clear_btn.setObjectName("registerDangerButton")
        self.clear_btn.setGeometry(0, 754, REGISTER_BTN_W, REGISTER_BTN_H)

        self.close_window_btn = QPushButton("Закрыть", self.panel)
        self.close_window_btn.setObjectName("registerButton")
        self.close_window_btn.setGeometry(0, 829, REGISTER_BTN_W, REGISTER_BTN_H)

        self.success_dialog = SuccessRegistrationDialog(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.min_btn.clicked.connect(self.showMinimized)
        self.max_btn.clicked.connect(self.toggle_scaled_maximize)
        self.close_btn.clicked.connect(self.return_to_admin)
        self.close_window_btn.clicked.connect(self.return_to_admin)
        self.camera_btn.clicked.connect(self.toggle_camera)
        self.snapshot_btn.clicked.connect(self.capture_sample)
        self.upload_btn.clicked.connect(self.load_photos)
        self.clear_btn.clicked.connect(self.clear_samples)
        self.save_btn.clicked.connect(self.save_user)

        self.apply_scaled_geometry()
        self.apply_black_screen()
        self._apply_styles()

    def apply_scaled_geometry(self) -> None:
        self.title_bar.setGeometry(self.sr(0, 0, REGISTER_W, REGISTER_HEADER_H))
        self.header_line.setGeometry(self.sr(0, REGISTER_HEADER_H, REGISTER_W, 1))

        self.min_btn.setGeometry(self.sr(REGISTER_W - REGISTER_HEADER_BTN_W * 3, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H))
        self.max_btn.setGeometry(self.sr(REGISTER_W - REGISTER_HEADER_BTN_W * 2, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H))
        self.close_btn.setGeometry(self.sr(REGISTER_W - REGISTER_HEADER_BTN_W, 0, REGISTER_HEADER_BTN_W, REGISTER_HEADER_BTN_H))

        self.video_frame.setGeometry(self.sr(REGISTER_VIDEO_X, REGISTER_VIDEO_Y, REGISTER_VIDEO_W, REGISTER_VIDEO_H))
        self.video_label.setGeometry(self.sr(BORDER_WIDTH, BORDER_WIDTH, REGISTER_VIDEO_W - 2, REGISTER_VIDEO_H - 2))
        self.overlay.setGeometry(self.sr(BORDER_WIDTH, BORDER_WIDTH, REGISTER_VIDEO_W - 2, REGISTER_VIDEO_H - 2))

        self.panel.setGeometry(self.sr(REGISTER_PANEL_X, REGISTER_PANEL_Y, REGISTER_PANEL_W, REGISTER_PANEL_H))

        self.nickname_label.setGeometry(self.sr(0, 0, REGISTER_INPUT_W, 34))
        self.nickname_input.setGeometry(self.sr(0, 38, REGISTER_INPUT_W, REGISTER_INPUT_H))
        self.fullname_label.setGeometry(self.sr(0, 105, REGISTER_INPUT_W, 34))
        self.fullname_input.setGeometry(self.sr(0, 143, REGISTER_INPUT_W, REGISTER_INPUT_H))

        self.line_1.setGeometry(self.sr(0, 238, REGISTER_LINE_W, REGISTER_LINE_H))
        self.camera_status_label.setGeometry(self.sr(0, 266, REGISTER_INPUT_W, 34))
        self.samples_label.setGeometry(self.sr(0, 316, REGISTER_INPUT_W, 34))

        self.progress_frame.setGeometry(self.sr(0, 360, 210, 14))
        progress_w = int(210 * len(self.samples) / MAX_SAMPLES)
        self.progress_fill.setGeometry(self.sr(0, 0, progress_w, 14))

        self.camera_btn.setGeometry(self.sr(0, 398, REGISTER_BTN_W, REGISTER_BTN_H))
        self.snapshot_btn.setGeometry(self.sr(0, 468, REGISTER_BTN_W, REGISTER_BTN_H))
        self.upload_btn.setGeometry(self.sr(0, 538, REGISTER_BTN_W, REGISTER_BTN_H))
        self.line_2.setGeometry(self.sr(0, 620, REGISTER_LINE_W, REGISTER_LINE_H))
        self.save_btn.setGeometry(self.sr(0, 657, REGISTER_BTN_W, REGISTER_SAVE_BTN_H))
        self.clear_btn.setGeometry(self.sr(0, 754, REGISTER_BTN_W, REGISTER_BTN_H))
        self.close_window_btn.setGeometry(self.sr(0, 829, REGISTER_BTN_W, REGISTER_BTN_H))

    def resizeEvent(self, event) -> None:
        if not hasattr(self, "title_bar"):
            super().resizeEvent(event)
            return
        self.scale_x = max(self.width(), 1) / REGISTER_W
        self.scale_y = max(self.height(), 1) / REGISTER_H
        self.apply_scaled_geometry()
        if not self.camera_running:
            self.apply_black_screen()
        super().resizeEvent(event)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            f"""
            #registerRoot {{ background: {WINDOW_BG}; }}
            #registerTitleBar {{ background: {HEADER_BG}; }}
            #registerHeaderLine {{ background: {LINE_BG}; }}
            #registerVideoFrame {{
                background: {VIDEO_BG};
                border: {BORDER_WIDTH}px solid {BORDER_COLOR};
                border-radius: {CORNER_RADIUS}px;
            }}
            #registerLabel, #registerText {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                font-weight: 500;
            }}
            #registerInput {{
                background: {REGISTER_INPUT_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                padding-left: 12px;
                padding-right: 12px;
            }}
            #registerInput:focus {{ border: 1px solid {AUTH_LOGIN_BTN}; }}
            #registerLine {{ background: {LINE_BG}; }}
            #registerButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                font-weight: 500;
            }}
            #registerButton:hover {{ background: {BTN_HOVER}; }}
            #registerCameraButton {{
                background: {REGISTER_CAMERA_BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                font-weight: 500;
            }}
            #registerCameraButton:hover {{ background: {BTN_HOVER}; }}
            #registerSaveButton {{
                background: {AUTH_LOGIN_BTN};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                font-weight: 500;
            }}
            #registerSaveButton:hover {{ background: {AUTH_LOGIN_BTN_HOVER}; }}
            #registerDangerButton {{
                background: {BTN_DANGER};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {REGISTER_FONT}px;
                font-weight: 500;
            }}
            #registerDangerButton:hover {{ background: {BTN_DANGER_HOVER}; }}
            #registerProgressFrame {{
                background: transparent;
                border: 1px solid {BORDER_COLOR};
            }}
            #registerProgressFill {{ background: {AUTH_LOGIN_BTN}; }}
            #successDialog {{
                background: {WINDOW_BG};
                border: 1px solid {LINE_BG};
                border-radius: 0px;
            }}
            #successHeader {{
                background: {WINDOW_BG};
                border-bottom: 1px solid {LINE_BG};
            }}
            #successBottom {{
                background: {WINDOW_BG};
                border-top: 1px solid {LINE_BG};
            }}
            #successMessage {{
                color: {TEXT_WHITE};
                font-family: "Fira Code";
                font-size: {SUCCESS_TEXT_FONT}px;
                font-weight: 500;
            }}
            #successContinueButton {{
                background: {BTN_BG};
                color: {TEXT_WHITE};
                border: none;
                border-radius: 5px;
                font-family: "Fira Code";
                font-size: {SUCCESS_BTN_FONT}px;
                font-weight: 500;
            }}
            #successContinueButton:hover {{ background: {BTN_HOVER}; }}
            """
        )

    def show_success_dialog(self) -> None:
        self.success_dialog.show_centered()

    def _set_info(self, text: str) -> None:
        self.camera_status_label.setText(text)

    def _update_samples_ui(self) -> None:
        count = len(self.samples)
        self.samples_count = count
        self.samples_label.setText(f"Образцов: {count}/{MAX_SAMPLES}")
        width = int(210 * count / MAX_SAMPLES)
        self.progress_fill.setGeometry(self.sr(0, 0, width, 14))

    def _validate_fields(self) -> tuple[str, str] | None:
        username = self.nickname_input.text().strip()
        full_name = self.fullname_input.text().strip()
        if not username:
            show_warning(self, "Ошибка", "Введите никнейм.")
            return None
        if not full_name:
            show_warning(self, "Ошибка", "Введите ФИО.")
            return None
        return username, full_name

    def capture_sample(self) -> None:
        validated = self._validate_fields()
        if validated is None:
            return
        username, _ = validated
        if self.pipeline_error:
            show_error(self, "Ошибка", f"Пайплайн регистрации не загружен:\n{self.pipeline_error}")
            return
        if not self.camera_running or self.current_frame is None:
            show_warning(self, "Ошибка", "Сначала запустите камеру.")
            return
        if len(self.samples) >= MAX_SAMPLES:
            show_warning(self, "Ограничение", f"Максимум {MAX_SAMPLES} образцов.")
            return
        if self.detector is None or self.embedder is None or crop_largest_face is None or save_face_sample is None:
            show_warning(self, "Ошибка", "Пайплайн распознавания не загружен.")
            return

        face = crop_largest_face(self.current_frame, self.detector, margin=0.2)
        if face is None:
            show_warning(self, "Ошибка", "Лицо не найдено.")
            return

        saved = save_face_sample(
            username=username,
            face_bgr=face,
            embedder=self.embedder,
            sample_index=len(self.samples),
        )
        if saved is None:
            show_warning(self, "Ошибка", "Не удалось построить эмбеддинг.")
            return
        self.samples.append(saved)
        self._update_samples_ui()
        self._set_info(f"Образец добавлен: {len(self.samples)}/{MAX_SAMPLES}")

    def load_photos(self) -> None:
        validated = self._validate_fields()
        if validated is None:
            return
        username, _ = validated
        if self.pipeline_error:
            show_error(self, "Ошибка", f"Пайплайн регистрации не загружен:\n{self.pipeline_error}")
            return
        remaining = MAX_SAMPLES - len(self.samples)
        if remaining <= 0:
            show_warning(self, "Ограничение", f"Максимум {MAX_SAMPLES} образцов.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите фотографии", "",
            "Images (*.jpg *.jpeg *.png *.bmp);;All files (*.*)",
        )
        if not paths:
            return
        if self.detector is None or self.embedder is None or process_uploaded_files is None:
            show_warning(self, "Ошибка", "Пайплайн распознавания не загружен.")
            return

        loaded = process_uploaded_files(
            username=username,
            file_paths=list(paths),
            detector=self.detector,
            embedder=self.embedder,
            start_index=len(self.samples),
            limit=remaining,
        )
        if not loaded:
            show_warning(self, "Результат", "Не удалось обработать выбранные файлы.")
            return
        self.samples.extend(loaded)
        self._update_samples_ui()
        self._set_info(f"Добавлено образцов: {len(loaded)}")

    def clear_samples(self) -> None:
        self.samples.clear()
        self._update_samples_ui()
        self._set_info("Образцы очищены")

    def save_user(self) -> None:
        validated = self._validate_fields()
        if validated is None:
            return
        username, full_name = validated
        if len(self.samples) < MIN_SAMPLES:
            show_warning(self, "Ошибка", f"Нужно минимум {MIN_SAMPLES} образца.")
            return
        if get_user_by_username is None or add_user is None or add_face_template is None:
            show_error(self, "Ошибка", f"Модуль базы данных не загружен:\n{IMPORT_ERROR}")
            return
        if get_user_by_username(username) is not None:
            show_warning(self, "Ошибка", "Пользователь с таким никнеймом уже существует.")
            return
        try:
            user_id = add_user(username=username, full_name=full_name)
            for photo_path, embedding in self.samples:
                add_face_template(user_id=user_id, photo_path=photo_path, embedding=embedding)
        except Exception as exc:
            show_error(self, "Ошибка", f"Не удалось сохранить пользователя:\n{exc}")
            return
        self.show_success_dialog()
        self.nickname_input.clear()
        self.fullname_input.clear()
        self.clear_samples()

    def apply_black_screen(self) -> None:
        black = QPixmap(self.video_label.width(), self.video_label.height())
        black.fill(QColor("#000000"))
        self.video_label.setVideoPixmap(black)
        self.overlay.set_face_rect(None)

    def toggle_camera(self) -> None:
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        self.stop_camera()
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        try:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        except Exception:
            pass
        self.camera_running = True
        self.camera_status_label.setText("Камера: активна")
        self.camera_btn.setText("Остановить камеру")
        self.timer.start(30)

    def stop_camera(self) -> None:
        self.timer.stop()
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        self.capture = None
        self.camera_running = False
        self.camera_status_label.setText("Камера: не запущена")
        self.camera_btn.setText("Запустить камеру")
        self.apply_black_screen()

    def update_frame(self) -> None:
        if self.capture is None or not self.capture.isOpened():
            self.stop_camera()
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        self.show_frame(frame)
        self.detect_face(frame)

    def show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setVideoPixmap(scaled)

    def detect_face(self, frame) -> None:
        if self.face_cascade.empty():
            self.overlay.set_face_rect(None)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(90, 90))
        if len(faces) == 0:
            self.overlay.set_face_rect(None)
            return
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        frame_h, frame_w = frame.shape[:2]
        target_w = self.overlay.width()
        target_h = self.overlay.height()
        scale = max(target_w / frame_w, target_h / frame_h)
        scaled_w = frame_w * scale
        scaled_h = frame_h * scale
        offset_x = (scaled_w - target_w) / 2
        offset_y = (scaled_h - target_h) / 2
        draw_x = int(x * scale - offset_x)
        draw_y = int(y * scale - offset_y)
        draw_w = int(w * scale)
        draw_h = int(h * scale)
        self.overlay.set_face_rect(QRect(draw_x, draw_y, draw_w, draw_h))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= REGISTER_HEADER_H:
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
        self.stop_camera()
        if self.admin_window is not None:
            self.admin_window.show()
        self.close()

    def closeEvent(self, event) -> None:
        self.stop_camera()
        super().closeEvent(event)