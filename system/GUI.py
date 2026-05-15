import csv
import math
import os
import shutil
import sys
from functools import partial

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, QUrl, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
inference_path = os.path.join(current_dir, "muti_input")
music_path = os.path.join(current_dir, "music")

if inference_path not in sys.path:
    sys.path.append(inference_path)
if music_path not in sys.path:
    sys.path.append(music_path)

try:
    from inference import EMOTIONS, InferenceService
except ImportError as e:
    print(f"❌ 导入失败，请检查推理模块配置: {e}")
    EMOTIONS = ["Anger", "Frustrated", "Neutral", "Happiness", "Excited", "Sadness"]
    InferenceService = None

try:
    from make_index import MusicEmotionIndexer, refresh_music_index, recheck_song_entry
except ImportError as e:
    print(f"检测索引失败: {e}")
    MusicEmotionIndexer = None
    refresh_music_index = None
    recheck_song_entry = None


class EmotionPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(520)
        self.setMinimumWidth(420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_point = [0.0, 0.0]
        self.start_point = [0.0, 0.0]
        self.target_point = [0.0, 0.0]
        self.current_emoji = "🙂"
        self.animation_step = 0
        self.animation_steps = 18
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_animation)

    def reset(self):
        self.timer.stop()
        self.current_point = [0.0, 0.0]
        self.start_point = [0.0, 0.0]
        self.target_point = [0.0, 0.0]
        self.current_emoji = "🙂"
        self.update()

    def set_emotion(self, valence, arousal, emoji):
        self.start_point = self.current_point[:]
        self.target_point = [max(-1.0, min(1.0, float(valence))), max(-1.0, min(1.0, float(arousal)))]
        self.current_emoji = emoji
        self.animation_step = 0
        self.timer.start(16)
        self.update()

    def _advance_animation(self):
        self.animation_step += 1
        progress = min(1.0, self.animation_step / self.animation_steps)
        self.current_point = [
            self.start_point[0] + (self.target_point[0] - self.start_point[0]) * progress,
            self.start_point[1] + (self.target_point[1] - self.start_point[1]) * progress,
        ]
        self.update()
        if progress >= 1.0:
            self.timer.stop()

    def _point_to_canvas(self, valence, arousal, rect):
        x = rect.left() + (valence + 1) * 0.5 * rect.width()
        y = rect.bottom() - (arousal + 1) * 0.5 * rect.height()
        return x, y

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f9fbff"))
        outer_rect = self.rect().adjusted(18, 18, -18, -18)
        square_side = min(outer_rect.width() - 40, outer_rect.height() - 56)
        square_side = max(square_side, 120)
        left = outer_rect.left() + (outer_rect.width() - square_side) / 2
        top = outer_rect.top() + (outer_rect.height() - square_side) / 2 - 8
        plot_rect = outer_rect.adjusted(0, 0, 0, 0)
        plot_rect.setLeft(int(left))
        plot_rect.setTop(int(top))
        plot_rect.setWidth(int(square_side))
        plot_rect.setHeight(int(square_side))
        painter.setPen(QPen(QColor("#dfe7f2"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(plot_rect, 18, 18)
        painter.setPen(QPen(QColor("#d5dfec"), 1, Qt.DashLine))
        for fraction in (0.25, 0.5, 0.75):
            x = plot_rect.left() + plot_rect.width() * fraction
            y = plot_rect.top() + plot_rect.height() * fraction
            painter.drawLine(int(x), plot_rect.top(), int(x), plot_rect.bottom())
            painter.drawLine(plot_rect.left(), int(y), plot_rect.right(), int(y))
        painter.setPen(QPen(QColor("#62748a"), 2))
        center_x, center_y = self._point_to_canvas(0.0, 0.0, plot_rect)
        painter.drawLine(int(center_x), plot_rect.top(), int(center_x), plot_rect.bottom())
        painter.drawLine(plot_rect.left(), int(center_y), plot_rect.right(), int(center_y))
        painter.setPen(QColor("#718399"))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.drawText(plot_rect.center().x() - 18, plot_rect.top() - 8, "高唤醒")
        painter.drawText(plot_rect.center().x() - 18, plot_rect.bottom() + 22, "低唤醒")
        painter.drawText(plot_rect.left() - 2, plot_rect.bottom() + 22, "低效价")
        painter.drawText(plot_rect.right() - 34, plot_rect.bottom() + 22, "高效价")
        x, y = self._point_to_canvas(self.current_point[0], self.current_point[1], plot_rect)
        painter.setFont(QFont("Segoe UI Emoji", 26))
        painter.drawText(int(x - 16), int(y + 12), self.current_emoji)


class AutoResizingTextEdit(QTextEdit):
    submit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.document().documentLayout().documentSizeChanged.connect(self.update_height)
        self.update_height()

    def update_height(self):
        document_height = self.document().size().height()
        self.setFixedHeight(int(max(58, min(180, document_height + 22))))

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            self.submit_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
        self.update_height()


class ChatInputPanel(QFrame):
    image_dropped = pyqtSignal(str)
    submit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.drag_active = False
        self.image_path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.attachment_label = QLabel("未附加图片")
        self.attachment_label.setStyleSheet("color: #8b98aa; font-size: 12px;")
        layout.addWidget(self.attachment_label)

        self.text_edit = AutoResizingTextEdit()
        self.text_edit.setPlaceholderText("输入你的心情，或把图片拖进这个输入框区域。Enter 发送，Shift+Enter 换行。")
        self.text_edit.setStyleSheet("border: none; background: transparent; font-size: 14px; padding: 0; color: #243244;")
        self.text_edit.submit_requested.connect(self.submit_requested.emit)
        layout.addWidget(self.text_edit)

        bottom_row = QHBoxLayout()
        self.attach_button = QToolButton()
        self.attach_button.setText("🖼")
        self.attach_button.setToolTip("选择或拖入图片")
        self.attach_button.setCursor(Qt.PointingHandCursor)
        self.attach_button.setAutoRaise(True)
        self.attach_button.setStyleSheet("font-size: 18px; color: #607086; padding: 4px;")
        bottom_row.addWidget(self.attach_button)

        self.tip_label = QLabel("支持文本或图片单独输入，也支持双模态联合推理。")
        self.tip_label.setStyleSheet("color: #93a0b0; font-size: 12px;")
        bottom_row.addWidget(self.tip_label, 1)

        self.send_button = QToolButton()
        self.send_button.setText("➜")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setStyleSheet(
            "background: #165dff; color: white; border-radius: 16px; font-size: 16px; padding: 8px 14px;"
        )
        bottom_row.addWidget(self.send_button)
        layout.addLayout(bottom_row)
        self._apply_style()

    def _apply_style(self):
        border_color = "#bfd5ff" if self.drag_active else "#d7e1ee"
        background = "#eef5ff" if self.drag_active else "#ffffff"
        self.setStyleSheet(f"QFrame {{ background: {background}; border: 1px solid {border_color}; border-radius: 20px; }}")

    def set_image_path(self, path):
        self.image_path = path or ""
        if self.image_path:
            self.attachment_label.setText(f"已附加图片: {os.path.basename(self.image_path)}")
            self.attachment_label.setStyleSheet("color: #46607f; font-size: 12px;")
        else:
            self.attachment_label.setText("未附加图片")
            self.attachment_label.setStyleSheet("color: #8b98aa; font-size: 12px;")

    def clear_input(self):
        self.text_edit.clear()
        self.text_edit.update_height()
        self.set_image_path("")

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and self._is_supported_image(urls[0].toLocalFile()):
            self.drag_active = True
            self._apply_style()
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.drag_active = False
        self._apply_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.drag_active = False
        self._apply_style()
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        local_path = urls[0].toLocalFile()
        if self._is_supported_image(local_path):
            self.image_dropped.emit(local_path)
            event.acceptProposedAction()
            return
        event.ignore()

    @staticmethod
    def _is_supported_image(path):
        return bool(path) and path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))


class InferenceWorker(QObject):
    inference_finished = pyqtSignal(object)
    inference_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.service = None

    @pyqtSlot(str, str)
    def run_inference(self, image_path, text):
        if InferenceService is None:
            self.inference_failed.emit("推理模块未正确加载，无法执行情感分析。")
            return
        try:
            if self.service is None:
                self.service = InferenceService()
            result = self.service.infer(image_path=image_path or None, text=text or None)
            if result is None:
                self.inference_failed.emit("未检测到可用输入，请至少提供图片或文本。")
                return
            self.inference_finished.emit(result)
        except Exception as e:
            self.inference_failed.emit(str(e))


class MusicEmotionApp(QMainWindow):
    request_inference = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("双模态智能情绪识别音乐系统")
        self.setMinimumSize(1400, 900)
        self.resize(1400, 900)

        self.system_dir = current_dir
        self.music_dir = os.path.join(self.system_dir, "music")
        self.music_upload_dir = os.path.join(self.music_dir, "upload")
        self.csv_path = os.path.join(self.music_dir, "music_emotion_index.csv")
        os.makedirs(self.music_upload_dir, exist_ok=True)

        self.selected_image_path = ""
        self.music_rows = []
        self.display_rows = []
        self.filtered_rows = []
        self.recommended_song_ids = []
        self.current_track_index = -1
        self.latest_result = None
        self.is_slider_dragging = False
        self.loader_frames = ["◜", "◠", "◝", "◞", "◡", "◟"]
        self.loader_index = 0
        self.music_indexer = None

        self.player = QMediaPlayer(self)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)
        self.player.mediaStatusChanged.connect(self.handle_media_status)

        self.send_loader_timer = QTimer(self)
        self.send_loader_timer.timeout.connect(self.animate_send_button)

        self.inference_thread = QThread(self)
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)
        self.request_inference.connect(self.inference_worker.run_inference)
        self.inference_worker.inference_finished.connect(self.handle_inference_result)
        self.inference_worker.inference_failed.connect(self.handle_inference_error)
        self.inference_thread.start()

        self.init_ui()
        self.load_music_data()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.setStyleSheet(
            """
            QWidget { background: #edf2f7; color: #1f2d3d; }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #dbe4ef;
                border-radius: 22px;
                margin-top: 14px;
                padding-top: 18px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 6px;
            }
            QLineEdit, QTableWidget {
                background: #ffffff;
                border: 1px solid #d7e1ee;
                border-radius: 16px;
                padding: 8px 12px;
            }
            QTableWidget {
                gridline-color: #edf2f7;
                selection-background-color: #e4eeff;
                selection-color: #1f2d3d;
            }
            QHeaderView::section {
                background: #f4f8fd;
                color: #506073;
                border: none;
                padding: 8px;
                font-weight: 600;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #d7e1ef;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
                background: #165dff;
            }
            QPushButton#sidebarButton {
                background: #ffffff;
                color: #263445;
                border: 1px solid #d6dfeb;
                border-radius: 16px;
                padding: 12px 10px;
                font-weight: 600;
            }
            QPushButton#sidebarButton:hover {
                background: #f5f9ff;
                border-color: #bcd0f4;
            }
            QToolButton#transportButton {
                background: #f4f8fe;
                color: #2c3b4c;
                border: 1px solid #dbe5f1;
                border-radius: 18px;
                padding: 8px 10px;
                font-size: 18px;
            }
            QToolButton#transportButton:hover {
                background: #e6f0ff;
                border-color: #bcd0f4;
            }
            """
        )

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        sidebar = self.build_sidebar_column()
        sidebar.setFixedWidth(128)

        content_column = QWidget()
        content_layout = QVBoxLayout(content_column)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        content_layout.addWidget(self.build_plot_card(), 8)
        content_layout.addWidget(self.build_composer_card(), 2)

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self.build_library_column(), 7)
        right_layout.addWidget(self.build_player_card(), 3)

        root_layout.addWidget(sidebar, 0)
        root_layout.addWidget(content_column, 7)
        root_layout.addWidget(right_column, 7)
        self.statusBar().showMessage("等待文本或图片输入。")

    def build_plot_card(self):
        box = QGroupBox("情绪坐标")
        layout = QVBoxLayout(box)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        self.va_value_label = QLabel("尚未推理")
        self.va_value_label.setFont(QFont("Microsoft YaHei", 13, QFont.Bold))
        title_row.addWidget(self.va_value_label)

        self.va_help_button = QToolButton()
        self.va_help_button.setText("?")
        self.va_help_button.setCursor(Qt.PointingHandCursor)
        self.va_help_button.setAutoRaise(True)
        self.va_help_button.setToolTip("点击查看 Valence / Arousal 的含义")
        self.va_help_button.setStyleSheet("color: #a1acb8; font-size: 15px; padding: 0 4px;")
        self.va_help_button.clicked.connect(self.show_va_help)
        title_row.addWidget(self.va_help_button)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.branch_label = QLabel("图像分支: - | 文本分支: - | 融合权重: -")
        self.branch_label.setStyleSheet("color: #7a8899;")
        layout.addWidget(self.branch_label)

        self.recommendation_label = QLabel("完成一次推理后，这里会显示当前情绪下最接近的推荐歌曲。")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet(
            "background: #f4f8fd; border: 1px solid #e2eaf4; border-radius: 14px; padding: 12px; color: #46566a;"
        )
        layout.addWidget(self.recommendation_label)

        self.plot_widget = EmotionPlotWidget()
        layout.addWidget(self.plot_widget, 1)
        return box

    def build_player_card(self):
        box = QGroupBox("播放器")
        layout = QVBoxLayout(box)
        layout.setSpacing(14)

        self.now_playing_label = QLabel("当前未播放歌曲")
        self.now_playing_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        self.player_hint_label = QLabel("空格可暂停/继续，播放结束自动切到下一首。")
        self.player_hint_label.setStyleSheet("color: #7d8c9f;")
        layout.addWidget(self.now_playing_label)
        layout.addWidget(self.player_hint_label)

        progress_row = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.progress_slider.sliderMoved.connect(self.preview_seek_position)
        self.total_time_label = QLabel("00:00")
        progress_row.addWidget(self.current_time_label)
        progress_row.addWidget(self.progress_slider, 1)
        progress_row.addWidget(self.total_time_label)
        layout.addLayout(progress_row)

        control_row = QHBoxLayout()
        control_row.addStretch(1)
        self.prev_button = self.create_transport_button("⏮", self.play_previous, "上一首")
        self.pause_button = self.create_transport_button("⏯", self.toggle_playback, "暂停/继续")
        self.next_button = self.create_transport_button("⏭", self.play_next, "下一首")
        control_row.addWidget(self.prev_button)
        control_row.addWidget(self.pause_button)
        control_row.addWidget(self.next_button)
        control_row.addStretch(1)
        layout.addLayout(control_row)
        return box

    def build_composer_card(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.chat_input = ChatInputPanel()
        self.chat_input.attach_button.clicked.connect(self.select_image)
        self.chat_input.send_button.clicked.connect(self.run_inference)
        self.chat_input.image_dropped.connect(self.attach_image)
        self.chat_input.submit_requested.connect(self.run_inference)
        layout.addWidget(self.chat_input)
        return container

    def build_library_column(self):
        box = QGroupBox("曲库")
        layout = QVBoxLayout(box)
        layout.setSpacing(12)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索歌曲名")
        self.search_input.textChanged.connect(self.apply_music_filter)
        layout.addWidget(self.search_input)

        self.library_count_label = QLabel("0 首歌曲")
        self.library_count_label.setStyleSheet("color: #8391a3;")
        layout.addWidget(self.library_count_label)

        self.music_table = QTableWidget()
        self.music_table.setColumnCount(6)
        self.music_table.setHorizontalHeaderLabels(["歌曲", "Valence", "Arousal", "距离", "▶"])
        self.music_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.music_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.music_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.music_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.music_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.music_table.setAlternatingRowColors(True)
        self.music_table.cellDoubleClicked.connect(self.play_from_table_row)
        layout.addWidget(self.music_table, 1)
        return box

    def build_sidebar_column(self):
        panel = QFrame()
        panel.setStyleSheet("QFrame { background: #ffffff; border: 1px solid #dbe4ef; border-radius: 24px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 14)
        layout.setSpacing(12)

        title = QLabel("操作")
        title.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        self.analyze_button = QPushButton("分析")
        self.analyze_button.setObjectName("sidebarButton")
        self.analyze_button.clicked.connect(self.run_inference)
        layout.addWidget(self.analyze_button)

        self.upload_button = QPushButton("上传曲库")
        self.upload_button.setObjectName("sidebarButton")
        self.upload_button.clicked.connect(self.upload_music)
        layout.addWidget(self.upload_button)

        self.refresh_button = QPushButton("刷新索引")
        self.refresh_button.setObjectName("sidebarButton")
        self.refresh_button.clicked.connect(self.load_music_data)
        layout.addWidget(self.refresh_button)

        self.clear_image_button = QPushButton("移除图片")
        self.clear_image_button.setObjectName("sidebarButton")
        self.clear_image_button.clicked.connect(lambda: self.attach_image(""))
        layout.addWidget(self.clear_image_button)

        self.sidebar_state_label = QLabel("图像未附加")
        self.sidebar_state_label.setWordWrap(True)
        self.sidebar_state_label.setAlignment(Qt.AlignCenter)
        self.sidebar_state_label.setStyleSheet("color: #92a0b2; padding: 10px 4px;")
        layout.addWidget(self.sidebar_state_label)
        layout.addStretch(1)
        return panel

    def create_transport_button(self, text, handler, tooltip):
        button = QToolButton()
        button.setObjectName("transportButton")
        button.setText(text)
        button.setToolTip(tooltip)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(handler)
        return button

    def start_loading_indicator(self):
        self.loader_index = 0
        self.chat_input.send_button.setEnabled(False)
        self.chat_input.send_button.setText(self.loader_frames[self.loader_index])
        self.chat_input.send_button.setStyleSheet(
            "background: #d4dae3; color: #5e6875; border-radius: 16px; font-size: 17px; padding: 8px 14px;"
        )
        self.send_loader_timer.start(120)

    def stop_loading_indicator(self):
        self.send_loader_timer.stop()
        self.chat_input.send_button.setEnabled(True)
        self.chat_input.send_button.setText("➜")
        self.chat_input.send_button.setStyleSheet(
            "background: #165dff; color: white; border-radius: 16px; font-size: 16px; padding: 8px 14px;"
        )

    def animate_send_button(self):
        self.loader_index = (self.loader_index + 1) % len(self.loader_frames)
        self.chat_input.send_button.setText(self.loader_frames[self.loader_index])

    def show_va_help(self):
        QMessageBox.information(
            self,
            "VA 坐标说明",
            "Valence 表示情绪的积极或消极程度，越靠右越积极。\n"
            "Arousal 表示唤醒强度，越靠上越兴奋。\n"
            "图中的 emoji 会移动到当前推理出的情绪坐标。",
        )

    def attach_image(self, path):
        self.selected_image_path = path or ""
        self.chat_input.set_image_path(self.selected_image_path)
        if self.selected_image_path:
            self.sidebar_state_label.setText(f"已附加图片\n{os.path.basename(self.selected_image_path)}")
            self.statusBar().showMessage("图片已附加，等待推理。")
        else:
            self.sidebar_state_label.setText("图像未附加")
            self.statusBar().showMessage("已移除图片附件。")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.attach_image(path)

    def upload_music(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音乐文件",
            "",
            "Audio (*.mp3 *.wav *.m4a *.flac *.ogg)",
        )
        if not paths:
            return

        copied_count = 0
        for path in paths:
            try:
                shutil.copy2(path, self.music_upload_dir)
                copied_count += 1
            except Exception as e:
                QMessageBox.warning(self, "上传失败", f"复制文件失败：{os.path.basename(path)}\n{e}")

        QMessageBox.information(
            self,
            "上传完成",
            f"已复制 {copied_count} 首音乐到上传目录。\n如需出现在曲库中，请重新生成音乐情绪索引。",
        )

    def normalize_song_path(self, raw_path):
        raw_path = (raw_path or "").strip()
        if not raw_path:
            return ""
        if os.path.isabs(raw_path):
            return os.path.normpath(raw_path)
        return os.path.normpath(os.path.join(self.music_dir, raw_path))

    def load_music_data(self):
        self.music_rows = []
        self.recommended_song_ids = []
        if not os.path.exists(self.csv_path):
            self.display_rows = []
            self.filtered_rows = []
            self.refresh_music_table()
            self.recommendation_label.setText("未找到 music_emotion_index.csv，请先生成索引。")
            return

        try:
            with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    try:
                        self.music_rows.append(
                            {
                                "song_id": str(row.get("song_id", "")).strip(),
                                "valence": float(row.get("valence", 0.0)),
                                "arousal": float(row.get("arousal", 0.0)),
                                "file_path": self.normalize_song_path(row.get("file_path", "")),
                                "distance": None,
                            }
                        )
                    except (TypeError, ValueError):
                        continue
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"读取音乐索引失败：\n{e}")

        self.display_rows = list(self.music_rows)
        self.apply_music_filter(self.search_input.text())
        self.statusBar().showMessage(f"已加载 {len(self.music_rows)} 首歌曲。")

    def apply_music_filter(self, keyword):
        keyword = (keyword or "").strip().lower()
        source_rows = self.display_rows if self.display_rows else self.music_rows
        if keyword:
            self.filtered_rows = [row for row in source_rows if keyword in row["song_id"].lower()]
        else:
            self.filtered_rows = list(source_rows)
        self.refresh_music_table()

    def refresh_music_table(self):
        rows = self.filtered_rows if self.filtered_rows is not None else self.music_rows
        self.music_table.setRowCount(len(rows))
        self.library_count_label.setText(f"{len(rows)} 首歌曲")

        for row_index, row in enumerate(rows):
            self.music_table.setItem(row_index, 0, QTableWidgetItem(row["song_id"]))
            self.music_table.setItem(row_index, 1, QTableWidgetItem(f"{row['valence']:.3f}"))
            self.music_table.setItem(row_index, 2, QTableWidgetItem(f"{row['arousal']:.3f}"))
            distance = "-" if row.get("distance") is None else f"{row['distance']:.3f}"
            self.music_table.setItem(row_index, 3, QTableWidgetItem(distance))

            play_button = QToolButton()
            play_button.setText("▶")
            play_button.setCursor(Qt.PointingHandCursor)
            play_button.setAutoRaise(True)
            play_button.clicked.connect(partial(self.play_from_table_row, row_index, 0))
            self.music_table.setCellWidget(row_index, 4, play_button)

        if rows:
            self.music_table.selectRow(0)

    def run_inference(self):
        text_content = self.chat_input.text_edit.toPlainText().strip()
        if not self.selected_image_path and not text_content:
            QMessageBox.information(self, "缺少输入", "请至少输入文本，或拖入一张图片。")
            return

        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("分析中")
        self.start_loading_indicator()
        self.statusBar().showMessage("正在执行双模态推理，请稍候。")
        self.request_inference.emit(self.selected_image_path, text_content)

    def handle_inference_result(self, result):
        self.latest_result = result
        final_va = result["final_va"]
        valence = float(final_va[0])
        arousal = float(final_va[1])
        emoji, summary = self.resolve_emoji_and_summary(valence, arousal)
        self.va_value_label.setText(f"{emoji}  当前坐标 ({valence:.3f}, {arousal:.3f}) · {summary}")

        details = result["details"]
        img_label = self.extract_branch_label(details.get("img"))
        txt_label = self.extract_branch_label(details.get("txt"))
        w_img, w_txt = result["weights"]
        self.branch_label.setText(
            f"图像分支: {img_label} | 文本分支: {txt_label} | 融合权重: 图像 {w_img:.2f} / 文本 {w_txt:.2f}"
        )
        self.plot_widget.set_emotion(valence, arousal, emoji)
        self.rank_music_by_emotion(valence, arousal)

        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("分析")
        self.stop_loading_indicator()
        self.statusBar().showMessage("推理完成，推荐结果已更新。")
        self.chat_input.clear_input()
        self.attach_image("")

    def handle_inference_error(self, error_message):
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("分析")
        self.stop_loading_indicator()
        self.statusBar().showMessage("推理失败。")
        QMessageBox.warning(self, "推理失败", error_message)

    def extract_branch_label(self, branch_result):
        if not branch_result:
            return "-"
        label_index = int(branch_result.get("max_idx", 2))
        return EMOTIONS[label_index] if 0 <= label_index < len(EMOTIONS) else "-"

    def resolve_emoji_and_summary(self, valence, arousal):
        if valence >= 0.35 and arousal >= 0.25:
            return "😄", "明亮高能"
        if valence < -0.25 and arousal >= 0.2:
            return "😠", "紧张激烈"
        if valence < -0.2 and arousal < -0.1:
            return "😢", "低落安静"
        if valence >= 0.2 and arousal < -0.15:
            return "😌", "放松平和"
        if abs(valence) <= 0.15 and abs(arousal) <= 0.15:
            return "😐", "中性稳定"
        if valence >= 0:
            return "🙂", "偏积极"
        return "🙁", "偏消极"

    def rank_music_by_emotion(self, valence, arousal):
        if not self.music_rows:
            self.recommendation_label.setText("当前曲库为空，无法生成推荐。")
            return

        ranked_rows = []
        for row in self.music_rows:
            row_copy = dict(row)
            row_copy["distance"] = math.dist((valence, arousal), (row["valence"], row["arousal"]))
            ranked_rows.append(row_copy)

        ranked_rows.sort(key=lambda item: item["distance"])
        top_five = ranked_rows[:5]
        top_ids = [song["song_id"] for song in top_five]
        remaining_rows = [dict(row) for row in self.music_rows if row["song_id"] not in top_ids]
        self.recommended_song_ids = top_ids
        self.display_rows = top_five + remaining_rows
        self.apply_music_filter(self.search_input.text())

        top_song = top_five[0]
        self.recommendation_label.setText(
            "推荐前 5 首：\n"
            + "\n".join(
                [
                    f"{index + 1}. {song['song_id']}  距离 {song['distance']:.3f}"
                    for index, song in enumerate(top_five)
                ]
            )
        )
        self.play_track_by_song(top_song, autoplay=True)

    def current_table_rows(self):
        return self.filtered_rows

    def play_from_table_row(self, row, column):
        rows = self.current_table_rows()
        if 0 <= row < len(rows):
            self.play_track_by_song(rows[row], autoplay=True)

    def play_track_by_song(self, song, autoplay=True):
        file_path = song["file_path"]
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "文件不存在", f"找不到音频文件：\n{file_path}")
            return

        self.current_track_index = next((idx for idx, item in enumerate(self.display_rows) if item["song_id"] == song["song_id"]), -1)
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        if autoplay:
            self.player.play()
            self.now_playing_label.setText(f"正在播放：{song['song_id']}")
        else:
            self.now_playing_label.setText(f"已加载：{song['song_id']}")

        rows = self.current_table_rows()
        selected_index = next((idx for idx, item in enumerate(rows) if item["song_id"] == song["song_id"]), -1)
        if selected_index >= 0:
            self.music_table.selectRow(selected_index)

    def play_previous(self):
        if not self.display_rows:
            return
        next_index = len(self.display_rows) - 1 if self.current_track_index <= 0 else self.current_track_index - 1
        self.play_track_by_song(self.display_rows[next_index], autoplay=True)

    def play_next(self):
        if not self.display_rows:
            return
        next_index = 0 if self.current_track_index < 0 else (self.current_track_index + 1) % len(self.display_rows)
        self.play_track_by_song(self.display_rows[next_index], autoplay=True)

    def toggle_playback(self):
        if self.player.mediaStatus() == QMediaPlayer.NoMedia and self.display_rows:
            self.play_track_by_song(self.display_rows[0], autoplay=True)
            return
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
            if self.now_playing_label.text().startswith("正在播放"):
                self.now_playing_label.setText(self.now_playing_label.text().replace("正在播放", "已暂停", 1))
        else:
            self.player.play()
            if 0 <= self.current_track_index < len(self.display_rows):
                self.now_playing_label.setText(f"正在播放：{self.display_rows[self.current_track_index]['song_id']}")

    def update_position(self, position):
        if not self.is_slider_dragging:
            self.progress_slider.setValue(position)
        self.current_time_label.setText(self.format_milliseconds(position))

    def update_duration(self, duration):
        self.progress_slider.setRange(0, max(duration, 0))
        self.total_time_label.setText(self.format_milliseconds(duration))

    def on_slider_pressed(self):
        self.is_slider_dragging = True

    def on_slider_released(self):
        self.is_slider_dragging = False
        self.player.setPosition(self.progress_slider.value())

    def preview_seek_position(self, position):
        self.current_time_label.setText(self.format_milliseconds(position))
        if self.is_slider_dragging:
            return
        self.player.setPosition(position)

    def handle_media_status(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.play_next()

    def format_milliseconds(self, value):
        total_seconds = max(0, int(value / 1000))
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.player.stop()
        self.inference_thread.quit()
        self.inference_thread.wait(2000)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MusicEmotionApp()
    window.show()
    sys.exit(app.exec_())
