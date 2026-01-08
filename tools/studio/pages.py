# Project MUSE - pages.py
# Studio Pages (5-Step Wizard with Auto-Processing)
# (C) 2025 MUSE Corp.

import os
import cv2
import glob
import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QDialog, QMessageBox, QComboBox, QSizePolicy, QProgressBar, QTextEdit,
    QListWidget, QListWidgetItem, QAbstractItemView, QCheckBox, QRadioButton, QButtonGroup,
    QGridLayout
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QMutex, QMutexLocker, QSize
from PySide6.QtGui import QPixmap, QImage, QIcon

from studio.widgets import NewProfileDialog, ProfileActionDialog
from studio.workers import CameraLoader, PipelineWorker
from studio.gl_widget import CameraGLWidget

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False


# ==============================================================================
# BASE CLASS for all step pages
# ==============================================================================
class StudioPageBase(QWidget):
    """Base class for all studio step pages"""
    step_completed = Signal()
    request_settings = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_active = False
        self._is_completed = False

    def activate(self):
        """Called when page becomes visible"""
        self._is_active = True

    def deactivate(self):
        """Called when leaving this page"""
        self._is_active = False

    def is_completed(self) -> bool:
        """Check if step completion conditions are met"""
        return self._is_completed

    def mark_completed(self):
        """Mark step as completed and emit signal"""
        self._is_completed = True
        self.step_completed.emit()


# ==============================================================================
# [STEP 1] Profile Selection
# ==============================================================================
class Step1_ProfileSelect(StudioPageBase):
    """Profile selection page"""
    profile_selected = Signal(str)  # profile_name

    def __init__(self, personal_data_dir, parent=None):
        super().__init__(parent)
        self.personal_data_dir = personal_data_dir
        self.selected_profile = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        # Header
        header = QVBoxLayout()
        title = QLabel("프로필 선택")
        title.setObjectName("StepTitle")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: white;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("학습할 프로필을 선택하거나 새로 만드세요")
        subtitle.setObjectName("StepDescription")
        subtitle.setStyleSheet("font-size: 14px; color: #949ba4;")
        subtitle.setAlignment(Qt.AlignCenter)

        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Scroll area for profiles
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_content)
        self.scroll_layout.setSpacing(15)
        self.scroll_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll, stretch=1)

        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 13px; color: #00D4DB; font-weight: 600;")
        layout.addWidget(self.lbl_status)

    def activate(self):
        super().activate()
        self.refresh_profiles()

    def refresh_profiles(self):
        print(f"[Profile] refresh_profiles called, personal_data_dir={self.personal_data_dir}")
        # Clear existing
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        row, col = 0, 0
        max_cols = 3

        # New profile button
        btn_new = self._create_profile_card("+", "새 프로필 만들기", is_new=True)
        btn_new.clicked.connect(self._on_new_profile)
        self.scroll_layout.addWidget(btn_new, row, col)
        col += 1

        # Existing profiles
        if os.path.exists(self.personal_data_dir):
            profiles = sorted([d for d in os.listdir(self.personal_data_dir)
                               if os.path.isdir(os.path.join(self.personal_data_dir, d))])

            # Priority sort
            priority = ['front', 'top', 'under']
            sorted_profiles = []
            for p in priority:
                if p in profiles:
                    sorted_profiles.append(p)
                    profiles.remove(p)
            sorted_profiles.extend(profiles)

            for p_name in sorted_profiles:
                profile_path = os.path.join(self.personal_data_dir, p_name)

                # Check status
                video_pattern = os.path.join(profile_path, "train_video_*.mp4")
                video_files = glob.glob(video_pattern)
                has_data = len(video_files) > 0
                bg_path = os.path.join(profile_path, "background.jpg")
                has_bg = os.path.exists(bg_path)

                print(f"[Profile] {p_name}: path={profile_path}")
                print(f"[Profile] {p_name}: video_pattern={video_pattern}")
                print(f"[Profile] {p_name}: video_files found={len(video_files)}")
                print(f"[Profile] {p_name}: has_data={has_data}, has_bg={has_bg}")

                status = "준비됨" if (has_data and has_bg) else "설정 필요"

                card = self._create_profile_card(
                    p_name.upper()[0],
                    p_name.upper(),
                    status=status,
                    selected=(p_name == self.selected_profile)
                )
                card.clicked.connect(lambda checked=False, name=p_name: self._on_profile_selected(name))
                self.scroll_layout.addWidget(card, row, col)

                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

    def _create_profile_card(self, icon_text, name, status="", is_new=False, selected=False):
        """Create a profile card widget"""
        card = QPushButton()
        card.setFixedSize(200, 150)
        card.setCursor(Qt.PointingHandCursor)

        if is_new:
            card.setStyleSheet("""
                QPushButton {
                    background-color: rgba(88, 101, 242, 0.1);
                    border: 2px dashed #5865f2;
                    border-radius: 12px;
                    color: #5865f2;
                    font-size: 14px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(88, 101, 242, 0.2);
                    border-color: #7289da;
                }
            """)
            card.setText(f"{icon_text}\n\n{name}")
        else:
            border_color = "#00D4DB" if selected else "rgba(255, 255, 255, 0.1)"
            bg_color = "rgba(0, 212, 219, 0.1)" if selected else "#2b2d31"

            card.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_color};
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    color: white;
                    font-size: 14px;
                    font-weight: 600;
                    text-align: center;
                }}
                QPushButton:hover {{
                    background-color: #383a40;
                    border-color: #00D4DB;
                }}
            """)

            # HTML 태그 제거 - 순수 텍스트만 사용
            if status:
                card.setText(f"{icon_text}\n\n{name}\n{status}")
            else:
                card.setText(f"{icon_text}\n\n{name}")

        return card

    def _on_new_profile(self):
        dlg = NewProfileDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name = dlg.get_name()
            if name:
                # Create directory
                profile_path = os.path.join(self.personal_data_dir, name)
                os.makedirs(profile_path, exist_ok=True)

                self.selected_profile = name
                self.refresh_profiles()
                self._update_status()

    def _on_profile_selected(self, name):
        self.selected_profile = name
        self.refresh_profiles()
        self._update_status()

    def _update_status(self):
        if self.selected_profile:
            self.lbl_status.setText(f"선택됨: {self.selected_profile.upper()}")
            self._is_completed = True
            self.profile_selected.emit(self.selected_profile)
            self.step_completed.emit()
        else:
            self.lbl_status.setText("")
            self._is_completed = False

    def get_selected_profile(self) -> str:
        return self.selected_profile


# ==============================================================================
# [STEP 2] Camera Connection
# ==============================================================================
class Step2_CameraConnect(StudioPageBase):
    """Camera connection page with auto-connect"""
    camera_ready = Signal(int)  # camera_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.connected_camera = None
        self.loader_thread = None
        self.gl_widget = None
        self.preview_timer = None
        self.test_cap = None
        self._gl_ready = False  # GL 초기화 완료 여부
        self._pending_cam_idx = None  # GL 초기화 대기 중인 카메라 인덱스
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        # Header
        header = QVBoxLayout()
        title = QLabel("카메라 연결")
        title.setObjectName("StepTitle")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: white;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("학습에 사용할 카메라를 연결합니다")
        subtitle.setObjectName("StepDescription")
        subtitle.setStyleSheet("font-size: 14px; color: #949ba4;")
        subtitle.setAlignment(Qt.AlignCenter)

        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Preview area
        preview_container = QFrame()
        preview_container.setStyleSheet("""
            QFrame {
                background-color: #0D0D0D;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }
        """)
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_widget = CameraGLWidget()
        self.gl_widget.gl_ready.connect(self._on_gl_ready)
        self.gl_widget.setMinimumHeight(360)
        preview_layout.addWidget(self.gl_widget)

        layout.addWidget(preview_container, stretch=1)

        # Status and controls
        control_area = QHBoxLayout()

        self.lbl_status = QLabel("카메라를 자동으로 연결합니다...")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #949ba4;")

        control_area.addWidget(self.lbl_status)
        control_area.addStretch()

        # Camera selector (shown if auto-connect fails)
        self.combo_cam = QComboBox()
        self.combo_cam.setMinimumWidth(250)
        self.combo_cam.setVisible(False)
        control_area.addWidget(self.combo_cam)

        self.btn_connect = QPushButton("연결")
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background-color: #5865f2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
        """)
        self.btn_connect.setVisible(False)
        self.btn_connect.clicked.connect(self._manual_connect)
        control_area.addWidget(self.btn_connect)

        layout.addLayout(control_area)

    def activate(self):
        super().activate()
        self._auto_connect()

    def deactivate(self):
        super().deactivate()
        self._stop_preview()

    def _auto_connect(self):
        """Try to auto-connect to default camera"""
        self.lbl_status.setText("카메라 연결 중...")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #949ba4;")

        # Try camera 0 first
        self.loader_thread = CameraLoader(0)
        self.loader_thread.finished.connect(self._on_auto_connected)
        self.loader_thread.error.connect(self._on_auto_failed)
        self.loader_thread.start()

    def _on_auto_connected(self, cap, idx):
        """Auto connection successful"""
        cap.release()
        self.connected_camera = idx

        self.lbl_status.setText(f"카메라 연결됨 (장치 {idx})")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #00D4DB; font-weight: 600;")

        # Start preview
        self._start_preview(idx)

        # Mark completed
        self._is_completed = True
        self.camera_ready.emit(idx)
        self.step_completed.emit()

    def _on_auto_failed(self, msg):
        """Auto connection failed, show manual selection"""
        self.lbl_status.setText("자동 연결 실패. 카메라를 선택하세요.")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #f0b232;")

        # Show manual controls
        self._refresh_camera_list()
        self.combo_cam.setVisible(True)
        self.btn_connect.setVisible(True)

    def _refresh_camera_list(self):
        self.combo_cam.clear()
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    self.combo_cam.addItem(f"[{i}] {name}", i)
            except:
                self._add_fallback_cameras()
        else:
            self._add_fallback_cameras()

    def _add_fallback_cameras(self):
        for i in range(5):
            self.combo_cam.addItem(f"카메라 장치 {i}", i)

    def _manual_connect(self):
        idx = self.combo_cam.currentData()
        if idx is None:
            return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("연결 중...")

        self.loader_thread = CameraLoader(idx)
        self.loader_thread.finished.connect(self._on_manual_connected)
        self.loader_thread.error.connect(self._on_manual_failed)
        self.loader_thread.start()

    def _on_manual_connected(self, cap, idx):
        cap.release()
        self.connected_camera = idx

        self.lbl_status.setText(f"카메라 연결됨 (장치 {idx})")
        self.lbl_status.setStyleSheet("font-size: 13px; color: #00D4DB; font-weight: 600;")

        self.combo_cam.setVisible(False)
        self.btn_connect.setVisible(False)

        self._start_preview(idx)

        self._is_completed = True
        self.camera_ready.emit(idx)
        self.step_completed.emit()

    def _on_manual_failed(self, msg):
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("연결")
        QMessageBox.warning(self, "연결 오류", msg)

    def _on_gl_ready(self):
        """OpenGL 초기화 완료 알림"""
        print("[Studio] GL Widget Ready - OpenGL context initialized")
        self._gl_ready = True

        # 대기 중인 카메라가 있으면 타이머 시작
        if self._pending_cam_idx is not None:
            print(f"[Studio] Starting preview timer for pending camera {self._pending_cam_idx}")
            self._start_preview_timer()
            self._pending_cam_idx = None

    def _start_preview(self, cam_idx):
        """Start camera preview"""
        print(f"[Studio] _start_preview({cam_idx}) called")

        # 위젯 가시성 확보
        if self.gl_widget and not self.gl_widget.isVisible():
            self.gl_widget.show()

        self.test_cap = cv2.VideoCapture(cam_idx)
        if self.test_cap.isOpened():
            self.test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"[Studio] Camera {cam_idx} opened successfully")

            # QLabel 기반이므로 바로 타이머 시작
            self._start_preview_timer()
        else:
            print(f"[Studio] Failed to open camera {cam_idx}")

    def _start_preview_timer(self):
        """프리뷰 타이머 시작"""
        if self.test_cap and self.test_cap.isOpened():
            print("[Studio] Starting preview timer (33ms interval)")
            self.preview_timer = QTimer()
            self.preview_timer.timeout.connect(self._update_preview)
            self.preview_timer.start(33)  # ~30fps

    def _update_preview(self):
        if self.test_cap and self.test_cap.isOpened():
            ret, frame = self.test_cap.read()
            if ret and self.gl_widget:
                self.gl_widget.render(frame)

    def _stop_preview(self):
        if self.preview_timer:
            self.preview_timer.stop()
            self.preview_timer = None

        if self.test_cap:
            self.test_cap.release()
            self.test_cap = None

        if self.gl_widget:
            self.gl_widget.cleanup()

    def get_camera_index(self) -> int:
        return self.connected_camera if self.connected_camera is not None else 0


# ==============================================================================
# RecorderWorker (shared between Step3 data recording)
# ==============================================================================
class RecorderWorker(QThread):
    time_updated = Signal(float)
    bg_status_updated = Signal(bool)

    def __init__(self, cam_index, profile_dir):
        super().__init__()
        self.cam_index = cam_index
        self.profile_dir = profile_dir
        self.running = True

        self.cap = None
        self.m_lock = QMutex()
        self.m_frame = None
        self.m_frame_id = 0

        self.is_recording = False

        self.cmd_start_rec = False
        self.cmd_stop_rec = False
        self.req_bg_capture = False

        self.video_writer = None
        self.current_start_time = 0
        self.accumulated_time = 0.0
        self.last_reported_int_time = -1

        self.split_counter = 0
        self.last_split_time = 0.0
        self.MAX_SPLIT = 60.0

    def _calc_existing_duration(self, folder):
        total = 0.0
        files = glob.glob(os.path.join(folder, "train_video_*.mp4"))
        for f in files:
            try:
                cap = cv2.VideoCapture(f)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        total += (frames / fps)
                cap.release()
            except:
                pass
        return total

    def start_recording(self):
        self.cmd_start_rec = True

    def stop_recording(self):
        self.cmd_stop_rec = True

    def trigger_bg_capture(self):
        self.req_bg_capture = True

    def _start_segment(self, w, h, fps):
        if self.video_writer:
            self.video_writer.release()

        timestamp = int(time.time())
        path = os.path.join(self.profile_dir, f"train_video_{timestamp}_{self.split_counter:02d}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        self.last_split_time = time.time()
        print(f"[REC] New Segment Started: {os.path.basename(path)}")

    def run(self):
        self.accumulated_time = self._calc_existing_duration(self.profile_dir)

        # 기존 녹화 시간이 있으면 UI에 즉시 표시
        if self.accumulated_time > 0:
            print(f"[CAM] [Worker] Existing recordings found: {self.accumulated_time:.1f} seconds")
            self.time_updated.emit(self.accumulated_time)
            self.last_reported_int_time = int(self.accumulated_time)

        print(f"[CAM] [Worker] Opening Camera {self.cam_index} Native...")
        self.cap = cv2.VideoCapture(self.cam_index)

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("[CAM] [Worker] Capture Loop Started (Thread-Local).")
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                self.msleep(5)
                continue

            with QMutexLocker(self.m_lock):
                self.m_frame = frame
                self.m_frame_id += 1

            # Background capture
            if self.req_bg_capture:
                bg_path = os.path.join(self.profile_dir, "background.jpg")
                os.makedirs(self.profile_dir, exist_ok=True)
                cv2.imwrite(bg_path, frame)
                self.bg_status_updated.emit(True)
                self.req_bg_capture = False

            # Recording logic
            if self.cmd_start_rec:
                self.cmd_start_rec = False
                if not self.is_recording:
                    self.is_recording = True
                    self.current_start_time = time.time()
                    self.split_counter = 0

                    h, w = frame.shape[:2]
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0

                    os.makedirs(self.profile_dir, exist_ok=True)
                    self._start_segment(w, h, fps)

            if self.cmd_stop_rec:
                self.cmd_stop_rec = False
                if self.is_recording:
                    self.is_recording = False
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.accumulated_time += (time.time() - self.current_start_time)
                    print("[CAM] [Worker] Recording stopped.")

            if self.is_recording:
                # Auto split
                if time.time() - self.last_split_time >= self.MAX_SPLIT:
                    self.split_counter += 1
                    h, w = frame.shape[:2]
                    fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self._start_segment(w, h, fps)

                elapsed = time.time() - self.current_start_time
                total_time = self.accumulated_time + elapsed

                if self.video_writer and self.video_writer.isOpened():
                    try:
                        self.video_writer.write(frame)
                    except Exception as e:
                        print(f"[ERROR] [Worker] Write Error: {e}")

                if int(total_time) > self.last_reported_int_time:
                    self.time_updated.emit(total_time)
                    self.last_reported_int_time = int(total_time)

        if self.video_writer:
            self.video_writer.release()
        if self.cap:
            self.cap.release()
        print("[CAM] [Worker] Capture Loop Ended.")

    def stop(self):
        self.running = False
        self.wait()


# ==============================================================================
# [STEP 3] Data Recording
# ==============================================================================
class Step3_DataRecording(StudioPageBase):
    """Data recording page"""

    def __init__(self, output_dir, parent=None):
        super().__init__(parent)
        self.output_dir = output_dir
        self.recorder_thread = None
        self.current_profile_dir = ""
        self.current_profile_name = ""
        self.cam_index = 0
        self.render_timer = QTimer(self)
        self.last_rendered_id = -1
        self.has_background = False
        self.min_record_seconds = 60  # 최소 1분
        self._gl_ready = False  # GL 초기화 완료 여부
        self._pending_start_render = False  # 렌더 타이머 시작 대기 중

        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left: Camera preview
        self.gl_widget = CameraGLWidget()
        self.gl_widget.gl_ready.connect(self._on_gl_ready)
        self.gl_widget.setMinimumSize(320, 240)
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.gl_widget, stretch=3)

        # Right: Sidebar
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #0D0D0D; border-left: 1px solid rgba(255, 255, 255, 0.04);")
        sidebar.setFixedWidth(400)

        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(30, 40, 30, 40)
        sb_layout.setSpacing(20)

        # Title
        lbl_title = QLabel("데이터 녹화")
        lbl_title.setStyleSheet("font-size: 24px; font-weight: 700; color: white; border: none;")
        sb_layout.addWidget(lbl_title)

        lbl_desc = QLabel("배경을 촬영하고 학습용 영상을 녹화하세요")
        lbl_desc.setStyleSheet("font-size: 13px; color: #949ba4; border: none;")
        sb_layout.addWidget(lbl_desc)

        # Status card
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.03);
                border-radius: 14px;
                padding: 18px;
                border: 1px solid rgba(255, 255, 255, 0.04);
            }
        """)
        sc_layout = QVBoxLayout(status_card)

        self.lbl_bg_status = QLabel("배경 촬영 필요")
        self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px; border: none;")
        sc_layout.addWidget(self.lbl_bg_status)

        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet("color: white; font-size: 32px; font-family: monospace; font-weight: bold; border: none;")
        self.lbl_time.setAlignment(Qt.AlignRight)
        sc_layout.addWidget(self.lbl_time)

        self.lbl_min_hint = QLabel(f"권장: 최소 {self.min_record_seconds}초 녹화")
        self.lbl_min_hint.setStyleSheet("color: #949ba4; font-size: 11px; border: none;")
        sc_layout.addWidget(self.lbl_min_hint)

        sb_layout.addWidget(status_card)

        # Background capture button
        self.btn_bg = QPushButton("배경 촬영하기 (단축키 B)")
        self.btn_bg.setStyleSheet("""
            QPushButton {
                background-color: #2b2d31;
                border: 1px solid #3f4147;
                color: #dbdee1;
                font-size: 14px;
                font-weight: 500;
                padding: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #383a40;
                border-color: #5865f2;
            }
        """)
        self.btn_bg.setCursor(Qt.PointingHandCursor)
        self.btn_bg.clicked.connect(self.capture_background)
        sb_layout.addWidget(self.btn_bg)

        # Record button
        self.btn_record = QPushButton("녹화 시작")
        self.btn_record.setStyleSheet("""
            QPushButton {
                background-color: #2b2d31;
                border: 1px solid #3f4147;
                color: #dbdee1;
                font-size: 14px;
                font-weight: 600;
                padding: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #383a40;
            }
        """)
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.clicked.connect(self.toggle_record)
        sb_layout.addWidget(self.btn_record)

        sb_layout.addStretch()

        layout.addWidget(sidebar)

    def setup_session(self, cam_index: int, profile_name: str, profile_dir: str):
        """Setup recording session"""
        self.cam_index = cam_index
        self.current_profile_name = profile_name
        self.current_profile_dir = profile_dir
        self.last_rendered_id = -1

        print(f"[Studio] Step3 setup_session: profile_dir={profile_dir}")

        # Check if background exists
        bg_path = os.path.join(profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            print(f"[Studio] Step3 Existing background found: {bg_path}")
            self._on_bg_captured(True)
        else:
            print(f"[Studio] Step3 No background found at: {bg_path}")
            self.has_background = False
            self.lbl_bg_status.setText("배경 촬영 필요")
            self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px; border: none;")
            self.btn_record.setEnabled(False)

    def _on_gl_ready(self):
        """OpenGL 초기화 완료 알림"""
        print("[Studio] Step3 GL Widget Ready - OpenGL context initialized")
        self._gl_ready = True

        # 대기 중인 렌더 타이머가 있으면 시작
        if self._pending_start_render:
            print("[Studio] Step3 Starting render timer after GL ready")
            self._start_render_timer()
            self._pending_start_render = False

    def activate(self):
        super().activate()
        print(f"[Studio] Step3 activate() called, profile_dir={self.current_profile_dir}")

        # 위젯 가시성 확보
        if self.gl_widget and not self.gl_widget.isVisible():
            self.gl_widget.show()

        if self.current_profile_dir:
            # Start recorder thread
            self.recorder_thread = RecorderWorker(self.cam_index, self.current_profile_dir)
            self.recorder_thread.time_updated.connect(self._update_time_label)
            self.recorder_thread.bg_status_updated.connect(self._on_bg_captured)
            self.recorder_thread.start()

            # QLabel 기반이므로 바로 렌더 타이머 시작
            self._start_render_timer()
        else:
            print("[Studio] Step3 WARNING: current_profile_dir is empty!")

    def _start_render_timer(self):
        """렌더 타이머 시작"""
        if self.recorder_thread:
            print("[Studio] Step3 Starting render timer (16ms interval)")
            # 중복 연결 방지
            try:
                self.render_timer.timeout.disconnect()
            except:
                pass
            self.render_timer.timeout.connect(self._update_view)
            self.render_timer.start(16)

    def deactivate(self):
        super().deactivate()
        self._stop_recording()

    def _update_view(self):
        if not self.recorder_thread:
            return

        frame = None
        curr_id = -1

        with QMutexLocker(self.recorder_thread.m_lock):
            if self.recorder_thread.m_frame is not None:
                frame = self.recorder_thread.m_frame.copy()  # 복사본 사용
                curr_id = self.recorder_thread.m_frame_id

        if frame is not None and curr_id > self.last_rendered_id:
            self.gl_widget.render(frame)
            self.last_rendered_id = curr_id
            # 디버그: 첫 프레임만 로그
            if curr_id == 1:
                print(f"[Studio] Step3 First frame rendered, shape={frame.shape}")

    def capture_background(self):
        if self.recorder_thread:
            self.recorder_thread.trigger_bg_capture()

    def _on_bg_captured(self, success):
        if success:
            self.has_background = True
            self.lbl_bg_status.setText("배경 준비 완료")
            self.lbl_bg_status.setStyleSheet("color: #00D4DB; font-weight: bold; font-size: 14px; border: none;")
            self.btn_record.setEnabled(True)

    def toggle_record(self):
        if not self.recorder_thread:
            return

        if self.btn_record.isChecked():
            self.recorder_thread.start_recording()
            self.btn_record.setText("녹화 중지")
            self.btn_record.setStyleSheet("""
                QPushButton {
                    background-color: #FF5252;
                    color: white;
                    border-radius: 12px;
                    font-weight: bold;
                    font-size: 16px;
                    border: none;
                    padding: 16px;
                }
            """)
        else:
            self.recorder_thread.stop_recording()
            self.btn_record.setText("녹화 시작")
            self.btn_record.setStyleSheet("""
                QPushButton {
                    background-color: #2b2d31;
                    border: 1px solid #3f4147;
                    color: #dbdee1;
                    font-size: 14px;
                    font-weight: 600;
                    padding: 16px;
                    border-radius: 8px;
                }
                QPushButton:hover {
                    background-color: #383a40;
                }
            """)

    def _update_time_label(self, total_seconds):
        total = int(total_seconds)
        self.lbl_time.setText(f"{total // 60:02d}:{total % 60:02d}")

        # Check completion
        if total >= self.min_record_seconds and self.has_background:
            if not self._is_completed:
                self._is_completed = True
                self.step_completed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_B:
            self.capture_background()
        else:
            super().keyPressEvent(event)

    def _stop_recording(self):
        if self.render_timer.isActive():
            self.render_timer.stop()

        if self.recorder_thread:
            self.recorder_thread.stop()
            self.recorder_thread = None

        if self.gl_widget:
            self.gl_widget.cleanup()


# ==============================================================================
# [STEP 4] Preview Selection (NEW)
# ==============================================================================
class Step4_PreviewSelect(StudioPageBase):
    """Preview selection page - shows analysis previews and allows user to exclude bad ones"""
    previews_confirmed = Signal(list)  # List of selected video names

    def __init__(self, root_dir, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.selected_videos = []
        self.checkboxes = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)

        # Header
        header = QVBoxLayout()
        title = QLabel("프리뷰 확인")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: white;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("분석 결과를 확인하고 품질이 좋지 않은 영상은 체크 해제하세요")
        subtitle.setStyleSheet("font-size: 14px; color: #949ba4;")
        subtitle.setAlignment(Qt.AlignCenter)

        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Status label
        self.lbl_status = QLabel("프리뷰 로딩 중...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00D4DB; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # Scroll area for preview grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                background-color: #2b2d31;
            }
            QScrollBar:vertical {
                background: #2b2d31;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #4e5058;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(15, 15, 15, 15)
        scroll.setWidget(self.grid_container)
        layout.addWidget(scroll, stretch=1)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        self.btn_select_all = QPushButton("모두 선택")
        self.btn_select_all.setStyleSheet("""
            QPushButton {
                background-color: #4e5058;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #5c5f66; }
        """)
        self.btn_select_all.clicked.connect(self._select_all)
        btn_layout.addWidget(self.btn_select_all)

        self.btn_deselect_all = QPushButton("모두 해제")
        self.btn_deselect_all.setStyleSheet("""
            QPushButton {
                background-color: #4e5058;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #5c5f66; }
        """)
        self.btn_deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(self.btn_deselect_all)

        btn_layout.addStretch()

        self.lbl_selected_count = QLabel("선택: 0/0")
        self.lbl_selected_count.setStyleSheet("color: #949ba4; font-size: 13px;")
        btn_layout.addWidget(self.lbl_selected_count)

        layout.addLayout(btn_layout)

        # Progress bar for preview generation (hidden by default)
        self.pbar = QProgressBar()
        self.pbar.setFormat("%p%")
        self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #4e5058;
                border-radius: 4px;
                height: 20px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #5865f2;
                border-radius: 4px;
            }
        """)
        self.pbar.hide()
        layout.addWidget(self.pbar)

    def activate(self):
        super().activate()
        # 분석 단계에서 이미 프리뷰가 생성되어 있으므로 바로 로드
        QTimer.singleShot(100, self._load_previews)

    def _load_previews(self):
        """Load preview images from all profiles"""
        # Clear existing
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.checkboxes.clear()

        # Find all preview images
        data_root = os.path.join(self.root_dir, "recorded_data", "personal_data")
        if not os.path.exists(data_root):
            self.lbl_status.setText("데이터 폴더가 없습니다")
            return

        profiles = [d for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))]

        all_previews = []
        for profile in profiles:
            preview_dir = os.path.join(data_root, profile, "previews")
            if os.path.exists(preview_dir):
                for img_file in sorted(os.listdir(preview_dir)):
                    if img_file.endswith(".jpg"):
                        all_previews.append({
                            "profile": profile,
                            "path": os.path.join(preview_dir, img_file),
                            "video_name": img_file.replace(".jpg", "")
                        })

        if not all_previews:
            self.lbl_status.setText("프리뷰 이미지가 없습니다. 먼저 녹화를 진행하세요.")
            return

        self.lbl_status.setText(f"총 {len(all_previews)}개의 영상 프리뷰")

        # Add to grid (4 columns)
        cols = 4
        for idx, preview_info in enumerate(all_previews):
            row = idx // cols
            col = idx % cols

            item_widget = self._create_preview_item(preview_info)
            self.grid_layout.addWidget(item_widget, row, col)

        self._update_selected_count()
        self._is_completed = True
        self.step_completed.emit()

    def _create_preview_item(self, preview_info):
        """Create a preview item widget with checkbox"""
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #1e1f22;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.06);
            }
            QFrame:hover {
                border: 1px solid rgba(0, 212, 219, 0.5);
            }
        """)

        item_layout = QVBoxLayout(container)
        item_layout.setContentsMargins(8, 8, 8, 8)
        item_layout.setSpacing(8)

        # Preview image
        img_label = QLabel()
        img_label.setFixedSize(200, 112)  # 16:9 aspect ratio
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("border-radius: 4px;")

        # Load and scale image
        pixmap = QPixmap(preview_info["path"])
        if not pixmap.isNull():
            scaled = pixmap.scaled(200, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled)
        else:
            img_label.setText("이미지 로드 실패")
            img_label.setStyleSheet("color: #949ba4; font-size: 11px;")

        item_layout.addWidget(img_label)

        # Checkbox with video name
        checkbox = QCheckBox(preview_info["video_name"][:20] + "...")
        checkbox.setChecked(True)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #dbdee1;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #4e5058;
                background: #2b2d31;
            }
            QCheckBox::indicator:checked {
                background: #5865f2;
                border: 1px solid #5865f2;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #5865f2;
            }
        """)
        checkbox.setToolTip(preview_info["video_name"])
        checkbox.stateChanged.connect(self._update_selected_count)

        video_key = f"{preview_info['profile']}|{preview_info['video_name']}"
        self.checkboxes[video_key] = checkbox

        item_layout.addWidget(checkbox)

        return container

    def _select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def _update_selected_count(self):
        total = len(self.checkboxes)
        selected = sum(1 for cb in self.checkboxes.values() if cb.isChecked())
        self.lbl_selected_count.setText(f"선택: {selected}/{total}")

    def get_selected_videos(self):
        """Return list of selected video names (profile|video_name format)"""
        return [key for key, cb in self.checkboxes.items() if cb.isChecked()]

    def save_selection(self):
        """Save selection to a JSON file for the analysis step"""
        selected = self.get_selected_videos()
        selection_file = os.path.join(self.root_dir, "recorded_data", "personal_data", "selected_videos.json")

        import json
        with open(selection_file, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)

        self.previews_confirmed.emit(selected)
        return selected


# ==============================================================================
# [STEP 5] AI Analysis (Renamed from Step 4)
# ==============================================================================
class Step5_AiAnalysis(StudioPageBase):
    """AI analysis page (auto-processing)"""

    def __init__(self, root_dir, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        # Header
        header = QVBoxLayout()
        title = QLabel("AI 분석")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: white;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("녹화된 영상을 AI가 자동으로 분석합니다")
        subtitle.setStyleSheet("font-size: 14px; color: #949ba4;")
        subtitle.setAlignment(Qt.AlignCenter)

        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Progress bar
        self.pbar = QProgressBar()
        self.pbar.setFormat("%p%")
        self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #4e5058;
                border-radius: 4px;
                height: 20px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #5865f2;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.pbar)

        # Status
        self.lbl_status = QLabel("준비됨")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #AAA; font-size: 14px;")
        layout.addWidget(self.lbl_status)

        self.lbl_time_info = QLabel("총 소요: 00:00 | 현재 단계: 00:00")
        self.lbl_time_info.setAlignment(Qt.AlignCenter)
        self.lbl_time_info.setStyleSheet("color: #00D4DB; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_time_info)

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: #050505;
                color: #00FF88;
                font-family: Consolas, D2Coding, monospace;
                font-size: 12px;
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 12px;
                padding: 12px;
            }
        """)
        layout.addWidget(self.log_view, stretch=1)

        # Start button
        self.btn_start = QPushButton("분석 시작")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #5865f2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #4752c4;
            }
            QPushButton:disabled {
                background-color: #4e5058;
                color: #6d6f78;
            }
        """)
        self.btn_start.clicked.connect(self.start_analysis)
        layout.addWidget(self.btn_start)

    def activate(self):
        super().activate()
        # Auto-start analysis
        QTimer.singleShot(500, self.start_analysis)

    def deactivate(self):
        super().deactivate()
        if self.worker and self.worker.isRunning():
            self.worker.request_early_stop()

    def start_analysis(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("분석 중...")
        self.log_view.clear()

        self.worker = PipelineWorker(self.root_dir, mode="analyze")
        self.worker.log_signal.connect(self.log_view.append)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, percent, status_text, time_info):
        self.pbar.setValue(percent)
        self.lbl_status.setText(status_text)
        self.lbl_time_info.setText(time_info)

    def _on_finished(self):
        self.btn_start.setText("분석 완료")
        self.lbl_status.setText("분석이 완료되었습니다.")
        self._is_completed = True
        self.step_completed.emit()


# ==============================================================================
# [STEP 6] Model Training (Renamed from Step 5)
# ==============================================================================
class Step6_ModelTraining(StudioPageBase):
    """Model training page"""
    training_finished = Signal()

    def __init__(self, root_dir, parent=None):
        super().__init__(parent)
        self.root_dir = root_dir
        self.worker = None
        self.selected_track = "STUDENT"
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 40, 60, 40)
        layout.setSpacing(20)

        # Header
        header = QVBoxLayout()
        title = QLabel("모델 학습")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: white;")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("분석된 데이터로 개인화 모델을 학습합니다")
        subtitle.setStyleSheet("font-size: 14px; color: #949ba4;")
        subtitle.setAlignment(Qt.AlignCenter)

        header.addWidget(title)
        header.addWidget(subtitle)
        layout.addLayout(header)

        # Progress bar
        self.pbar = QProgressBar()
        self.pbar.setFormat("%p%")
        self.pbar.setAlignment(Qt.AlignCenter)
        self.pbar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #4e5058;
                border-radius: 4px;
                height: 20px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.pbar)

        # Status
        self.lbl_status = QLabel("학습 트랙을 선택하세요")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #AAA; font-size: 14px;")
        layout.addWidget(self.lbl_status)

        self.lbl_time_info = QLabel("")
        self.lbl_time_info.setAlignment(Qt.AlignCenter)
        self.lbl_time_info.setStyleSheet("color: #00D4DB; font-size: 13px; font-weight: bold;")
        layout.addWidget(self.lbl_time_info)

        # Track selection
        track_widget = QWidget()
        track_layout = QHBoxLayout(track_widget)
        track_layout.setAlignment(Qt.AlignCenter)
        track_layout.setSpacing(30)

        self.rb_student = QRadioButton("전체 최적화 (Student)")
        self.rb_student.setChecked(True)
        self.rb_student.setStyleSheet("font-weight: bold; color: white; font-size: 14px;")

        self.rb_lora = QRadioButton("정밀 보정 (LoRA)")
        self.rb_lora.setStyleSheet("font-weight: bold; color: #FF9800; font-size: 14px;")

        self.track_group = QButtonGroup(self)
        self.track_group.addButton(self.rb_student, 0)
        self.track_group.addButton(self.rb_lora, 1)
        self.track_group.buttonClicked.connect(self._on_track_changed)

        track_layout.addWidget(self.rb_student)
        track_layout.addWidget(self.rb_lora)

        layout.addWidget(track_widget)

        # Log view
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: #050505;
                color: #00FF88;
                font-family: Consolas, D2Coding, monospace;
                font-size: 12px;
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 12px;
                padding: 12px;
            }
        """)
        layout.addWidget(self.log_view, stretch=1)

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_start = QPushButton("학습 시작")
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #43A047;
            }
            QPushButton:disabled {
                background-color: #4e5058;
                color: #6d6f78;
            }
        """)
        self.btn_start.clicked.connect(self.start_training)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("중단")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 30px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
        """)
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.btn_stop)

        layout.addLayout(btn_layout)

    def _on_track_changed(self, btn):
        if btn == self.rb_student:
            self.selected_track = "STUDENT"
        elif btn == self.rb_lora:
            self.selected_track = "LORA"

    def start_training(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setVisible(False)
        self.btn_stop.setVisible(True)
        self.rb_student.setEnabled(False)
        self.rb_lora.setEnabled(False)
        self.log_view.clear()
        self.log_view.append(f"[INFO] Starting training (Track: {self.selected_track})...")

        mode_flag = "train" if self.selected_track == "STUDENT" else "train_lora"

        self.worker = PipelineWorker(self.root_dir, mode=mode_flag)
        self.worker.log_signal.connect(self.log_view.append)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "오류", e))
        self.worker.start()

    def stop_training(self):
        if self.worker:
            self.worker.request_early_stop()
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("중단 요청됨...")

    def _on_progress(self, percent, status_text, time_info):
        self.pbar.setValue(percent)
        self.lbl_status.setText(status_text)
        self.lbl_time_info.setText(time_info)

    def _on_finished(self):
        self.btn_stop.setVisible(False)
        self.btn_start.setVisible(True)
        self.btn_start.setText("학습 완료")
        self.lbl_status.setText("모든 과정이 성공적으로 끝났습니다.")
        self._is_completed = True
        self.step_completed.emit()
        self.training_finished.emit()


# ==============================================================================
# Legacy compatibility exports
# ==============================================================================
# For backward compatibility with existing code
Page1_ProfileSelect = Step1_ProfileSelect
Page2_CameraConnect = Step2_CameraConnect
Page3_DataCollection = Step3_DataRecording
Page4_AiTraining = Step6_ModelTraining
