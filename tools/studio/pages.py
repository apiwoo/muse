# Project MUSE - pages.py
# Wizard Pages for Studio

import os
import cv2
import glob
import time
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QDialog, QMessageBox, QComboBox, QSizePolicy, QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage

# Import other studio modules
from studio.widgets import NewProfileDialog, ProfileActionDialog
from studio.workers import CameraLoader, PipelineWorker

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

# ==============================================================================
# [PAGE 1] Profile Selection
# ==============================================================================
class Page1_ProfileSelect(QWidget):
    profile_confirmed = Signal(str, str) # name, mode ('append' or 'reset')

    def __init__(self, personal_data_dir):
        super().__init__()
        self.personal_data_dir = personal_data_dir
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)

        # Header
        header_layout = QVBoxLayout()
        title = QLabel("Welcome to MUSE Studio")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Select a profile to start training your AI persona.")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Content Area (Scrollable Grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(15)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll)

        self.refresh_profiles()

    def refresh_profiles(self):
        # Clear existing
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # 1. New Profile Button (Primary Action)
        btn_new = QPushButton("+  Create New Profile")
        btn_new.setProperty("class", "primary") # Apply Primary Style
        btn_new.setCursor(Qt.PointingHandCursor)
        btn_new.clicked.connect(self.on_click_new)
        self.scroll_layout.addWidget(btn_new)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 20px 0;")
        self.scroll_layout.addWidget(line)

        # 2. Existing Profiles (Cards)
        if os.path.exists(self.personal_data_dir):
            profiles = sorted([d for d in os.listdir(self.personal_data_dir) 
                               if os.path.isdir(os.path.join(self.personal_data_dir, d))])
            
            # Prioritize default names
            priority = ['front', 'top', 'under']
            sorted_profiles = []
            for p in priority:
                if p in profiles:
                    sorted_profiles.append(p)
                    profiles.remove(p)
            sorted_profiles.extend(profiles)

            if sorted_profiles:
                lbl_exist = QLabel("EXISTING PROFILES")
                lbl_exist.setStyleSheet("color: #666; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
                self.scroll_layout.addWidget(lbl_exist)

            for p_name in sorted_profiles:
                btn = QPushButton(f"📂   {p_name.upper()}")
                btn.setProperty("class", "card") # Apply Card Style
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(lambda checked=False, name=p_name: self.on_click_existing(name))
                self.scroll_layout.addWidget(btn)

    def on_click_new(self):
        dlg = NewProfileDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name = dlg.get_name()
            if name:
                self.profile_confirmed.emit(name, 'reset')

    def on_click_existing(self, name):
        dlg = ProfileActionDialog(name, self)
        result = dlg.exec()
        if result == 1:
            self.profile_confirmed.emit(name, 'append')
        elif result == 2:
            self.profile_confirmed.emit(name, 'reset')

# ==============================================================================
# [PAGE 2] Camera Connection
# ==============================================================================
class Page2_CameraConnect(QWidget):
    camera_ready = Signal(object) # cap object
    go_back = Signal()

    def __init__(self):
        super().__init__()
        self.loader_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 80, 80, 80)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignCenter)

        # Header
        self.lbl_title = QLabel("Connect Camera")
        self.lbl_title.setObjectName("Title")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)
        
        self.lbl_info = QLabel("Target: ???")
        self.lbl_info.setObjectName("Subtitle")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        # Card Container
        card = QFrame()
        card.setStyleSheet("background-color: #252525; border-radius: 15px; border: 1px solid #333;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)

        # Combo & Refresh
        hbox = QHBoxLayout()
        self.combo_cam = QComboBox()
        self.combo_cam.setMinimumHeight(50)
        hbox.addWidget(self.combo_cam, stretch=1)
        
        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedSize(50, 50)
        btn_refresh.setStyleSheet("background-color: #444; color: white; border-radius: 6px; font-size: 20px; border: none;")
        btn_refresh.clicked.connect(self.refresh_cameras)
        hbox.addWidget(btn_refresh)
        card_layout.addLayout(hbox)

        # Connect Button
        self.btn_connect = QPushButton("Connect Camera")
        self.btn_connect.setProperty("class", "primary")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.clicked.connect(self.connect_camera)
        card_layout.addWidget(self.btn_connect)
        
        layout.addWidget(card)

        # Back Button
        btn_back = QPushButton("← Back")
        btn_back.setStyleSheet("background: transparent; color: #888; font-size: 14px; border: none;")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(self.go_back.emit)
        layout.addWidget(btn_back)

    def set_target(self, name, mode):
        self.target_profile = name
        mode_str = "Appending Data" if mode == 'append' else "Reset & New Data"
        self.lbl_info.setText(f"Profile: {name.upper()}  |  Mode: {mode_str}")
        self.refresh_cameras()

    def refresh_cameras(self):
        self.combo_cam.clear()
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    self.combo_cam.addItem(f"[{i}] {name}", i)
            except:
                self.combo_cam.addItem("❌ Camera Search Failed")
        else:
            self.combo_cam.addItem("⚠️ pygrabber missing (Use ID)")
            for i in range(5):
                self.combo_cam.addItem(f"Camera Device {i}", i)

    def connect_camera(self):
        idx = self.combo_cam.currentData()
        if idx is None:
             if self.combo_cam.count() > 0: idx = self.combo_cam.currentIndex()
             else: return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Connecting... ⏳")
        
        self.loader_thread = CameraLoader(idx)
        self.loader_thread.finished.connect(self.on_connected)
        self.loader_thread.error.connect(self.on_error)
        self.loader_thread.start()

    def on_connected(self, cap, idx):
        self.btn_connect.setText("Connect Camera")
        self.btn_connect.setEnabled(True)
        self.camera_ready.emit(cap)

    def on_error(self, msg):
        self.btn_connect.setText("Connect Camera")
        self.btn_connect.setEnabled(True)
        QMessageBox.warning(self, "Connection Error", msg)

# ==============================================================================
# [PAGE 3] Data Collection
# ==============================================================================
class Page3_DataCollection(QWidget):
    go_home = Signal()
    go_train = Signal()

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_profile_dir = ""
        self.is_recording = False
        self.video_writer = None
        self.start_time = 0
        self.accumulated_time = 0.0
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Left: Preview (Full Height)
        self.lbl_camera = QLabel("Camera Preview")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000;")
        
        # [Fix] Expanding -> Ignored
        self.lbl_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        layout.addWidget(self.lbl_camera, stretch=3)

        # Right: Controls (Sidebar)
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #1E1E1E; border-left: 1px solid #333;")
        sidebar.setFixedWidth(400)
        
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(30, 40, 30, 40)
        sb_layout.setSpacing(20)

        # Title
        lbl_title = QLabel("Data Studio")
        lbl_title.setObjectName("Title")
        lbl_title.setStyleSheet("font-size: 24px; border: none;")
        sb_layout.addWidget(lbl_title)

        # Status Card
        self.status_card = QFrame()
        self.status_card.setStyleSheet("background-color: #2D2D2D; border-radius: 10px; padding: 15px;")
        sc_layout = QVBoxLayout(self.status_card)
        
        self.lbl_bg_status = QLabel("❌ Background Missing")
        self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px;")
        sc_layout.addWidget(self.lbl_bg_status)
        
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet("color: white; font-size: 32px; font-family: monospace; font-weight: bold;")
        self.lbl_time.setAlignment(Qt.AlignRight)
        sc_layout.addWidget(self.lbl_time)
        
        sb_layout.addWidget(self.status_card)

        # Buttons
        self.btn_bg = QPushButton("📸  Capture Background")
        self.btn_bg.setProperty("class", "card")
        self.btn_bg.setCursor(Qt.PointingHandCursor)
        self.btn_bg.clicked.connect(self.capture_background)
        sb_layout.addWidget(self.btn_bg)

        self.btn_record = QPushButton("🔴  Start Recording")
        self.btn_record.setProperty("class", "card") # Default style
        self.btn_record.setStyleSheet("text-align: center; font-weight: bold;") 
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.clicked.connect(self.toggle_record)
        sb_layout.addWidget(self.btn_record)

        sb_layout.addStretch()

        # Navigation
        self.btn_train = QPushButton("Next: Start AI Training  →")
        self.btn_train.setProperty("class", "primary")
        self.btn_train.setCursor(Qt.PointingHandCursor)
        self.btn_train.clicked.connect(self.on_train_click)
        sb_layout.addWidget(self.btn_train)

        btn_home = QPushButton("Cancel & Home")
        btn_home.setStyleSheet("background: transparent; color: #666; margin-top: 10px; border: none;")
        btn_home.setCursor(Qt.PointingHandCursor)
        btn_home.clicked.connect(self.on_home_click)
        sb_layout.addWidget(btn_home)

        layout.addWidget(sidebar)

    def setup_session(self, cap, profile_name, profile_dir):
        self.cap = cap
        self.current_profile_dir = profile_dir
        
        bg_path = os.path.join(profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.lbl_bg_status.setText("✅ Background Ready")
            self.lbl_bg_status.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 14px; border:none;")
            self.btn_record.setEnabled(True)
        else:
            self.lbl_bg_status.setText("⚠️ Missing Background")
            self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px; border:none;")
            self.btn_record.setEnabled(False)
            
        self.accumulated_time = self._calc_existing_duration(profile_dir)
        self.update_time_label(0)
        self.timer.start(30)

    def _calc_existing_duration(self, folder):
        total = 0.0
        for f in glob.glob(os.path.join(folder, "train_video_*.mp4")):
            try:
                cap = cv2.VideoCapture(f)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0: total += cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                cap.release()
            except: pass
        return total

    def capture_background(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(os.path.join(self.current_profile_dir, "background.jpg"), frame)
                self.lbl_bg_status.setText("✅ Background Captured")
                self.lbl_bg_status.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 14px; border:none;")
                self.btn_record.setEnabled(True)

    def toggle_record(self):
        if self.btn_record.isChecked():
            timestamp = int(time.time())
            path = os.path.join(self.current_profile_dir, f"train_video_{timestamp}.mp4")
            w, h = int(self.cap.get(3)), int(self.cap.get(4))
            self.video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
            self.is_recording = True
            self.start_time = time.time()
            self.btn_record.setText("⏹  STOP RECORDING")
            self.btn_record.setStyleSheet("background-color: #FF5252; color: white; border-radius: 12px; font-weight: bold; font-size: 16px; border: none;")
        else:
            self.is_recording = False
            if self.video_writer: self.video_writer.release()
            self.accumulated_time += (time.time() - self.start_time)
            self.btn_record.setText("🔴  Start Recording")
            self.btn_record.setProperty("class", "card")
            self.btn_record.setStyleSheet("text-align: center; font-weight: bold;") 
            # Re-polish stylesheet
            self.btn_record.style().unpolish(self.btn_record)
            self.btn_record.style().polish(self.btn_record)

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        
        if self.is_recording:
            self.update_time_label(time.time() - self.start_time)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            if self.video_writer: self.video_writer.write(frame)
        else:
            self.update_time_label(0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.lbl_camera.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)).scaled(
            self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_time_label(self, current):
        total = int(self.accumulated_time + current)
        self.lbl_time.setText(f"{total//60:02d}:{total%60:02d}")

    def on_train_click(self):
        self._stop(); self.go_train.emit()
    def on_home_click(self):
        self._stop(); self.go_home.emit()
    def _stop(self):
        self.is_recording = False
        self.timer.stop()
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()

# ==============================================================================
# [PAGE 4] AI Training
# ==============================================================================
class Page4_AiTraining(QWidget):
    go_home = Signal()

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(25)

        # Header
        layout.addWidget(QLabel("AI Model Generation", objectName="Title"), alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("Processing Pipeline: Labeling -> Training -> Optimization", objectName="Subtitle"), alignment=Qt.AlignCenter)

        # Progress Section
        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)
        
        self.lbl_status = QLabel("Ready to start.")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00ADB5; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.lbl_status)

        # Start Button
        self.btn_start = QPushButton("Start Pipeline")
        self.btn_start.setProperty("class", "primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_pipeline)
        layout.addWidget(self.btn_start)

        # Console Log
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QTextEdit {
                background-color: #111; color: #00FF00; 
                font-family: Consolas; font-size: 12px; 
                border: 1px solid #333; border-radius: 8px; padding: 10px;
            }
        """)
        layout.addWidget(self.log_view)

        # Home Button
        self.btn_home = QPushButton("Done. Go Home")
        self.btn_home.setStyleSheet("background: #333; color: white; padding: 15px; border-radius: 8px; border:none;")
        self.btn_home.setVisible(False)
        self.btn_home.clicked.connect(self.go_home.emit)
        layout.addWidget(self.btn_home)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Processing... Do NOT Close")
        self.log_view.clear()
        
        self.worker = PipelineWorker(self.root_dir)
        self.worker.log_signal.connect(self.log_view.append)
        self.worker.progress_signal.connect(lambda v, t: (self.pbar.setValue(v), self.lbl_status.setText(t)))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

    def on_finished(self):
        self.btn_start.setText("Completed")
        self.lbl_status.setText("All tasks finished successfully.")
        self.btn_home.setVisible(True)