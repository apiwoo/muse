# Project MUSE - pages.py
# Wizard Pages for Studio (OpenGL Integrated + Shared Memory Optimization)
# (C) 2025 MUSE Corp.

import os
import cv2
import glob
import time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QFrame, QDialog, QMessageBox, QComboBox, QSizePolicy, QProgressBar, QTextEdit
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QMutex, QMutexLocker
from PySide6.QtGui import QPixmap, QImage

# Import other studio modules
from studio.widgets import NewProfileDialog, ProfileActionDialog
from studio.workers import CameraLoader, PipelineWorker
from studio.gl_widget import CameraGLWidget  # OpenGL Viewport

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
        # [한글화] 환영 메시지
        title = QLabel("MUSE 스튜디오에 오신 것을 환영합니다")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("나만의 AI 모델을 만들기 위해 프로파일을 선택하세요.")
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
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # [한글화] 새 프로파일 버튼
        btn_new = QPushButton("+  새 프로파일 만들기")
        btn_new.setProperty("class", "primary") 
        btn_new.setCursor(Qt.PointingHandCursor)
        btn_new.clicked.connect(self.on_click_new)
        self.scroll_layout.addWidget(btn_new)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #333; margin: 20px 0;")
        self.scroll_layout.addWidget(line)

        if os.path.exists(self.personal_data_dir):
            profiles = sorted([d for d in os.listdir(self.personal_data_dir) 
                               if os.path.isdir(os.path.join(self.personal_data_dir, d))])
            
            priority = ['front', 'top', 'under']
            sorted_profiles = []
            for p in priority:
                if p in profiles:
                    sorted_profiles.append(p)
                    profiles.remove(p)
            sorted_profiles.extend(profiles)

            if sorted_profiles:
                # [한글화] 기존 목록 라벨
                lbl_exist = QLabel("기존 프로파일 목록")
                lbl_exist.setStyleSheet("color: #666; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
                self.scroll_layout.addWidget(lbl_exist)

            for p_name in sorted_profiles:
                btn = QPushButton(f"📂   {p_name.upper()}")
                btn.setProperty("class", "card")
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
    camera_ready = Signal(int) # [Change] Emit Camera Index (int) instead of object
    go_back = Signal()
    go_train_direct = Signal() # [New] Debug Skip Signal

    def __init__(self):
        super().__init__()
        self.loader_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 80, 80, 80)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignCenter)

        # [한글화] 타이틀
        self.lbl_title = QLabel("카메라 연결하기")
        self.lbl_title.setObjectName("Title")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_title)
        
        # [한글화] 서브 타이틀
        self.lbl_info = QLabel("대상: ???")
        self.lbl_info.setObjectName("Subtitle")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_info)

        card = QFrame()
        card.setStyleSheet("background-color: #252525; border-radius: 15px; border: 1px solid #333;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)

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

        # [한글화] 버튼
        self.btn_connect = QPushButton("카메라 연결")
        self.btn_connect.setProperty("class", "primary")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.clicked.connect(self.connect_camera)
        card_layout.addWidget(self.btn_connect)
        
        layout.addWidget(card)

        # [한글화] 뒤로가기 버튼
        btn_back = QPushButton("← 뒤로 가기")
        btn_back.setStyleSheet("background: transparent; color: #888; font-size: 14px; border: none;")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(self.go_back.emit)
        layout.addWidget(btn_back)

        # [임시] 학습 메뉴로 바로 가기 버튼
        self.btn_skip = QPushButton("🚀 학습 메뉴로 바로 가기 (Debug)")
        self.btn_skip.setStyleSheet("background-color: #444; color: #BBB; border: 1px dashed #666; margin-top: 10px; padding: 10px;")
        self.btn_skip.setCursor(Qt.PointingHandCursor)
        self.btn_skip.clicked.connect(self.go_train_direct.emit)
        layout.addWidget(self.btn_skip)

    def set_target(self, name, mode):
        self.target_profile = name
        # [한글화] 모드 설명
        mode_str = "데이터 추가 (Append)" if mode == 'append' else "초기화 및 새로 만들기"
        self.lbl_info.setText(f"프로파일: {name.upper()}  |  모드: {mode_str}")
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
                self.combo_cam.addItem("❌ 카메라 검색 실패")
        else:
            self.combo_cam.addItem("⚠️ pygrabber 없음 (ID로 표시)")
            for i in range(5):
                self.combo_cam.addItem(f"카메라 장치 {i}", i)

    def connect_camera(self):
        idx = self.combo_cam.currentData()
        if idx is None:
             if self.combo_cam.count() > 0: idx = self.combo_cam.currentIndex()
             else: return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("연결 중... ⏳")
        
        self.loader_thread = CameraLoader(idx)
        self.loader_thread.finished.connect(self.on_connected)
        self.loader_thread.error.connect(self.on_error)
        self.loader_thread.start()

    def on_connected(self, cap, idx):
        # [Fix] Close the test connection immediately!
        # If we pass this open 'cap' to another thread, Windows (MSMF) will cause lag.
        # We release it here, and let the RecorderWorker re-open it natively.
        cap.release()
        
        self.btn_connect.setText("카메라 연결")
        self.btn_connect.setEnabled(True)
        self.camera_ready.emit(idx) # Emit INDEX, not OBJECT

    def on_error(self, msg):
        self.btn_connect.setText("카메라 연결")
        self.btn_connect.setEnabled(True)
        QMessageBox.warning(self, "연결 오류", msg)

# ==============================================================================
# [PAGE 3] Data Collection (Optimized)
# ==============================================================================

class RecorderWorker(QThread):
    """
    [Optimized] Shared Memory Worker (Self-Contained)
    - Opens camera INSIDE the thread (Debug Tool Strategy)
    - Enforces MJPG for 30FPS@1080p
    - [FIX] Thread-Safe Video Writing
    """
    time_updated = Signal(float)
    bg_status_updated = Signal(bool)
    
    def __init__(self, cam_index, profile_dir):
        super().__init__()
        self.cam_index = cam_index # Store ID, not Object
        self.profile_dir = profile_dir
        self.running = True
        
        self.cap = None
        self.m_lock = QMutex()
        self.m_frame = None
        self.m_frame_id = 0 
        
        # Internal Recording State
        self.is_recording = False
        
        # Command Flags (Thread Safe Control)
        self.cmd_start_rec = False
        self.cmd_stop_rec = False
        self.req_bg_capture = False
        
        self.video_writer = None
        self.current_start_time = 0
        self.accumulated_time = 0.0
        self.last_reported_int_time = -1

    def _calc_existing_duration(self, folder):
        total = 0.0
        files = glob.glob(os.path.join(folder, "train_video_*.mp4"))
        for f in files:
            try:
                cap = cv2.VideoCapture(f)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0: total += (frames / fps)
                cap.release()
            except: pass
        return total

    def start_recording(self):
        # Trigger flag for loop to handle
        self.cmd_start_rec = True

    def stop_recording(self):
        # Trigger flag for loop to handle
        self.cmd_stop_rec = True

    def trigger_bg_capture(self):
        self.req_bg_capture = True

    def run(self):
        self.accumulated_time = self._calc_existing_duration(self.profile_dir)
        
        # [Strategy] Open Camera LOCALLY (Inside Thread)
        print(f"📸 [Worker] Opening Camera {self.cam_index} Native...")
        self.cap = cv2.VideoCapture(self.cam_index)
        
        # [Force Settings]
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print("📸 [Worker] Capture Loop Started (Thread-Local).")
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read() 
            
            if not ret:
                self.msleep(5)
                continue
            
            # Shared Memory Update
            with QMutexLocker(self.m_lock):
                self.m_frame = frame
                self.m_frame_id += 1 
            
            # --- BACKGROUND CAPTURE ---
            if self.req_bg_capture:
                bg_path = os.path.join(self.profile_dir, "background.jpg")
                os.makedirs(self.profile_dir, exist_ok=True)
                cv2.imwrite(bg_path, frame)
                self.bg_status_updated.emit(True)
                self.req_bg_capture = False
            
            # --- RECORDING LOGIC (THREAD SAFE) ---
            
            # 1. Handle START Command
            if self.cmd_start_rec:
                self.cmd_start_rec = False
                if not self.is_recording:
                    self.is_recording = True
                    self.current_start_time = time.time()
                    
                    timestamp = int(time.time())
                    path = os.path.join(self.profile_dir, f"train_video_{timestamp}.mp4")
                    
                    # [Fix] Use ACTUAL frame size, not property
                    h, w = frame.shape[:2]
                    fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0: fps = 30.0
                    
                    os.makedirs(self.profile_dir, exist_ok=True)
                    self.video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    print(f"🎥 [Worker] Recording started: {w}x{h} @ {fps}fps -> {path}")

            # 2. Handle STOP Command
            if self.cmd_stop_rec:
                self.cmd_stop_rec = False
                if self.is_recording:
                    self.is_recording = False
                    if self.video_writer:
                        self.video_writer.release()
                        self.video_writer = None
                    self.accumulated_time += (time.time() - self.current_start_time)
                    print("🎥 [Worker] Recording stopped.")

            # 3. Write Frame
            if self.is_recording:
                elapsed = time.time() - self.current_start_time
                total_time = self.accumulated_time + elapsed
                
                if self.video_writer and self.video_writer.isOpened():
                    try:
                        self.video_writer.write(frame)
                    except Exception as e:
                        print(f"❌ [Worker] Write Error: {e}")
                
                if int(total_time) > self.last_reported_int_time:
                    self.time_updated.emit(total_time)
                    self.last_reported_int_time = int(total_time)
            
        # Clean up
        if self.video_writer:
            self.video_writer.release()
        if self.cap:
            self.cap.release()
        print("📸 [Worker] Capture Loop Ended.")

    def stop(self):
        self.running = False
        self.wait()

class Page3_DataCollection(QWidget):
    go_home = Signal()
    go_train = Signal()

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        # self.cap = None # No longer needed
        self.recorder_thread = None
        self.current_profile_dir = ""
        self.render_timer = QTimer(self)
        self.last_rendered_id = -1
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.gl_widget = CameraGLWidget()
        self.gl_widget.setMinimumSize(320, 240)
        self.gl_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.gl_widget, stretch=3)

        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #1E1E1E; border-left: 1px solid #333;")
        sidebar.setFixedWidth(400)
        
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setContentsMargins(30, 40, 30, 40)
        sb_layout.setSpacing(20)

        # [한글화]
        lbl_title = QLabel("데이터 스튜디오 (GPU)")
        lbl_title.setObjectName("Title")
        lbl_title.setStyleSheet("font-size: 24px; border: none;")
        sb_layout.addWidget(lbl_title)

        self.status_card = QFrame()
        self.status_card.setStyleSheet("background-color: #2D2D2D; border-radius: 10px; padding: 15px;")
        sc_layout = QVBoxLayout(self.status_card)
        
        self.lbl_bg_status = QLabel("❌ ⚠️ 배경 촬영 필요")
        self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px;")
        sc_layout.addWidget(self.lbl_bg_status)
        
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet("color: white; font-size: 32px; font-family: monospace; font-weight: bold;")
        self.lbl_time.setAlignment(Qt.AlignRight)
        sc_layout.addWidget(self.lbl_time)
        
        sb_layout.addWidget(self.status_card)

        # [한글화]
        self.btn_bg = QPushButton("📸  빈 배경 촬영하기 (단축키 B)")
        self.btn_bg.setProperty("class", "card")
        self.btn_bg.setCursor(Qt.PointingHandCursor)
        self.btn_bg.clicked.connect(self.capture_background)
        sb_layout.addWidget(self.btn_bg)

        # [한글화]
        self.btn_record = QPushButton("🔴  녹화 시작")
        self.btn_record.setProperty("class", "card")
        self.btn_record.setStyleSheet("text-align: center; font-weight: bold;") 
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False)
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.clicked.connect(self.toggle_record)
        sb_layout.addWidget(self.btn_record)

        sb_layout.addStretch()

        # [한글화]
        self.btn_train = QPushButton("다음: AI 학습 시작하기  →")
        self.btn_train.setProperty("class", "primary")
        self.btn_train.setCursor(Qt.PointingHandCursor)
        self.btn_train.clicked.connect(self.on_train_click)
        sb_layout.addWidget(self.btn_train)

        # [한글화]
        self.btn_home = QPushButton("취소하고 홈으로")
        self.btn_home.setStyleSheet("background: transparent; color: #666; margin-top: 10px; border: none;")
        self.btn_home.setCursor(Qt.PointingHandCursor)
        self.btn_home.clicked.connect(self.on_home_click)
        sb_layout.addWidget(self.btn_home)

        layout.addWidget(sidebar)

    def setup_session(self, cam_index, profile_name, profile_dir):
        # cam_index comes from Page2 signal
        self.current_profile_dir = profile_dir
        self.last_rendered_id = -1 
        
        bg_path = os.path.join(profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.on_bg_captured(True)
        else:
            self.lbl_bg_status.setText("⚠️ 배경 촬영 필요")
            self.lbl_bg_status.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 14px; border:none;")
            self.btn_record.setEnabled(False)
            
        # Create Worker with ID, let it open the camera internally
        self.recorder_thread = RecorderWorker(cam_index, self.current_profile_dir)
        self.recorder_thread.time_updated.connect(self.update_time_label)
        self.recorder_thread.bg_status_updated.connect(self.on_bg_captured)
        self.recorder_thread.start()
        
        # High Speed Timer
        self.render_timer.timeout.connect(self.update_view)
        # [FIX] UI 렌더링 부하 감소 (1ms -> 16ms, 약 60FPS)
        # 너무 잦은 UI 갱신 요청은 메인 스레드에 병목을 유발합니다.
        self.render_timer.start(16)

    def update_view(self):
        if not self.recorder_thread: return
        
        frame = None
        curr_id = -1

        with QMutexLocker(self.recorder_thread.m_lock):
            if self.recorder_thread.m_frame is not None:
                frame = self.recorder_thread.m_frame
                curr_id = self.recorder_thread.m_frame_id
        
        if frame is not None and curr_id > self.last_rendered_id:
            self.gl_widget.render(frame)
            self.last_rendered_id = curr_id

    def capture_background(self):
        if self.recorder_thread:
            self.recorder_thread.trigger_bg_capture()

    def on_bg_captured(self, success):
        if success:
            # [한글화]
            self.lbl_bg_status.setText("✅ 배경 준비 완료")
            self.lbl_bg_status.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 14px; border:none;")
            self.btn_record.setEnabled(True)

    def toggle_record(self):
        if not self.recorder_thread: return

        if self.btn_record.isChecked():
            self.recorder_thread.start_recording()
            # [한글화]
            self.btn_record.setText("⏹  녹화 중지")
            self.btn_record.setStyleSheet("background-color: #FF5252; color: white; border-radius: 12px; font-weight: bold; font-size: 16px; border: none;")
        else:
            self.recorder_thread.stop_recording()
            # [한글화]
            self.btn_record.setText("🔴  녹화 시작")
            self.btn_record.setProperty("class", "card")
            self.btn_record.setStyleSheet("text-align: center; font-weight: bold;") 
            self.btn_record.style().unpolish(self.btn_record)
            self.btn_record.style().polish(self.btn_record)

    def update_time_label(self, total_seconds):
        total = int(total_seconds)
        self.lbl_time.setText(f"{total//60:02d}:{total%60:02d}")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_B:
            self.capture_background()
        else:
            super().keyPressEvent(event)

    def on_train_click(self):
        self._stop(); self.go_train.emit()
    def on_home_click(self):
        self._stop(); self.go_home.emit()
    def _stop(self):
        if self.render_timer.isActive():
            self.render_timer.stop()
            
        if self.recorder_thread:
            self.recorder_thread.stop()
            self.recorder_thread = None
        
        # Cap is managed inside worker now
        # if self.cap: self.cap.release() 
        
        if self.gl_widget:
            self.gl_widget.cleanup()

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

        # [한글화]
        layout.addWidget(QLabel("AI 모델 생성 마법사", objectName="Title"), alignment=Qt.AlignCenter)
        layout.addWidget(QLabel("진행 과정: 라벨링 → 학습 → 최적화", objectName="Subtitle"), alignment=Qt.AlignCenter)

        self.pbar = QProgressBar()
        layout.addWidget(self.pbar)
        
        # [한글화]
        self.lbl_status = QLabel("시작할 준비가 되었습니다.")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00ADB5; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(self.lbl_status)

        # [한글화]
        self.btn_start = QPushButton("학습 시작하기")
        self.btn_start.setProperty("class", "primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_pipeline)
        layout.addWidget(self.btn_start)

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

        # [한글화]
        self.btn_home = QPushButton("완료. 홈으로 이동")
        self.btn_home.setStyleSheet("background: #333; color: white; padding: 15px; border-radius: 8px; border:none;")
        self.btn_home.setVisible(False)
        self.btn_home.clicked.connect(self.go_home.emit)
        layout.addWidget(self.btn_home)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        # [한글화]
        self.btn_start.setText("작업 중입니다... 창을 닫지 마세요!")
        self.log_view.clear()
        
        self.worker = PipelineWorker(self.root_dir)
        self.worker.log_signal.connect(self.log_view.append)
        self.worker.progress_signal.connect(lambda v, t: (self.pbar.setValue(v), self.lbl_status.setText(t)))
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "오류 발생", e))
        self.worker.start()

    def on_finished(self):
        # [한글화]
        self.btn_start.setText("작업 완료")
        self.lbl_status.setText("모든 학습 과정이 성공적으로 끝났습니다.")
        self.btn_home.setVisible(True)