# Project MUSE - muse_studio.py
# The All-in-One GUI Launcher for Non-Tech Users
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import cv2
import time
import glob
import shutil
import subprocess
import numpy as np

# [Log Fix] OpenCV 로그 레벨 조정
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# [New] PyGrabber Check
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QStackedWidget, QComboBox, QProgressBar, 
    QMessageBox, QGroupBox, QScrollArea, QFrame, QSizePolicy, QDialog, 
    QLineEdit, QTextEdit
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QIcon

# [Theme Setup]
try:
    import qdarktheme
except ImportError:
    qdarktheme = None

# ==============================================================================
# [Helper Classes] Thread & Dialogs
# ==============================================================================

class CameraLoader(QThread):
    """
    [Background Worker] 카메라 연결 시 UI 멈춤 방지용 스레드
    """
    finished = Signal(object, int) # cap_obj, camera_index
    error = Signal(str)

    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index

    def run(self):
        try:
            # 실제 카메라 연결 시도
            cap = cv2.VideoCapture(self.camera_index)
            
            # 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if cap.isOpened():
                # 연결 성공
                self.finished.emit(cap, self.camera_index)
            else:
                self.error.emit("카메라를 열 수 없습니다.")
        except Exception as e:
            self.error.emit(f"연결 중 오류 발생: {e}")

class PipelineWorker(QThread):
    """
    [New] 원클릭 학습 파이프라인 (라벨링 -> 학습 -> 변환)
    """
    log_signal = Signal(str)
    progress_signal = Signal(int, str) # percent, status_text
    finished_signal = Signal()
    error_signal = Signal(str)

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.tools_dir = os.path.join(root_dir, "tools")

    def run(self):
        try:
            # Step 1: Labeling
            self.progress_signal.emit(10, "Step 1/3: 데이터 가공 중 (Auto-Labeling)...")
            self.run_script(os.path.join(self.tools_dir, "auto_labeling", "run_labeling.py"), ["personal_data"])
            
            # Step 2: Training
            self.progress_signal.emit(40, "Step 2/3: AI 모델 학습 중 (Training)...")
            self.run_script(os.path.join(self.tools_dir, "train_student.py"), ["personal_data"])
            
            # Step 3: Conversion
            self.progress_signal.emit(80, "Step 3/3: 실시간 엔진 변환 중 (Optimization)...")
            self.run_script(os.path.join(self.tools_dir, "convert_student_to_trt.py"), [])
            
            self.progress_signal.emit(100, "완료! 모든 작업이 끝났습니다.")
            self.finished_signal.emit()
            
        except Exception as e:
            self.error_signal.emit(str(e))

    def run_script(self, script_path, args):
        cmd = [sys.executable, script_path] + args
        self.log_signal.emit(f"\n🚀 Executing: {os.path.basename(script_path)}")
        
        # Windows에서 subprocess 실행 시 콘솔 창 숨기기
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            startupinfo=startupinfo
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                self.log_signal.emit(line)
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Script failed with code {process.returncode}")

class ProfileActionDialog(QDialog):
    """
    [Custom Dialog] 기존 프로파일 선택 시 작업 유형 선택 (Append vs Reset)
    """
    def __init__(self, profile_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("작업 유형 선택")
        self.resize(500, 350)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 안내 문구
        lbl_title = QLabel(f"프로파일 [{profile_name}]")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #00ADB5; margin-bottom: 5px;")
        layout.addWidget(lbl_title)
        
        lbl_desc = QLabel("어떤 작업을 진행하시겠습니까?")
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("font-size: 14px; color: #aaa; margin-bottom: 20px;")
        layout.addWidget(lbl_desc)
        
        # 버튼 1: Append
        self.btn_append = QPushButton("이어서 학습 (추가 촬영)\n[기존 데이터 유지 + 새 데이터 추가]")
        self.btn_append.setMinimumHeight(80)
        self.btn_append.setCursor(Qt.PointingHandCursor)
        self.btn_append.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold; 
                background-color: #2196F3; color: white; 
                border-radius: 10px; border: 1px solid #1976D2;
            }
            QPushButton:hover { background-color: #42A5F5; }
        """)
        
        # 버튼 2: Reset
        self.btn_reset = QPushButton("처음부터 다시 (초기화)\n[기존 데이터 백업 후 삭제]")
        self.btn_reset.setMinimumHeight(80)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                font-size: 16px; font-weight: bold; 
                background-color: #F44336; color: white; 
                border-radius: 10px; border: 1px solid #D32F2F;
            }
            QPushButton:hover { background-color: #EF5350; }
        """)
        
        # 버튼 3: Cancel
        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                font-size: 13px;
                background-color: #444; color: white; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #555; }
        """)
        
        layout.addWidget(self.btn_append)
        layout.addWidget(self.btn_reset)
        layout.addSpacing(10)
        layout.addWidget(self.btn_cancel)
        
        # 결과 코드: 1=Append, 2=Reset, 0=Cancel
        self.btn_append.clicked.connect(lambda: self.done(1))
        self.btn_reset.clicked.connect(lambda: self.done(2))
        self.btn_cancel.clicked.connect(lambda: self.done(0))

class NewProfileDialog(QDialog):
    """
    [Custom Dialog] 새 프로파일 이름 입력
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("새 프로파일 생성")
        self.resize(400, 200)
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
        layout.addWidget(QLabel("새 프로파일 이름 입력 (예: side_cam, detail_view)"))
        
        self.input_name = QLineEdit()
        self.input_name.setStyleSheet("padding: 10px; font-size: 14px; background-color: #444; border: 1px solid #666; color: white;")
        layout.addWidget(self.input_name)
        
        btn_box = QHBoxLayout()
        btn_ok = QPushButton("확인")
        btn_ok.clicked.connect(self.accept)
        btn_ok.setStyleSheet("background-color: #00ADB5; color: white; padding: 10px; border-radius: 5px;")
        
        btn_cancel = QPushButton("취소")
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet("background-color: #555; color: white; padding: 10px; border-radius: 5px;")
        
        btn_box.addWidget(btn_ok)
        btn_box.addWidget(btn_cancel)
        layout.addLayout(btn_box)

    def get_name(self):
        return self.input_name.text().strip()

# ==============================================================================
# [PAGE 1] Profile Selection (Start Screen)
# ==============================================================================
class Page1_ProfileSelect(QWidget):
    profile_confirmed = Signal(str, str) # name, mode ('append' or 'reset')

    def __init__(self, personal_data_dir):
        super().__init__()
        self.personal_data_dir = personal_data_dir
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)

        # Title
        title = QLabel("작업할 프로파일을 선택하세요")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #E0E0E0; margin-bottom: 20px;")
        layout.addWidget(title)

        # Scroll Area for Buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(15)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        
        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll)

        # Refresh on Show (나중에 메인윈도우에서 호출)
        self.refresh_profiles()

    def refresh_profiles(self):
        # Clear existing buttons
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)

        # 1. [+ 새로 만들기] Button
        btn_new = QPushButton("[+]  새로 만들기")
        btn_new.setMinimumHeight(70)
        btn_new.setCursor(Qt.PointingHandCursor)
        btn_new.setStyleSheet("""
            QPushButton {
                background-color: #00ADB5; color: white; 
                font-size: 18px; font-weight: bold; border-radius: 10px;
            }
            QPushButton:hover { background-color: #00C4CC; }
        """)
        btn_new.clicked.connect(self.on_click_new)
        self.scroll_layout.addWidget(btn_new)

        # 2. Existing Profiles
        if os.path.exists(self.personal_data_dir):
            profiles = sorted([d for d in os.listdir(self.personal_data_dir) 
                               if os.path.isdir(os.path.join(self.personal_data_dir, d))])
            
            # Default profiles first if exists
            priority = ['front', 'top', 'under']
            sorted_profiles = []
            for p in priority:
                if p in profiles:
                    sorted_profiles.append(p)
                    profiles.remove(p)
            sorted_profiles.extend(profiles)

            for p_name in sorted_profiles:
                btn = QPushButton(f"[{p_name}]")
                btn.setMinimumHeight(60)
                btn.setCursor(Qt.PointingHandCursor)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #393E46; color: #EEEEEE; 
                        font-size: 16px; border-radius: 8px; border: 1px solid #555;
                    }
                    QPushButton:hover { background-color: #4E545E; border-color: #00ADB5; }
                """)
                # Closure capture issue fix: default argument
                btn.clicked.connect(lambda checked=False, name=p_name: self.on_click_existing(name))
                self.scroll_layout.addWidget(btn)

    def on_click_new(self):
        dlg = NewProfileDialog(self)
        if dlg.exec() == QDialog.Accepted:
            name = dlg.get_name()
            if name:
                # 새 프로파일은 무조건 Reset(Init) 모드
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
        self.target_profile = ""
        self.target_mode = ""
        self.loader_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignCenter)

        # Header Info
        self.lbl_info = QLabel("Target: ???")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("font-size: 20px; color: #BBB; font-weight: bold;")
        layout.addWidget(self.lbl_info)

        # Camera Select Group
        grp = QGroupBox("카메라 연결")
        grp.setStyleSheet("QGroupBox { font-size: 16px; font-weight: bold; color: white; border: 1px solid #555; padding: 20px; }")
        grp_layout = QVBoxLayout()
        
        hbox = QHBoxLayout()
        self.combo_cam = QComboBox()
        self.combo_cam.setMinimumHeight(40)
        self.combo_cam.setStyleSheet("font-size: 14px; padding: 5px;")
        hbox.addWidget(self.combo_cam, stretch=1)
        
        btn_refresh = QPushButton("새로고침")
        btn_refresh.setMinimumHeight(40)
        btn_refresh.clicked.connect(self.refresh_cameras)
        hbox.addWidget(btn_refresh)
        grp_layout.addLayout(hbox)
        
        grp_layout.addSpacing(20)
        
        self.btn_connect = QPushButton("카메라 켜기")
        self.btn_connect.setMinimumHeight(80)
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white; 
                font-size: 20px; font-weight: bold; border-radius: 10px;
            }
            QPushButton:hover { background-color: #42A5F5; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_connect.clicked.connect(self.connect_camera)
        grp_layout.addWidget(self.btn_connect)
        
        grp.setLayout(grp_layout)
        layout.addWidget(grp)

        # Back Button
        btn_back = QPushButton("뒤로가기")
        btn_back.setStyleSheet("background-color: transparent; color: #888; text-decoration: underline;")
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(self.go_back.emit)
        layout.addWidget(btn_back)

    def set_target(self, name, mode):
        self.target_profile = name
        self.target_mode = mode
        mode_str = "추가 학습 (Append)" if mode == 'append' else "초기화 (Reset)"
        self.lbl_info.setText(f"Target: [{name}] - {mode_str}")
        self.refresh_cameras()

    def refresh_cameras(self):
        self.combo_cam.clear()
        if HAS_PYGRABBER:
            try:
                graph = FilterGraph()
                devices = graph.get_input_devices()
                for i, name in enumerate(devices):
                    self.combo_cam.addItem(f"[{i}] {name}", i)
            except Exception:
                self.combo_cam.addItem("❌ 카메라 검색 실패")
        else:
            self.combo_cam.addItem("⚠️ pygrabber 없음 (ID로만 연결)")
            for i in range(5):
                self.combo_cam.addItem(f"Camera Device {i}", i)

    def connect_camera(self):
        idx = self.combo_cam.currentData()
        if idx is None: 
            # pygrabber 없을 때 fallback
            if self.combo_cam.count() > 0:
                idx = self.combo_cam.currentIndex()
            else:
                return

        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("연결 중... ⏳")
        
        self.loader_thread = CameraLoader(idx)
        self.loader_thread.finished.connect(self.on_connected)
        self.loader_thread.error.connect(self.on_error)
        self.loader_thread.start()

    def on_connected(self, cap, idx):
        self.btn_connect.setText("카메라 켜기")
        self.btn_connect.setEnabled(True)
        self.camera_ready.emit(cap)

    def on_error(self, msg):
        self.btn_connect.setText("카메라 켜기")
        self.btn_connect.setEnabled(True)
        QMessageBox.warning(self, "오류", msg)

# ==============================================================================
# [PAGE 3] Data Collection
# ==============================================================================
class Page3_DataCollection(QWidget):
    go_home = Signal()
    go_train = Signal() # [New] 학습 페이지로 이동 신호

    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_profile_dir = ""
        self.profile_name = ""
        
        self.is_recording = False
        self.video_writer = None
        self.start_time = 0
        self.accumulated_time = 0.0 # 기존 파일들의 총 길이
        
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Left: Preview ---
        self.lbl_camera = QLabel("Preview")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: black;")
        self.lbl_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.lbl_camera, stretch=3)

        # --- Right: Controls ---
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("background-color: #222; border-left: 1px solid #444;")
        ctrl_frame.setFixedWidth(350)
        vbox = QVBoxLayout(ctrl_frame)
        vbox.setContentsMargins(20, 30, 20, 30)
        vbox.setSpacing(20)

        # 1. Guide Info
        lbl_guide_title = QLabel("⚠️ 촬영 가이드")
        lbl_guide_title.setStyleSheet("color: #FFC107; font-weight: bold; font-size: 16px;")
        vbox.addWidget(lbl_guide_title)
        
        lbl_guide = QLabel("- 옷마다 2~4분가량 촬영\n- 총 10분 이상 권장\n- 다양한 포즈와 각도")
        lbl_guide.setStyleSheet("color: #DDD; font-size: 13px; line-height: 140%;")
        vbox.addWidget(lbl_guide)
        
        vbox.addWidget(self._h_line())

        # 2. Background Control
        lbl_step1 = QLabel("Step 1. 배경 확보")
        lbl_step1.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 16px;")
        vbox.addWidget(lbl_step1)
        
        self.lbl_bg_status = QLabel("배경 없음 (촬영 필요)")
        self.lbl_bg_status.setStyleSheet("color: #F44336; font-weight: bold;")
        vbox.addWidget(self.lbl_bg_status)
        
        self.btn_bg = QPushButton("배경 찍기 (Clean Plate)")
        self.btn_bg.setMinimumHeight(50)
        self.btn_bg.setStyleSheet("background-color: #444; color: white; border-radius: 5px;")
        self.btn_bg.clicked.connect(self.capture_background)
        vbox.addWidget(self.btn_bg)
        
        vbox.addWidget(self._h_line())

        # 3. Recording Control
        lbl_step2 = QLabel("Step 2. 데이터 수집")
        lbl_step2.setStyleSheet("color: #00ADB5; font-weight: bold; font-size: 16px;")
        vbox.addWidget(lbl_step2)

        self.lbl_time = QLabel("현재 클립: 00:00\n누적 시간: 00:00")
        self.lbl_time.setStyleSheet("font-family: Consolas; font-size: 16px; color: white; background: #111; padding: 10px; border-radius: 5px;")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.lbl_time)
        
        self.btn_record = QPushButton("데이터 수집하기 (녹화)")
        self.btn_record.setMinimumHeight(80)
        self.btn_record.setCheckable(True)
        self.btn_record.setEnabled(False) # 배경 찍기 전까지 비활성
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_record.setStyleSheet("""
            QPushButton { 
                background-color: #333; color: #666; 
                font-size: 20px; font-weight: bold; border-radius: 10px;
            }
            QPushButton:enabled {
                background-color: #D32F2F; color: white;
            }
            QPushButton:checked { 
                background-color: #FFEB3B; color: black; border: 2px solid #FBC02D;
            }
        """)
        vbox.addWidget(self.btn_record)
        
        vbox.addStretch()
        
        # [New] Next Step: Train
        self.btn_train = QPushButton("다음: AI 학습하러 가기")
        self.btn_train.setMinimumHeight(60)
        self.btn_train.setCursor(Qt.PointingHandCursor)
        self.btn_train.clicked.connect(self.on_train_click)
        self.btn_train.setStyleSheet("""
            QPushButton {
                background-color: #673AB7; color: white; font-weight: bold; font-size: 16px; border-radius: 10px;
            }
            QPushButton:hover { background-color: #7E57C2; }
        """)
        vbox.addWidget(self.btn_train)

        # 4. Home Button
        btn_home = QPushButton("처음으로 (촬영 종료)")
        btn_home.setMinimumHeight(40)
        btn_home.setStyleSheet("background-color: #333; color: #888; border-radius: 5px;")
        btn_home.clicked.connect(self.on_home_click)
        vbox.addWidget(btn_home)

        layout.addWidget(ctrl_frame)

    def _h_line(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #555;")
        return line

    def setup_session(self, cap, profile_name, profile_dir):
        self.cap = cap
        self.profile_name = profile_name
        self.current_profile_dir = profile_dir
        
        # 1. 배경 확인
        bg_path = os.path.join(profile_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.lbl_bg_status.setText("✅ 배경 확보됨")
            self.lbl_bg_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.btn_record.setEnabled(True)
        else:
            self.lbl_bg_status.setText("⛔ 배경 없음 (촬영 필요)")
            self.lbl_bg_status.setStyleSheet("color: #F44336; font-weight: bold;")
            self.btn_record.setEnabled(False)
            
        # 2. 누적 시간 계산 (기존 영상들)
        self.accumulated_time = self._calc_existing_duration(profile_dir)
        self.update_time_label(0)
        
        # 3. 프리뷰 시작
        self.timer.start(30)

    def _calc_existing_duration(self, folder):
        total_sec = 0.0
        files = glob.glob(os.path.join(folder, "train_video_*.mp4"))
        for f in files:
            try:
                cap = cv2.VideoCapture(f)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        total_sec += (count / fps)
                cap.release()
            except: pass
        return total_sec

    def capture_background(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                path = os.path.join(self.current_profile_dir, "background.jpg")
                cv2.imwrite(path, frame)
                
                self.lbl_bg_status.setText("✅ 배경 확보됨")
                self.lbl_bg_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.btn_record.setEnabled(True)
                QMessageBox.information(self, "성공", "배경이 저장되었습니다.\n이제 데이터 수집이 가능합니다.")

    def toggle_record(self):
        if self.btn_record.isChecked():
            # Start
            timestamp = int(time.time())
            video_path = os.path.join(self.current_profile_dir, f"train_video_{timestamp}.mp4")
            
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            self.is_recording = True
            self.start_time = time.time()
            self.btn_record.setText("녹화 중지 (STOP)")
        else:
            # Stop
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            # 누적 시간 갱신
            clip_dur = time.time() - self.start_time
            self.accumulated_time += clip_dur
            
            self.btn_record.setText("데이터 수집하기 (녹화)")
            QMessageBox.information(self, "저장됨", f"{clip_dur:.1f}초 영상이 저장되었습니다.")

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return

        # UI Update
        current_clip = 0.0
        if self.is_recording:
            current_clip = time.time() - self.start_time
            # Rec Indicator
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if self.video_writer:
                self.video_writer.write(frame)
        
        self.update_time_label(current_clip)
        
        # Display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_camera.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def update_time_label(self, current_clip_sec):
        total_sec = self.accumulated_time + current_clip_sec
        
        def fmt(s):
            m = int(s // 60)
            ss = int(s % 60)
            return f"{m:02d}:{ss:02d}"
        
        self.lbl_time.setText(f"현재 클립: {fmt(current_clip_sec)}\n누적 시간: {fmt(total_sec)}")

    def on_train_click(self):
        self._stop_camera()
        self.go_train.emit()

    def on_home_click(self):
        self._stop_camera()
        self.go_home.emit()

    def _stop_camera(self):
        if self.is_recording:
            # 강제 중지
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

# ==============================================================================
# [PAGE 4] AI Training (Auto Pipeline)
# ==============================================================================
class Page4_AiTraining(QWidget):
    go_home = Signal()

    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(20)

        # Title
        title = QLabel("AI 모델 생성 (원클릭 파이프라인)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #E0E0E0;")
        layout.addWidget(title)

        # Description
        desc = QLabel("버튼을 누르면 [데이터 가공 > 학습 > 변환] 과정이 자동으로 진행됩니다.\n작업이 완료될 때까지 프로그램을 종료하지 마세요.")
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("color: #AAA; font-size: 14px; margin-bottom: 20px;")
        layout.addWidget(desc)

        # Main Button
        self.btn_start = QPushButton("AI 모델 생성 시작 (Start Pipeline)")
        self.btn_start.setMinimumHeight(80)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #E91E63; color: white; 
                font-size: 20px; font-weight: bold; border-radius: 10px;
            }
            QPushButton:hover { background-color: #F06292; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.btn_start.clicked.connect(self.start_pipeline)
        layout.addWidget(self.btn_start)

        # Progress
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setStyleSheet("QProgressBar {height: 30px; border-radius: 5px; text-align: center;} QProgressBar::chunk {background-color: #00ADB5;}")
        layout.addWidget(self.pbar)

        self.lbl_status = QLabel("대기 중...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00ADB5; font-weight: bold;")
        layout.addWidget(self.lbl_status)

        # Log View
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #111; color: #0F0; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.log_view)

        # Home Button (Hidden initially)
        self.btn_home = QPushButton("처음으로 돌아가기")
        self.btn_home.setMinimumHeight(50)
        self.btn_home.setVisible(False)
        self.btn_home.clicked.connect(self.go_home.emit)
        self.btn_home.setStyleSheet("background-color: #333; color: white; border-radius: 5px;")
        layout.addWidget(self.btn_home)

    def start_pipeline(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setText("작업 진행 중... (멈추지 마세요)")
        self.log_view.clear()
        
        self.worker = PipelineWorker(self.root_dir)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def append_log(self, text):
        self.log_view.append(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def update_progress(self, val, text):
        self.pbar.setValue(val)
        self.lbl_status.setText(text)

    def on_finished(self):
        self.btn_start.setText("작업 완료!")
        self.lbl_status.setText("모든 공정이 성공적으로 끝났습니다.")
        self.btn_home.setVisible(True)
        QMessageBox.information(self, "완료", "AI 모델 생성이 완료되었습니다.\n이제 방송을 시작할 수 있습니다.")

    def on_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("다시 시도")
        self.lbl_status.setText(f"오류 발생: {msg}")
        QMessageBox.critical(self, "오류", msg)

# ==============================================================================
# [Main Window] Muse Studio Wizard
# ==============================================================================
class MuseStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE Studio v2.1 - All-in-One Pipeline")
        self.resize(1280, 800)
        
        # Paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.root_dir, "recorded_data")
        self.personal_data_dir = os.path.join(self.data_dir, "personal_data")
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        os.makedirs(self.personal_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "backup"), exist_ok=True)

        self.init_ui()

    def init_ui(self):
        # Main Stack
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Pages
        self.page1 = Page1_ProfileSelect(self.personal_data_dir)
        self.page2 = Page2_CameraConnect()
        self.page3 = Page3_DataCollection(self.data_dir)
        self.page4 = Page4_AiTraining(self.root_dir)

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)
        self.stack.addWidget(self.page4)

        # Connect Signals
        self.page1.profile_confirmed.connect(self.on_profile_confirmed)
        
        self.page2.go_back.connect(lambda: self.stack.setCurrentIndex(0))
        self.page2.camera_ready.connect(self.on_camera_ready)
        
        self.page3.go_home.connect(lambda: self.stack.setCurrentIndex(0))
        self.page3.go_train.connect(lambda: self.stack.setCurrentIndex(3)) # Go to Page 4
        
        self.page4.go_home.connect(lambda: self.stack.setCurrentIndex(0))

    def on_profile_confirmed(self, name, mode):
        # 1. 디렉토리 준비
        target_dir = os.path.join(self.personal_data_dir, name)
        
        if mode == 'reset':
            # 백업 후 초기화
            if os.path.exists(target_dir):
                self.backup_profile(name, target_dir)
                shutil.rmtree(target_dir) # 완전 삭제 후 재생성
            os.makedirs(target_dir, exist_ok=True)
        else:
            # Append 모드: 그냥 폴더 생성 (있으면 유지)
            os.makedirs(target_dir, exist_ok=True)
            
        # 2. 페이지 이동
        self.page2.set_target(name, mode)
        self.current_profile_info = (name, target_dir)
        self.stack.setCurrentIndex(1)

    def on_camera_ready(self, cap):
        # Page 2 -> Page 3
        name, target_dir = self.current_profile_info
        self.page3.setup_session(cap, name, target_dir)
        self.stack.setCurrentIndex(2)

    def backup_profile(self, name, target_dir):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.data_dir, "backup", f"{timestamp}_{name}")
        try:
            shutil.move(target_dir, os.path.join(backup_path, "data"))
            print(f"✅ 백업 완료: {backup_path}")
        except Exception as e:
            print(f"⚠️ 백업 실패: {e}")

# ==============================================================================
# [Legacy Tabs] - Preserved for code safety (Processing / Live)
# ==============================================================================
class ProcessingTab(QWidget):
    # (코드 보존: 이전 버전의 ProcessingTab 로직 유지)
    pass 
    # [Note] 실제 구현체는 필요 시 이전 버전에서 복원하여 사용 가능.
    # 이번 리비전에서는 Recorder Wizard에 집중하기 위해 숨김 처리됨.

def main():
    app = QApplication(sys.argv)
    if qdarktheme:
        qdarktheme.setup_theme("dark")
    else:
        app.setStyle("Fusion")
    
    win = MuseStudio()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()