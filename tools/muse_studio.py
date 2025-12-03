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
    QLabel, QPushButton, QTabWidget, QComboBox, QLineEdit, 
    QTextEdit, QProgressBar, QMessageBox, QGroupBox, QScrollArea,
    QCheckBox, QDialog, QDialogButtonBox, QInputDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QProcess, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont

# [Theme Setup]
try:
    import qdarktheme
except ImportError:
    qdarktheme = None

# ==============================================================================
# [Helper Classes] Thread & Dialog
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
            # 실제 카메라 연결 시도 (시간이 걸리는 작업)
            cap = cv2.VideoCapture(self.camera_index)
            
            # 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            if cap.isOpened():
                # 연결 성공 시 객체 반환
                self.finished.emit(cap, self.camera_index)
            else:
                self.error.emit("카메라를 열 수 없습니다.")
        except Exception as e:
            self.error.emit(f"연결 중 오류 발생: {e}")

class ProfileActionDialog(QDialog):
    """
    [Custom Dialog] 버튼 크기를 키운 작업 선택창
    """
    def __init__(self, profile_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("작업 유형 선택")
        self.resize(500, 350) # 넉넉한 크기
        self.setStyleSheet("background-color: #2b2b2b; color: #ffffff;")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 안내 문구
        lbl_title = QLabel(f"프로파일 [{profile_name}]이(가) 이미 존재합니다.")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ADB5; margin-bottom: 5px;")
        layout.addWidget(lbl_title)
        
        lbl_desc = QLabel("어떤 작업을 진행하시겠습니까?")
        lbl_desc.setAlignment(Qt.AlignCenter)
        lbl_desc.setStyleSheet("font-size: 14px; color: #aaa; margin-bottom: 20px;")
        layout.addWidget(lbl_desc)
        
        # 버튼 1: Append
        self.btn_append = QPushButton("이어서 학습 (Append)\n[추가 촬영 데이터 수집]")
        self.btn_append.setMinimumHeight(70)
        self.btn_append.setCursor(Qt.PointingHandCursor)
        self.btn_append.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold; 
                background-color: #2196F3; color: white; 
                border-radius: 10px; border: 1px solid #1976D2;
            }
            QPushButton:hover { background-color: #42A5F5; }
        """)
        
        # 버튼 2: Reset
        self.btn_reset = QPushButton("처음부터 다시 (Reset)\n[기존 데이터 백업 후 초기화]")
        self.btn_reset.setMinimumHeight(70)
        self.btn_reset.setCursor(Qt.PointingHandCursor)
        self.btn_reset.setStyleSheet("""
            QPushButton {
                font-size: 15px; font-weight: bold; 
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
                background-color: #555; color: white; 
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #666; }
        """)
        
        layout.addWidget(self.btn_append)
        layout.addWidget(self.btn_reset)
        layout.addSpacing(10)
        layout.addWidget(self.btn_cancel)
        
        # 결과 코드: 1=Append, 2=Reset, 0=Cancel
        self.btn_append.clicked.connect(lambda: self.done(1))
        self.btn_reset.clicked.connect(lambda: self.done(2))
        self.btn_cancel.clicked.connect(lambda: self.done(0))

# ==============================================================================
# [TAB 1] Recorder Widget
# ==============================================================================
class RecorderTab(QWidget):
    def __init__(self, output_dir, model_dir):
        super().__init__()
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.personal_data_dir = os.path.join(output_dir, "personal_data")
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_recording = False
        self.video_writer = None
        self.record_start_time = 0
        self.clean_plate = None
        self.current_profile_dir = ""
        self.current_profile_name = ""
        
        self.loader_thread = None # 카메라 로더 스레드

        self.init_ui()
        self.refresh_camera_list()
        self.refresh_profile_list()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: Preview Area ---
        preview_layout = QVBoxLayout()
        self.lbl_camera = QLabel("카메라 연결 대기 중...")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000; border: 2px solid #333; color: #666;")
        self.lbl_camera.setMinimumSize(640, 360)
        self.lbl_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        preview_layout.addWidget(self.lbl_camera)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.lbl_status)
        
        layout.addLayout(preview_layout, stretch=2)

        # --- Right: Controls ---
        control_panel = QGroupBox("스튜디오 제어")
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(15)

        # 1. Camera Select
        ctrl_layout.addWidget(QLabel("1. 카메라 연결"))
        self.combo_camera = QComboBox()
        cam_box = QHBoxLayout()
        cam_box.addWidget(self.combo_camera)
        
        self.btn_cam_refresh = QPushButton("R")
        self.btn_cam_refresh.setFixedWidth(40) # 크기 증가
        self.btn_cam_refresh.setStyleSheet("font-weight: bold; color: #00ADB5; border: 1px solid #444;")
        self.btn_cam_refresh.clicked.connect(self.refresh_camera_list)
        cam_box.addWidget(self.btn_cam_refresh)
        ctrl_layout.addLayout(cam_box)

        self.btn_connect = QPushButton("카메라 켜기")
        self.btn_connect.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_connect.clicked.connect(self.toggle_camera)
        ctrl_layout.addWidget(self.btn_connect)

        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("------------------------------------------------"))
        ctrl_layout.addSpacing(10)

        # 2. Profile Management
        ctrl_layout.addWidget(QLabel("2. 프로파일(앵글) 선택"))
        
        self.combo_profile = QComboBox()
        self.combo_profile.setEditable(True)
        self.combo_profile.setPlaceholderText("예: front, side, top...")
        ctrl_layout.addWidget(self.combo_profile)
        
        self.btn_load_profile = QPushButton("프로파일 확정 및 작업 시작")
        self.btn_load_profile.setStyleSheet("background-color: #009688; color: white; padding: 12px; font-weight: bold; font-size: 13px; border-radius: 5px;")
        self.btn_load_profile.clicked.connect(self.on_profile_decision)
        # self.btn_load_profile.setEnabled(False) -> [Change] 항상 활성화 (사용자가 원함)
        ctrl_layout.addWidget(self.btn_load_profile)
        
        self.lbl_profile_info = QLabel("프로파일을 입력하고 확정하세요.")
        self.lbl_profile_info.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_profile_info)

        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("------------------------------------------------"))
        ctrl_layout.addSpacing(10)

        # 3. Recording
        ctrl_layout.addWidget(QLabel("3. 데이터 수집"))
        self.btn_bg = QPushButton("배경 촬영 (Clean Plate)")
        self.btn_bg.setStyleSheet("background-color: #FF9800; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_bg.clicked.connect(self.capture_background)
        self.btn_bg.setEnabled(False)
        ctrl_layout.addWidget(self.btn_bg)

        # [Change] 녹화 버튼 강조 및 텍스트 변경
        self.btn_record = QPushButton("녹화")
        self.btn_record.setMinimumHeight(60)
        self.btn_record.setStyleSheet("""
            QPushButton { 
                background-color: #333; color: #666; 
                font-size: 20px; font-weight: bold; 
                border-radius: 10px; border: 2px solid #222;
            }
            QPushButton:enabled { 
                background-color: #D32F2F; color: white; 
                border-color: #B71C1C;
            }
            QPushButton:checked { 
                background-color: #FFEB3B; color: black; 
                border-color: #FBC02D;
            }
        """)
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)
        self.btn_record.setEnabled(False)
        ctrl_layout.addWidget(self.btn_record)

        ctrl_layout.addStretch()
        control_panel.setLayout(ctrl_layout)
        layout.addWidget(control_panel, stretch=1)

    # --- Logic ---

    def refresh_camera_list(self):
        self.combo_camera.clear()
        
        if not HAS_PYGRABBER:
            self.combo_camera.addItem("⚠️ 설치 필요: pygrabber")
            self.btn_connect.setEnabled(False)
            return
        
        try:
            self.btn_connect.setEnabled(True)
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for i, name in enumerate(devices):
                self.combo_camera.addItem(f"[{i}] {name}", i)
        except Exception as e:
            self.combo_camera.addItem("❌ 장치 검색 실패")
            print(f"Camera Scan Error: {e}")

        if self.combo_camera.count() == 0:
            self.combo_camera.addItem("카메라 없음")

    def refresh_profile_list(self):
        self.combo_profile.clear()
        if os.path.exists(self.personal_data_dir):
            profiles = [d for d in os.listdir(self.personal_data_dir) if os.path.isdir(os.path.join(self.personal_data_dir, d))]
            for p in sorted(profiles):
                self.combo_profile.addItem(p)

    def toggle_camera(self):
        # 1. 카메라 끄기 (이미 켜져있을 때)
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.lbl_camera.setPixmap(QPixmap())
            self.lbl_camera.setText("카메라 연결 해제됨")
            self.btn_connect.setText("카메라 켜기")
            self.btn_connect.setStyleSheet("background-color: #2196F3; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
            self.btn_bg.setEnabled(False)
            self.btn_record.setEnabled(False)
            return

        # 2. 카메라 켜기 (로더 스레드 시작)
        idx = self.combo_camera.currentData()
        if idx is None: return

        if self.loader_thread and self.loader_thread.isRunning():
            return # 이미 로딩 중

        self.btn_connect.setText("연결 중... ⏳")
        self.btn_connect.setEnabled(False) # 중복 클릭 방지
        self.lbl_camera.setText("카메라 초기화 중입니다...\n잠시만 기다려주세요.")
        
        self.loader_thread = CameraLoader(idx)
        self.loader_thread.finished.connect(self.on_camera_loaded)
        self.loader_thread.error.connect(self.on_camera_error)
        self.loader_thread.start()

    def on_camera_loaded(self, cap_obj, idx):
        self.cap = cap_obj
        self.timer.start(30)
        self.btn_connect.setText("카메라 끄기")
        self.btn_connect.setStyleSheet("background-color: #555; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.btn_connect.setEnabled(True)
        self.lbl_camera.setText("")
        
        # 프로파일이 이미 로드된 상태라면 버튼 활성화
        if self.current_profile_dir:
            self.btn_bg.setEnabled(True)
            if self.clean_plate is not None:
                self.btn_record.setEnabled(True)

    def on_camera_error(self, msg):
        self.btn_connect.setText("카메라 켜기")
        self.btn_connect.setEnabled(True)
        self.lbl_camera.setText(f"❌ {msg}")
        QMessageBox.warning(self, "연결 실패", msg)

    def update_frame(self):
        if not self.cap: return
        ret, frame = self.cap.read()
        if not ret: return
        
        # Overlay Guide
        if self.is_recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_camera.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
            elapsed = time.time() - self.record_start_time
            self.lbl_status.setText(f"Recording... {elapsed:.1f}s")

    def on_profile_decision(self):
        p_name = self.combo_profile.currentText().strip()
        if not p_name:
            QMessageBox.warning(self, "경고", "프로파일 이름을 입력하세요.")
            return

        target_dir = os.path.join(self.personal_data_dir, p_name)
        
        # [Workflow] 카메라가 안 켜져 있으면 자동 연결 시도
        if self.cap is None:
            print("💡 카메라 자동 연결 시도...")
            self.toggle_camera() 
            # toggle_camera는 비동기이므로, 폴더 설정은 일단 진행하되
            # 버튼 활성화는 on_camera_loaded에서 처리됨

        # 1. 기존 프로파일 존재 여부 확인
        if os.path.exists(target_dir):
            # [Change] Custom Dialog 사용
            dlg = ProfileActionDialog(p_name, self)
            result = dlg.exec() # 1:Append, 2:Reset, 0:Cancel
            
            if result == 0: return
            elif result == 2: self._run_reset_logic(p_name, target_dir)
            elif result == 1: self._run_append_logic(p_name, target_dir)
        else:
            # 2. 신규 프로파일
            ret = QMessageBox.question(self, "신규 생성", f"새 프로파일 [{p_name}]을 생성하시겠습니까?")
            if ret == QMessageBox.Yes:
                self._run_reset_logic(p_name, target_dir, is_new=True)

    def _run_reset_logic(self, p_name, target_dir, is_new=False):
        """백업 및 초기화"""
        if not is_new:
            # Backup Logic (생략 없이 유지)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_root = os.path.join(self.output_dir, "backup", f"{timestamp}_{p_name}")
            os.makedirs(backup_root, exist_ok=True)
            try:
                shutil.move(target_dir, os.path.join(backup_root, "data"))
                model_backup = os.path.join(backup_root, "models")
                os.makedirs(model_backup, exist_ok=True)
                model_patterns = [f"student_{p_name}.*", f"student_{p_name}_*"]
                for pat in model_patterns:
                    for f in glob.glob(os.path.join(self.model_dir, pat)):
                        shutil.move(f, model_backup)
                self.lbl_profile_info.setText(f"✅ 초기화됨 (백업: {timestamp})")
                QMessageBox.information(self, "안내", "백업 완료. 촬영을 시작하세요.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"백업 실패: {e}")
                return

        os.makedirs(target_dir, exist_ok=True)
        self.current_profile_dir = target_dir
        self.current_profile_name = p_name
        self.clean_plate = None
        
        # 버튼 상태 업데이트
        if self.cap:
            self.btn_bg.setEnabled(True)
            self.btn_record.setEnabled(False)
        self.lbl_status.setText(f"Profile [{p_name}] - Initial Mode")

    def _run_append_logic(self, p_name, target_dir):
        self.current_profile_dir = target_dir
        self.current_profile_name = p_name
        
        bg_path = os.path.join(target_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.clean_plate = cv2.imread(bg_path)
            if self.cap:
                self.btn_bg.setEnabled(True)
                self.btn_record.setEnabled(True)
            self.lbl_status.setText(f"Profile [{p_name}] - Append Mode")
            self.lbl_profile_info.setText("기존 데이터에 이어서 녹화합니다.")
        else:
            self.clean_plate = None
            if self.cap:
                self.btn_bg.setEnabled(True)
                self.btn_record.setEnabled(False)
            self.lbl_status.setText(f"Profile [{p_name}] - Append Mode (No BG)")
            QMessageBox.information(self, "안내", "배경 이미지가 없습니다.\n배경을 먼저 촬영해주세요.")

    def capture_background(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                save_path = os.path.join(self.current_profile_dir, "background.jpg")
                cv2.imwrite(save_path, frame)
                self.clean_plate = frame
                self.btn_record.setEnabled(True)
                QMessageBox.information(self, "성공", "배경이 저장되었습니다.")

    def toggle_record(self):
        if self.btn_record.isChecked():
            timestamp = int(time.time())
            video_path = os.path.join(self.current_profile_dir, f"train_video_{timestamp}.mp4")
            
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            self.is_recording = True
            self.record_start_time = time.time()
            self.btn_record.setText("녹화 중지 (STOP)")
            self.btn_load_profile.setEnabled(False)
            self.btn_record.setStyleSheet("""
                QPushButton { 
                    background-color: #FFEB3B; color: black; 
                    font-size: 20px; font-weight: bold; 
                    border-radius: 10px; border: 2px solid #FBC02D;
                }
            """)
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.btn_record.setText("녹화")
            self.lbl_status.setText("Saved.")
            self.btn_record.setStyleSheet("""
                QPushButton { 
                    background-color: #D32F2F; color: white; 
                    font-size: 20px; font-weight: bold; 
                    border-radius: 10px; border: 2px solid #B71C1C;
                }
            """)
            QMessageBox.information(self, "완료", "녹화가 저장되었습니다.")

# ==============================================================================
# [TAB 2] Processing Tab (Labeling -> Training -> Conversion)
# ==============================================================================
class ProcessWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal()

    def __init__(self, command, args):
        super().__init__()
        self.command = command
        self.args = args

    def run(self):
        cmd = [sys.executable, self.command] + self.args
        self.log_signal.emit(f"🚀 실행 중: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                self.log_signal.emit(line)
                if "[PROGRESS]" in line:
                    try:
                        val = int(line.split("]")[-1].strip().replace("%", ""))
                        self.progress_signal.emit(val)
                    except: pass
        
        process.wait()
        self.log_signal.emit("✅ 작업 완료.")
        self.finished_signal.emit()

class ProcessingTab(QWidget):
    def __init__(self, root_dir, data_dir):
        super().__init__()
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 1. Profile Info
        grp_info = QGroupBox("작업 정보")
        info_layout = QVBoxLayout()
        self.lbl_target = QLabel("현재 녹화된 데이터를 자동으로 감지하여 처리합니다.")
        info_layout.addWidget(self.lbl_target)
        grp_info.setLayout(info_layout)
        layout.addWidget(grp_info)

        # 2. Action Buttons
        grp_actions = QGroupBox("자동 학습 파이프라인")
        act_layout = QHBoxLayout()
        
        self.btn_step1 = QPushButton("Step 1: 데이터 가공\n(Labeling)")
        self.btn_step1.clicked.connect(lambda: self.run_process("labeling"))
        
        self.btn_step2 = QPushButton("Step 2: AI 학습\n(Training)")
        self.btn_step2.clicked.connect(lambda: self.run_process("training"))
        
        self.btn_step3 = QPushButton("Step 3: 최적화\n(Conversion)")
        self.btn_step3.clicked.connect(lambda: self.run_process("conversion"))

        for btn in [self.btn_step1, self.btn_step2, self.btn_step3]:
            btn.setStyleSheet("padding: 15px; font-weight: bold; font-size: 14px;")
            act_layout.addWidget(btn)
        
        grp_actions.setLayout(act_layout)
        layout.addWidget(grp_actions)

        # 3. Log
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setStyleSheet("QProgressBar {height: 30px; border-radius: 5px;} QProgressBar::chunk {background-color: #00ADB5;}")
        layout.addWidget(self.pbar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("background-color: #111; color: #0F0; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.log_view)

    def run_process(self, step):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "경고", "이미 작업이 진행 중입니다.")
            return

        tools_dir = os.path.join(self.root_dir, "tools")
        script = ""
        args = []

        if step == "labeling":
            script = os.path.join(tools_dir, "auto_labeling", "run_labeling.py")
            args = ["personal_data"] 
            self.log_view.append("\n=== 스마트 라벨링 시작 ===")
            self.log_view.append("기존 데이터는 유지하고, 새로운 영상만 처리합니다.")
        elif step == "training":
            script = os.path.join(tools_dir, "train_student.py")
            args = ["personal_data"]
            self.log_view.append("\n=== 모델 학습 시작 ===")
            self.log_view.append("전체 데이터를 사용하여 모델을 정밀 튜닝합니다.")
        elif step == "conversion":
            script = os.path.join(tools_dir, "convert_student_to_trt.py")
            args = []
            self.log_view.append("\n=== 모델 변환 시작 ===")

        self.pbar.setValue(0)
        self.set_buttons_enabled(False)

        self.worker = ProcessWorker(script, args)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.pbar.setValue)
        self.worker.finished_signal.connect(lambda: self.set_buttons_enabled(True))
        self.worker.start()

    def append_log(self, text):
        self.log_view.append(text)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_buttons_enabled(self, enabled):
        self.btn_step1.setEnabled(enabled)
        self.btn_step2.setEnabled(enabled)
        self.btn_step3.setEnabled(enabled)

# ==============================================================================
# [Main Window]
# ==============================================================================
class MuseStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MUSE Studio v2.0 - Creator Workflow")
        self.resize(1200, 800)
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.root_dir, "recorded_data")
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")

        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { padding: 12px 25px; font-weight: bold; font-size: 14px; }
            QTabBar::tab:selected { background: #00ADB5; color: white; }
        """)

        self.tab_record = RecorderTab(self.data_dir, self.model_dir)
        tabs.addTab(self.tab_record, "1. 촬영 및 관리 (Manage)")

        self.tab_process = ProcessingTab(self.root_dir, os.path.join(self.data_dir, "personal_data"))
        tabs.addTab(self.tab_process, "2. AI 처리 (Process)")

        tab_launch = QWidget()
        vbox = QVBoxLayout()
        lbl = QLabel("방송 준비가 완료되었습니다.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 20px; color: #AAA; font-weight: bold;")
        
        btn_run = QPushButton("LIVE START (방송 시스템 가동)")
        btn_run.setMinimumHeight(100)
        btn_run.setStyleSheet("font-size: 28px; font-weight: bold; background-color: #E91E63; color: white; border-radius: 15px;")
        btn_run.clicked.connect(self.launch_system)
        
        vbox.addStretch()
        vbox.addWidget(lbl)
        vbox.addWidget(btn_run)
        vbox.addStretch()
        tab_launch.setLayout(vbox)
        tabs.addTab(tab_launch, "3. 방송 시작 (Live)")

        self.setCentralWidget(tabs)

    def launch_system(self):
        script = os.path.join(self.root_dir, "tools", "run_muse.py")
        subprocess.Popen([sys.executable, script])

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