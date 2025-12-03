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

# [Log Fix] OpenCV 로그 레벨 조정 (불필요한 에러 억제)
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTabWidget, QComboBox, QLineEdit, 
    QTextEdit, QProgressBar, QMessageBox, QGroupBox, QScrollArea,
    QCheckBox, QDialog, QDialogButtonBox, QInputDialog, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QProcess
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont

# [Theme Setup]
try:
    import qdarktheme
except ImportError:
    qdarktheme = None

# ==============================================================================
# [TAB 1] Recorder Widget (Camera Capture & Data Mgmt)
# ==============================================================================
class RecorderTab(QWidget):
    def __init__(self, output_dir, model_dir):
        super().__init__()
        self.output_dir = output_dir
        self.model_dir = model_dir # 모델 백업을 위해 필요
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

        self.init_ui()
        self.refresh_camera_list()
        self.refresh_profile_list()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: Preview Area ---
        preview_layout = QVBoxLayout()
        self.lbl_camera = QLabel("카메라를 먼저 연결해주세요.")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: #000; border: 2px solid #333; color: #666;")
        self.lbl_camera.setMinimumSize(640, 360)
        
        # [Fix] setSizePolicy Error -> Use QSizePolicy.Expanding
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
        
        btn_cam_refresh = QPushButton("R")
        btn_cam_refresh.setFixedWidth(30)
        btn_cam_refresh.clicked.connect(self.refresh_camera_list)
        cam_box.addWidget(btn_cam_refresh)
        ctrl_layout.addLayout(cam_box)

        self.btn_connect = QPushButton("카메라 켜기")
        self.btn_connect.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        self.btn_connect.clicked.connect(self.toggle_camera)
        ctrl_layout.addWidget(self.btn_connect)

        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("------------------------------------------------"))
        ctrl_layout.addSpacing(10)

        # 2. Profile Management
        ctrl_layout.addWidget(QLabel("2. 프로파일(앵글) 선택"))
        
        self.combo_profile = QComboBox()
        self.combo_profile.setEditable(True) # 직접 입력 가능
        self.combo_profile.setPlaceholderText("예: front, side, top...")
        ctrl_layout.addWidget(self.combo_profile)
        
        self.btn_load_profile = QPushButton("프로파일 확정 및 작업 시작")
        self.btn_load_profile.setStyleSheet("background-color: #009688; color: white; padding: 10px; font-weight: bold;")
        self.btn_load_profile.clicked.connect(self.on_profile_decision)
        self.btn_load_profile.setEnabled(False) # 카메라 켜야 활성
        ctrl_layout.addWidget(self.btn_load_profile)
        
        self.lbl_profile_info = QLabel("대기 중...")
        self.lbl_profile_info.setStyleSheet("color: #AAA; font-size: 11px;")
        ctrl_layout.addWidget(self.lbl_profile_info)

        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("------------------------------------------------"))
        ctrl_layout.addSpacing(10)

        # 3. Recording
        ctrl_layout.addWidget(QLabel("3. 데이터 수집"))
        self.btn_bg = QPushButton("배경 촬영 (Clean Plate)")
        self.btn_bg.setStyleSheet("background-color: #FF9800; color: white; padding: 10px;")
        self.btn_bg.clicked.connect(self.capture_background)
        self.btn_bg.setEnabled(False)
        ctrl_layout.addWidget(self.btn_bg)

        self.btn_record = QPushButton("녹화 시작 (REC)")
        self.btn_record.setStyleSheet("""
            QPushButton { background-color: #444; color: #888; padding: 15px; font-size: 16px; font-weight: bold; border-radius: 5px; }
            QPushButton:enabled { background-color: #F44336; color: white; }
            QPushButton:checked { background-color: #B71C1C; }
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
        # [Fix] Reverted to default backend (same as recorder.py)
        for i in range(5):
            # CAP_DSHOW 제거
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.combo_camera.addItem(f"Camera Device {i}", i)
                cap.release()
        if self.combo_camera.count() == 0:
            self.combo_camera.addItem("카메라 없음")

    def refresh_profile_list(self):
        self.combo_profile.clear()
        if os.path.exists(self.personal_data_dir):
            profiles = [d for d in os.listdir(self.personal_data_dir) if os.path.isdir(os.path.join(self.personal_data_dir, d))]
            for p in sorted(profiles):
                self.combo_profile.addItem(p)

    def toggle_camera(self):
        if self.cap is None:
            idx = self.combo_camera.currentData()
            if idx is None: return
            
            # [Fix] Reverted to default backend
            self.cap = cv2.VideoCapture(idx)
            
            # 해상도 설정 시도
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if self.cap.isOpened():
                self.timer.start(30)
                self.btn_connect.setText("카메라 끄기")
                self.btn_load_profile.setEnabled(True)
            else:
                QMessageBox.critical(self, "에러", "카메라를 열 수 없습니다.\n다른 카메라를 선택하거나 장치를 확인하세요.")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.lbl_camera.setPixmap(QPixmap())
            self.btn_connect.setText("카메라 켜기")
            self.btn_load_profile.setEnabled(False)
            self.btn_bg.setEnabled(False)
            self.btn_record.setEnabled(False)

    def update_frame(self):
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
        """
        [핵심 로직] 프로파일 결정 및 분기 처리
        """
        p_name = self.combo_profile.currentText().strip()
        if not p_name:
            QMessageBox.warning(self, "경고", "프로파일 이름을 입력하세요.")
            return

        target_dir = os.path.join(self.personal_data_dir, p_name)
        
        # 1. 기존 프로파일 존재 여부 확인
        if os.path.exists(target_dir):
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("작업 유형 선택")
            msg_box.setText(f"이미 존재하는 프로파일입니다: [{p_name}]\n어떤 작업을 진행하시겠습니까?")
            
            # 버튼 커스텀
            btn_append = msg_box.addButton("이어서 학습 (Append)\n[추가 촬영]", QMessageBox.AcceptRole)
            btn_reset = msg_box.addButton("처음부터 다시 (Reset)\n[전체 백업 후 초기화]", QMessageBox.DestructiveRole)
            btn_cancel = msg_box.addButton("취소", QMessageBox.RejectRole)
            
            msg_box.exec()
            
            if msg_box.clickedButton() == btn_cancel:
                return
            elif msg_box.clickedButton() == btn_reset:
                self._run_reset_logic(p_name, target_dir)
            elif msg_box.clickedButton() == btn_append:
                self._run_append_logic(p_name, target_dir)
        
        else:
            # 2. 신규 프로파일
            ret = QMessageBox.question(self, "신규 생성", f"새 프로파일 [{p_name}]을 생성하시겠습니까?")
            if ret == QMessageBox.Yes:
                self._run_reset_logic(p_name, target_dir, is_new=True)

    def _run_reset_logic(self, p_name, target_dir, is_new=False):
        """백업 및 초기화"""
        if not is_new:
            # Backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_root = os.path.join(self.output_dir, "backup", f"{timestamp}_{p_name}")
            os.makedirs(backup_root, exist_ok=True)
            
            try:
                # 1. 데이터 이동
                shutil.move(target_dir, os.path.join(backup_root, "data"))
                
                # 2. 관련 모델 이동
                model_backup = os.path.join(backup_root, "models")
                os.makedirs(model_backup, exist_ok=True)
                
                # student_front.pth, student_front.engine 등
                model_patterns = [f"student_{p_name}.*", f"student_{p_name}_*"]
                for pat in model_patterns:
                    for f in glob.glob(os.path.join(self.model_dir, pat)):
                        shutil.move(f, model_backup)
                        
                self.lbl_profile_info.setText(f"초기화 완료 (백업위치: {backup_root})")
                QMessageBox.information(self, "안내", "기존 데이터와 모델이 백업되었습니다.\n이제 처음부터 촬영을 시작하세요.")
                
            except Exception as e:
                QMessageBox.critical(self, "백업 오류", f"백업 중 오류 발생: {e}")
                return

        # Re-create empty dir
        os.makedirs(target_dir, exist_ok=True)
        self.current_profile_dir = target_dir
        self.current_profile_name = p_name
        
        # Reset State
        self.clean_plate = None
        self.btn_bg.setEnabled(True)
        self.btn_record.setEnabled(False)
        self.combo_profile.setEnabled(False) # 작업 중 변경 금지
        self.lbl_status.setText(f"Profile [{p_name}] - Initial Mode")

    def _run_append_logic(self, p_name, target_dir):
        """기존 데이터 유지"""
        self.current_profile_dir = target_dir
        self.current_profile_name = p_name
        
        # 배경 로드 시도
        bg_path = os.path.join(target_dir, "background.jpg")
        if os.path.exists(bg_path):
            self.clean_plate = cv2.imread(bg_path)
            self.btn_bg.setEnabled(True) # 다시 찍고 싶을 수도 있으니
            self.btn_record.setEnabled(True)
            self.lbl_status.setText(f"Profile [{p_name}] - Append Mode (BG Loaded)")
            self.lbl_profile_info.setText("기존 데이터에 이어서 녹화합니다.")
        else:
            self.clean_plate = None
            self.btn_bg.setEnabled(True)
            self.btn_record.setEnabled(False)
            self.lbl_status.setText(f"Profile [{p_name}] - Append Mode (BG Missing)")
            QMessageBox.information(self, "안내", "기존 배경 파일이 없습니다.\n배경을 먼저 촬영해주세요.")
            
        self.combo_profile.setEnabled(False)

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
        else:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.btn_record.setText("녹화 시작 (REC)")
            self.lbl_status.setText("Saved.")
            QMessageBox.information(self, "완료", "녹화가 저장되었습니다.\n추가로 더 찍거나 다음 탭으로 이동하세요.")

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