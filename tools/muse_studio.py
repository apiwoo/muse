# Project MUSE - muse_studio.py
# The All-in-One GUI Launcher for Non-Tech Users
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import shutil
import time
import ctypes
import glob

# [Log Fix] OpenCV 로그 레벨 조정
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# [FIX] 30FPS 복구: MSMF 강제 비활성화 코드 제거
# 이 코드가 있으면 DSHOW 모드로 강제되면서 MJPG 압축이 풀려 6FPS로 떨어지는 현상이 발생합니다.
# 기본값(MSMF)을 사용하면 Windows가 자동으로 MJPG 코덱을 사용하여 대역폭을 확보합니다.
# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout
from PySide6.QtGui import QFontDatabase, QFont
try:
    import qdarktheme
except ImportError:
    qdarktheme = None

# [Module Imports]
# 'studio' 패키지를 찾기 위해 경로 추가 (현재 폴더 기준)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# src 경로 추가 (titlebar, frameless_base 사용)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from studio.styles import STYLESHEET
from studio.pages import (
    Page1_ProfileSelect, Page2_CameraConnect,
    Page3_DataCollection, Page4_AiTraining
)
from ui.titlebar import TitleBar
from ui.frameless_base import FramelessMixin


class MuseStudio(FramelessMixin, QMainWindow):
    def __init__(self):
        super().__init__()

        # [Custom Titlebar] Frameless 윈도우 설정
        self.setup_frameless()

        # [한글화] 윈도우 타이틀 변경
        self.setWindowTitle("MUSE 스튜디오 v3.1 (안전 백업 모드)")
        self.resize(1280, 800)

        # Apply Global Stylesheet
        self.setStyleSheet(STYLESHEET)

        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.personal_data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        os.makedirs(self.personal_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "recorded_data", "backup"), exist_ok=True)

        # [Custom Titlebar] 전체 컨테이너
        central_container = QWidget()
        central_layout = QVBoxLayout(central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # 1. 커스텀 타이틀바
        self.titlebar = TitleBar(self, title="MUSE 스튜디오")
        central_layout.addWidget(self.titlebar)

        # 2. 스택 위젯 (페이지들)
        self.stack = QStackedWidget()
        central_layout.addWidget(self.stack, stretch=1)

        self.setCentralWidget(central_container)

        self.page1 = Page1_ProfileSelect(self.personal_data_dir)
        self.page2 = Page2_CameraConnect()
        self.page3 = Page3_DataCollection(os.path.join(self.root_dir, "recorded_data"))
        self.page4 = Page4_AiTraining(self.root_dir)

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page3)
        self.stack.addWidget(self.page4)

        # Logic Connections
        self.page1.profile_confirmed.connect(self.on_profile_confirmed)
        self.page2.go_back.connect(lambda: self.stack.setCurrentIndex(0))
        self.page2.camera_ready.connect(self.on_camera_ready)
        
        # [New] Page 2 -> Page 4 (Direct Skip for Debugging)
        self.page2.go_train_direct.connect(lambda: self.stack.setCurrentIndex(3))
        
        self.page3.go_home.connect(lambda: self.stack.setCurrentIndex(0))
        self.page3.go_train.connect(lambda: self.stack.setCurrentIndex(3))
        self.page4.go_home.connect(lambda: self.stack.setCurrentIndex(0))

    # [Legacy] 아래 메서드는 커스텀 타이틀바로 대체됨 (참고용으로 주석 처리)
    # def _apply_dark_title_bar(self):
    #     """
    #     Windows 10/11의 네이티브 제목 표시줄을 다크 모드로 강제 전환합니다.
    #     DWMWA_USE_IMMERSIVE_DARK_MODE (20) 속성을 사용합니다.
    #     """
    #     if sys.platform == "win32":
    #         try:
    #             hwnd = int(self.winId())
    #             ctypes.windll.dwmapi.DwmSetWindowAttribute(
    #                 hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4
    #             )
    #         except Exception as e:
    #             print(f"⚠️ 다크 모드 타이틀바 적용 실패: {e}")

    def on_profile_confirmed(self, name, mode):
        target = os.path.join(self.personal_data_dir, name)
        
        # [Safety Logic] Reset 모드일 때 데이터와 모델 모두 백업 (Risk Free Update)
        if mode == 'reset':
            timestamp = int(time.time())
            # 백업 경로: recorded_data/backup/{시간}_{프로필명}
            backup_root = os.path.join(self.root_dir, "recorded_data", "backup", f"{timestamp}_{name}")
            
            # 1. 데이터 폴더 확인
            data_exists = os.path.exists(target)
            
            # 2. 모델 파일 확인 (.pth, .engine, .onnx)
            model_dir = os.path.join(self.root_dir, "assets", "models", "personal")
            model_files = []
            # student_프로필명.확장자 패턴을 찾음
            for ext in [".pth", ".engine", ".onnx"]:
                f_path = os.path.join(model_dir, f"student_{name}{ext}")
                if os.path.exists(f_path):
                    model_files.append(f_path)
            
            # 백업할 것이 하나라도 있으면 백업 수행
            if data_exists or model_files:
                try:
                    os.makedirs(backup_root, exist_ok=True)
                    print(f"\n[BACKUP] [Safety] Starting backup to: {backup_root}")
                    
                    # (1) 영상 데이터 이동 -> backup/data
                    if data_exists:
                        dest_data = os.path.join(backup_root, "data")
                        # shutil.move는 대상 디렉토리가 없으면 해당 이름으로 rename됨
                        shutil.move(target, dest_data)
                        print(f"   -> [DATA] Recorded data moved to 'data/'")
                    
                    # (2) 모델 파일 이동 -> backup/models
                    if model_files:
                        dest_model_dir = os.path.join(backup_root, "models")
                        os.makedirs(dest_model_dir, exist_ok=True)
                        for mf in model_files:
                            shutil.move(mf, dest_model_dir)
                            print(f"   -> [MODEL] {os.path.basename(mf)} moved to 'models/'")
                            
                    print("[BACKUP] Complete. Safe to reset.")
                            
                except Exception as e:
                    print(f"[ERROR] Backup failed: {e}")
                    # 백업 실패시 일단 진행하지만 로그 남김

        # 대상 폴더 생성 (초기화 또는 확인)
        os.makedirs(target, exist_ok=True)
        
        self.page2.set_target(name, mode)
        self.current_info = (name, target)
        self.stack.setCurrentIndex(1)

    def on_camera_ready(self, cap):
        self.page3.setup_session(cap, *self.current_info)
        self.stack.setCurrentIndex(2)

def main():
    app = QApplication(sys.argv)
    if qdarktheme: qdarktheme.setup_theme("dark") # Use as base, overlay STYLESHEET

    # [Font Loading] Pretendard 폰트 로드
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(root_dir, "assets", "fonts")
    if os.path.exists(fonts_dir):
        font_files = glob.glob(os.path.join(fonts_dir, "*.otf")) + glob.glob(os.path.join(fonts_dir, "*.ttf"))
        for font_path in font_files:
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0:
                print(f"[FONT] Loaded: {os.path.basename(font_path)}")

    # [Discord Style] 폰트 설정 - Inter 우선, Pretendard fallback
    app_font = QFont("Inter", 10)
    app_font.setFamilies(["Inter", "Pretendard", "Segoe UI", "Malgun Gothic"])
    app_font.setWeight(QFont.Normal)  # 400
    app_font.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(app_font)

    win = MuseStudio()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()