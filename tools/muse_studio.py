# Project MUSE - muse_studio.py
# The All-in-One GUI Launcher for Non-Tech Users
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import shutil
import time
import ctypes

# [Log Fix] OpenCV 로그 레벨 조정
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# [FIX] 30FPS 복구: MSMF 강제 비활성화 코드 제거
# 이 코드가 있으면 DSHOW 모드로 강제되면서 MJPG 압축이 풀려 6FPS로 떨어지는 현상이 발생합니다.
# 기본값(MSMF)을 사용하면 Windows가 자동으로 MJPG 코덱을 사용하여 대역폭을 확보합니다.
# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget
try:
    import qdarktheme
except ImportError:
    qdarktheme = None

# [Module Imports]
# 'studio' 패키지를 찾기 위해 경로 추가 (현재 폴더 기준)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from studio.styles import STYLESHEET
from studio.pages import (
    Page1_ProfileSelect, Page2_CameraConnect, 
    Page3_DataCollection, Page4_AiTraining
)

class MuseStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        # [한글화] 윈도우 타이틀 변경
        self.setWindowTitle("MUSE 스튜디오 v3.0 (한국어 버전)")
        self.resize(1280, 800)
        
        # [Win32 Native Dark Title Bar]
        self._apply_dark_title_bar()
        
        # Apply Global Stylesheet
        self.setStyleSheet(STYLESHEET)
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.personal_data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        os.makedirs(self.personal_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.root_dir, "recorded_data", "backup"), exist_ok=True)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

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
        self.page3.go_home.connect(lambda: self.stack.setCurrentIndex(0))
        self.page3.go_train.connect(lambda: self.stack.setCurrentIndex(3))
        self.page4.go_home.connect(lambda: self.stack.setCurrentIndex(0))

    def _apply_dark_title_bar(self):
        """
        Windows 10/11의 네이티브 제목 표시줄을 다크 모드로 강제 전환합니다.
        DWMWA_USE_IMMERSIVE_DARK_MODE (20) 속성을 사용합니다.
        """
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4
                )
            except Exception as e:
                print(f"⚠️ 다크 모드 타이틀바 적용 실패: {e}")

    def on_profile_confirmed(self, name, mode):
        target = os.path.join(self.personal_data_dir, name)
        if mode == 'reset' and os.path.exists(target):
            shutil.move(target, os.path.join(self.root_dir, "recorded_data", "backup", f"{int(time.time())}_{name}"))
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
    win = MuseStudio()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()