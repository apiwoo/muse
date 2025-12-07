# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (Dynamic Hotkeys & Loading State)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import signal
import time

from PySide6.QtWidgets import QApplication, QDialog, QSplashScreen
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QColor
import qdarktheme

# Add Paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.engine_loop import BeautyWorker
from ui.main_window import MainWindow
from ui.launcher import LauncherDialog

class MuseApp(MainWindow):
    """
    Main Application Class with Dynamic Hotkey Support
    """
    request_profile_switch = Signal(int)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        
        # Connect Signals
        worker.slider_sync_requested.connect(self.beauty_panel.update_sliders_from_config)
        self.request_profile_switch.connect(worker.switch_profile)
        
        # [New] Status Bar Update for Loading
        self.statusBar().showMessage("AI 엔진 초기화 중... 잠시만 기다려주세요.")

    def keyPressEvent(self, event):
        # [New] Dynamic Hotkey Matching
        current_profiles = self.worker.profiles
        matched = False
        
        # Construct KeySequence from Event
        key_int = event.key()
        modifiers = event.modifiers()
        
        # Ignore standalone modifier presses
        if key_int in [Qt.Key_Control, Qt.Key_Shift, Qt.Key_Alt, Qt.Key_Meta]:
            super().keyPressEvent(event)
            return

        # [Fix] PySide6 Compatibility: Use keyCombination() instead of bitwise OR
        # This fixes the TypeError: 'PySide6.QtCore.QKeyCombination.__init__' called with wrong argument types
        pressed_seq = QKeySequence(event.keyCombination())
        
        for idx, p_name in enumerate(current_profiles):
            config = self.worker.profile_mgr.get_config(p_name)
            hotkey_str = config.get("hotkey", "")
            
            if hotkey_str:
                target_seq = QKeySequence(hotkey_str)
                # Compare matches
                if pressed_seq.matches(target_seq) == QKeySequence.ExactMatch:
                    print(f"[KEY] Hotkey '{hotkey_str}' detected -> Switch to '{p_name}'")
                    self.request_profile_switch.emit(idx)
                    self.beauty_panel.set_profile_info(p_name)
                    matched = True
                    break
        
        if not matched:
            # Fallback: 'B' key for Background Reset
            if key_int == Qt.Key_B and modifiers == Qt.NoModifier:
                super().keyPressEvent(event)
            else:
                super().keyPressEvent(event)

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    # [Step 1] Show Launcher First
    launcher = LauncherDialog()
    if launcher.exec() != QDialog.Accepted:
        print("[EXIT] Launcher canceled.")
        sys.exit(0)

    # [Step 2] Get Start Config
    start_profile = launcher.get_start_config()
    print(f"[START] Launching Engine with Profile: {start_profile}")

    # [New] Splash Screen for Loading
    # 간단한 로딩 화면을 띄워 사용자가 멈춘 것으로 오해하지 않게 합니다.
    splash_pix = QPixmap(400, 200)
    splash_pix.fill(QColor("#1E1E1E"))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.showMessage(f"\n\n   MUSE AI 엔진 구동 중...\n   프로필: {start_profile}\n   (최대 10초 소요)", 
                       Qt.AlignCenter, Qt.white)
    splash.show()
    app.processEvents()

    # [Step 3] Start Engine
    # Worker 생성 시 모델 로딩이 발생하므로 시간이 걸립니다.
    worker = BeautyWorker(start_profile=start_profile)
    
    window = MuseApp(worker)
    window.connect_worker(worker)
    
    # Update UI title with profile
    window.beauty_panel.set_profile_info(start_profile)
    
    # Worker 시작
    worker.start()
    
    # 로딩 완료 후 메인 윈도우 표시
    # Worker가 첫 프레임을 보낼 때까지 기다리거나, 일정 시간 후 닫음
    # 여기서는 단순하게 Worker 시작 후 윈도우를 띄우고 스플래시를 닫습니다.
    window.show()
    splash.finish(window)
    
    app.exec()
    
    print("[STOP] [Main] Stopping worker thread...")
    worker.stop()
    worker.wait()
    print("[OK] [Main] Worker stopped. Exiting.")

if __name__ == "__main__":
    main()