# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (Multi-Profile Edition)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import signal

# [PySide6 GUI Framework]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, Signal
import qdarktheme

# [System Path Setup]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [MUSE Modules]
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

# [Refactoring] Core Engine Loop
from core.engine_loop import BeautyWorker
from ui.main_window import MainWindow

class MuseApp(MainWindow):
    """
    MainWindow를 상속받아 키보드 단축키 로직을 확장한 메인 앱 클래스
    """
    request_profile_switch = Signal(int)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        
        # Worker -> UI (슬라이더 동기화) 연결
        worker.slider_sync_requested.connect(self.beauty_panel.update_sliders_from_config)
        
        # UI -> Worker (프로파일 변경 요청) 연결
        self.request_profile_switch.connect(worker.switch_profile)

    def keyPressEvent(self, event):
        # 1~9 숫자키 감지 -> 프로파일 전환
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1 # 0-based index
            print(f"⌨️ [Key] Profile Switch Request: {idx + 1}")
            self.request_profile_switch.emit(idx)
            
            # 패널 제목 업데이트 (UI 반응성 향상)
            if idx < len(self.worker.profiles):
                self.beauty_panel.set_profile_info(self.worker.profiles[idx])
        
        elif key == Qt.Key_B:
            # B키는 MainWindow의 기본 동작(배경 리셋) 유지
            super().keyPressEvent(event)
        
        else:
            super().keyPressEvent(event)

def main():
    # Ctrl+C 종료 시그널 처리
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    # Worker 생성 (엔진 로직 스레드)
    worker = BeautyWorker()
    
    # Window 생성 (확장된 MuseApp 사용)
    window = MuseApp(worker)
    
    # Worker와 Window 연결 (영상/파라미터 교환)
    window.connect_worker(worker)
    
    # 실행
    worker.start()
    window.show()
    
    # 앱 루프 실행
    app.exec()
    
    # [Safety] 종료 절차 개선
    print("🛑 [Main] Stopping worker thread...")
    worker.stop()
    worker.wait() # 스레드가 완전히 종료될 때까지 대기
    print("✅ [Main] Worker stopped. Exiting.")

if __name__ == "__main__":
    main()