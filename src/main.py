# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (Multi-Profile Edition)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import signal

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, Signal
import qdarktheme

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.engine_loop import BeautyWorker
from ui.main_window import MainWindow

class MuseApp(MainWindow):
    """
    Main Application Class inheriting MainWindow with keyboard shortcuts
    """
    request_profile_switch = Signal(int)

    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        
        worker.slider_sync_requested.connect(self.beauty_panel.update_sliders_from_config)
        self.request_profile_switch.connect(worker.switch_profile)

    def keyPressEvent(self, event):
        # 1~9 Keys -> Profile Switch
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1 # 0-based index
            print(f"[KEY] [Input] Profile Switch Request: {idx + 1}")
            self.request_profile_switch.emit(idx)
            
            if idx < len(self.worker.profiles):
                self.beauty_panel.set_profile_info(self.worker.profiles[idx])
        
        elif key == Qt.Key_B:
            super().keyPressEvent(event)
        
        else:
            super().keyPressEvent(event)

def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    worker = BeautyWorker()
    
    window = MuseApp(worker)
    
    window.connect_worker(worker)
    
    worker.start()
    window.show()
    
    app.exec()
    
    print("[STOP] [Main] Stopping worker thread...")
    worker.stop()
    worker.wait()
    print("[OK] [Main] Worker stopped. Exiting.")

if __name__ == "__main__":
    main()