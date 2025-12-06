# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (Dynamic Hotkeys)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import os
import signal

from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
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

        pressed_seq = QKeySequence(key_int | modifiers)
        
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

    # [Step 3] Start Engine
    worker = BeautyWorker(start_profile=start_profile)
    
    window = MuseApp(worker)
    window.connect_worker(worker)
    
    # Update UI title with profile
    window.beauty_panel.set_profile_info(start_profile)
    
    worker.start()
    window.show()
    
    app.exec()
    
    print("[STOP] [Main] Stopping worker thread...")
    worker.stop()
    worker.wait()
    print("[OK] [Main] Worker stopped. Exiting.")

if __name__ == "__main__":
    main()