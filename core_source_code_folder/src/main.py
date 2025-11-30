# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (GUI Version)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import time
import cv2
import numpy as np
import os

# [PySide6 GUI Framework]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, Slot
import qdarktheme

# [System Path Setup]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# [MUSE Modules]
from utils.cuda_helper import setup_cuda_environment
setup_cuda_environment()

from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.tracking.facemesh import FaceMesh
# [New] BodyTracker Import
from ai.tracking.body_tracker import BodyTracker 
from graphics.beauty_engine import BeautyEngine
from ui.main_window import MainWindow

# High-Performance GPU Library Check
try:
    import cupy as cp
except ImportError:
    cp = None

class BeautyWorker(QThread):
    """
    [Background Thread]
    UI 멈춤(Freezing) 방지를 위해 무거운 AI/영상 처리는 별도 스레드에서 수행합니다.
    """
    frame_processed = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.params = {
            'eye_scale': 0.0, 
            'face_v': 0.0,
            'waist_slim': 0.0,
            'show_body_debug': False
        }

        # 설정
        self.DEVICE_ID = 1
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30

    def run(self):
        print("🧵 [Worker] 뷰티 프로세싱 스레드 시작")

        try:
            self.input_mgr = InputManager(device_id=self.DEVICE_ID, width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.tracker = FaceMesh(root_dir="assets")
            # [Step 1] Body Tracker 초기화
            self.body_tracker = BodyTracker()
            self.beauty_engine = BeautyEngine()
        except Exception as e:
            print(f"❌ [Worker] 초기화 실패: {e}")
            return

        prev_time = time.time()
        frame_count = 0

        while self.running:
            # [Step 1] Input
            frame_gpu, ret = self.input_mgr.read()
            if not ret:
                self.msleep(10)
                continue

            # [Step 2] AI Processing
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # 얼굴 트래킹
            faces = self.tracker.process(frame_cpu)
            
            # [New] 바디 트래킹
            body_landmarks = self.body_tracker.process(frame_cpu)

            # [Step 3] Beauty Processing (Warping)
            # 얼굴과 몸 정보를 모두 엔진에 전달
            frame_cpu = self.beauty_engine.process(frame_cpu, faces, body_landmarks, self.params)

            # [Debug] 몸 뼈대 그리기 (체크박스가 켜져있을 때만)
            if self.params.get('show_body_debug', False):
                frame_cpu = self.body_tracker.draw_debug(frame_cpu, body_landmarks)

            # [Step 4] Output
            self.virtual_cam.send(frame_cpu)
            
            self.frame_processed.emit(frame_cpu)

            # [Step 5] FPS Log
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                print(f"⚡ FPS: {frame_count} | Params: {self.params}")
                frame_count = 0
                prev_time = curr_time

        # 리소스 정리
        self.input_mgr.release()
        self.virtual_cam.close()
        print("🧵 [Worker] 스레드 종료")

    @Slot(dict)
    def update_params(self, new_params):
        """UI 슬라이더 변경 시 호출되는 슬롯"""
        self.params = new_params.copy()

    def stop(self):
        self.running = False
        self.wait()

def main():
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    window = MainWindow()
    worker = BeautyWorker()
    window.connect_worker(worker)
    
    worker.start()
    window.show()
    
    print("🚀 [System] MUSE GUI 가동 완료.")
    exit_code = app.exec()
    
    worker.stop()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()