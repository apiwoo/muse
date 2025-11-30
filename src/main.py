# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (GUI Version)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import time
import cv2
import numpy as np
import os
import signal # [Fix] 시그널 모듈 추가

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
        
        # 자원 핸들 초기화
        self.input_mgr = None
        self.virtual_cam = None
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None

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
            self.body_tracker = BodyTracker()
            self.beauty_engine = BeautyEngine()
        except Exception as e:
            print(f"❌ [Worker] 초기화 실패: {e}")
            return

        prev_time = time.time()
        frame_count = 0

        while self.running:
            # [Step 1] Input
            if self.input_mgr:
                frame_gpu, ret = self.input_mgr.read()
            else:
                break
                
            if not ret:
                self.msleep(10)
                continue

            # [Step 2] AI Processing
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu = frame_gpu.get()
            else:
                frame_cpu = frame_gpu

            # 얼굴 트래킹
            faces = []
            if self.tracker:
                faces = self.tracker.process(frame_cpu)
            
            # 바디 트래킹
            body_landmarks = None
            if self.body_tracker:
                body_landmarks = self.body_tracker.process(frame_cpu)

            # [Step 3] Beauty Processing (Warping + Segmentation)
            # 얼굴과 몸 정보를 모두 엔진에 전달
            if self.beauty_engine:
                frame_cpu = self.beauty_engine.process(frame_cpu, faces, body_landmarks, self.params)

            # [Debug] 몸 뼈대 그리기 (체크박스가 켜져있을 때만)
            if self.params.get('show_body_debug', False) and self.body_tracker:
                frame_cpu = self.body_tracker.draw_debug(frame_cpu, body_landmarks)

            # [Step 4] Output
            if self.virtual_cam:
                self.virtual_cam.send(frame_cpu)
            
            self.frame_processed.emit(frame_cpu)

            # [Step 5] FPS Log
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                print(f"⚡ FPS: {frame_count} | Params: {self.params}")
                frame_count = 0
                prev_time = curr_time

        # 루프 탈출 후 정리
        self.cleanup()
        print("🧵 [Worker] 스레드 종료")

    def cleanup(self):
        """자원 강제 해제 (중복 호출 방지)"""
        # 이미 해제되었다면 패스
        if self.input_mgr is None and self.virtual_cam is None:
            return

        print("🧹 [Worker] 자원 정리 시작...")
        if self.input_mgr:
            self.input_mgr.release()
            self.input_mgr = None
        
        if self.virtual_cam:
            self.virtual_cam.close()
            self.virtual_cam = None
            
        # AI 엔진 메모리 해제
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None
        print("✨ [Worker] 자원 정리 완료")

    @Slot(dict)
    def update_params(self, new_params):
        """UI 슬라이더 변경 시 호출되는 슬롯"""
        self.params = new_params.copy()

    def stop(self):
        self.running = False
        # 스레드가 루프를 돌고 있다면 빠져나오게 함

def main():
    # [Fix] Ctrl+C (SIGINT) 시그널을 운영체제 기본 동작(종료)으로 처리
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")

    window = MainWindow()
    worker = BeautyWorker()
    
    # [Core Fix] 데몬 스레드로 설정
    worker.setTerminationEnabled(True) 
    
    window.connect_worker(worker)
    
    worker.start()
    window.show()
    
    print("🚀 [System] MUSE GUI 가동 완료.")
    
    # GUI 실행 (종료될 때까지 대기)
    exit_code = app.exec()
    
    # --- 프로그램 종료 시퀀스 ---
    print("🛑 [System] 종료 시퀀스 시작...")
    
    # 1. 스레드 루프 중지 신호
    worker.stop()
    
    # 2. 스레드 종료 대기 (최대 1초)
    if not worker.wait(1000):
        print("⚠️ [System] 스레드가 반응하지 않아 강제 종료합니다.")
        worker.terminate()
    
    # 3. [Final Blow] 프로세스 강제 사살 (Kill Process)
    # sys.exit()은 파이썬 인터프리터가 정리 작업을 하느라 늦을 수 있습니다.
    # os._exit(0)은 운영체제 레벨에서 프로세스를 즉시 증발시킵니다.
    print("💀 [System] 프로세스 강제 소멸 (os._exit)")
    os._exit(0)

if __name__ == "__main__":
    main()