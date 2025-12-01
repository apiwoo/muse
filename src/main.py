# Project MUSE - main.py
# The Visual Singularity Engine Entry Point (GUI Version)
# (C) 2025 MUSE Corp. All rights reserved.

import sys
import time
import cv2
import numpy as np
import os
import signal

# [PySide6 GUI Framework]
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, Slot, Qt
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
        
        # [V4.0] 파라미터 초기화 (Body Quartet)
        self.params = {
            'eye_scale': 0.0, 
            'face_v': 0.0,
            'head_scale': 0.0,
            'shoulder_narrow': 0.0, # [1]
            'ribcage_slim': 0.0,    # [2]
            'waist_slim': 0.0,      # [3]
            'hip_widen': 0.0,       # [4]
            'show_body_debug': False
        }
        
        # 자원 핸들 초기화
        self.input_mgr = None
        self.virtual_cam = None
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None
        
        # 설정
        self.DEVICE_ID = 0
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
            # [Step 1] Input (GPU Array Returned)
            if self.input_mgr:
                frame_gpu, ret = self.input_mgr.read()
            else:
                break
                
            if not ret:
                self.msleep(10)
                continue

            # [Step 2] AI Processing Prep (Copy for Inference)
            if cp and hasattr(frame_gpu, 'get'):
                frame_cpu_for_ai = frame_gpu.get()
            else:
                frame_cpu_for_ai = frame_gpu
            
            # 얼굴 트래킹 (MediaPipe)
            faces = []
            if self.tracker:
                faces = self.tracker.process(frame_cpu_for_ai)
            
            # 바디 트래킹 (ViTPose)
            body_landmarks = None
            if self.body_tracker:
                body_landmarks = self.body_tracker.process(frame_cpu_for_ai)

            # [Step 3] Beauty Processing (GPU to GPU)
            if self.beauty_engine:
                frame_out_gpu = self.beauty_engine.process(frame_gpu, faces, body_landmarks, self.params)
            else:
                frame_out_gpu = frame_gpu

            # [Step 4] Output & Debug
            if self.virtual_cam:
                self.virtual_cam.send(frame_out_gpu)

            if hasattr(frame_out_gpu, 'get'):
                frame_out_cpu = frame_out_gpu.get()
            else:
                frame_out_cpu = frame_out_gpu

            if self.params.get('show_body_debug', False) and self.body_tracker:
                frame_out_cpu = self.body_tracker.draw_debug(frame_out_cpu, body_landmarks)
            
            self.frame_processed.emit(frame_out_cpu)

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
        """자원 강제 해제 (순서 중요)"""
        print("🧹 [Worker] 자원 정리 시작...")
        
        # 1. AI 엔진 및 GPU 메모리 해제
        self.tracker = None
        self.body_tracker = None
        self.beauty_engine = None
        
        if cp:
            try:
                # [중요] GPU 메모리 풀 강제 초기화 (드라이버 좀비 방지)
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                print("   -> GPU 메모리 풀(VRAM) 초기화 완료")
            except Exception as e:
                print(f"   ⚠️ GPU 메모리 해제 경고: {e}")

        # 2. 카메라 해제 (하드웨어 점유 해제)
        if self.input_mgr:
            try:
                self.input_mgr.release()
                print("   -> 카메라(Input) 연결 해제 완료")
            except: pass
            self.input_mgr = None
        
        # 3. 가상 카메라 해제
        if self.virtual_cam:
            try:
                self.virtual_cam.close()
                print("   -> 가상 카메라(Output) 연결 해제 완료")
            except: pass
            self.virtual_cam = None
            
        print("✨ [Worker] 모든 자원 정리 완료")

    @Slot(dict)
    def update_params(self, new_params):
        """UI 슬라이더 변경 시 호출되는 슬롯"""
        self.params = new_params.copy()

    def stop(self):
        self.running = False

def main():
    def signal_handler(sig, frame):
        print(f"\n🛑 [System] 종료 시그널 감지 ({sig}). 앱 종료를 요청합니다...")
        if QApplication.instance():
            QApplication.instance().quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark")
    
    app.setQuitOnLastWindowClosed(True)

    window = MainWindow()
    worker = BeautyWorker()
    worker.setTerminationEnabled(True) 
    
    # UI 연결
    window.connect_worker(worker)
    
    worker.start()
    window.show()
    
    print("🚀 [System] MUSE GUI 가동 완료.")
    
    exit_code = app.exec()
    
    print("🛑 [System] 메인 루프 종료. 안전 종료 절차 시작...")
    
    if worker.isRunning():
        print("   -> 워커 스레드 정지 신호 전송...")
        worker.stop()
        
        # [수정] 대기 시간 3초로 연장 (GPU 작업 마무리 시간 보장)
        if not worker.wait(3000):
            print("⚠️ [System] 스레드가 3초 내에 응답하지 않습니다. 강제 종료(Terminate)합니다.")
            print("   (주의: 이 과정에서 드라이버가 불안정해질 수 있습니다)")
            worker.terminate()
            worker.wait(1000)
        else:
            print("   -> 워커 스레드 정상 종료됨.")
    
    print("💀 [System] 프로세스 완전 종료 (os._exit)")
    # [중요] 모든 파이썬 스레드와 리소스를 강제로 끊고 나감
    os._exit(0)

if __name__ == "__main__":
    main()