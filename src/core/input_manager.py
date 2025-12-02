# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: Multi-Camera Support for Instant Switching + [Plan D] Threaded Triple Buffering

import cv2
import numpy as np
import time
import sys
import threading

# High-Performance GPU Library
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    # [Safety] 강제 종료 대신 예외 발생
    print("[Critical] CuPy not found. GPU acceleration unavailable.")
    HAS_CUDA = False
    # sys.exit(1) -> raise RuntimeError
    raise RuntimeError("CuPy library not found. Please run 'pip install cupy-cuda12x'.")

class CaptureWorker(threading.Thread):
    """
    [Plan D] Background Capture Thread
    - 메인 루프와 별개로 항상 최신 프레임을 가져옵니다.
    - 입력 지연(Input Lag)을 최소화하고 메인 스레드 병목을 방지합니다.
    """
    def __init__(self, caps):
        super().__init__()
        self.caps = caps # {id: cv2.VideoCapture}
        self.active_id = None
        self.latest_frame = None
        self.new_frame_available = False
        self.running = True
        self.lock = threading.Lock()
        self.daemon = True # 메인 프로그램 종료 시 자동 종료

    def set_active_camera(self, cid):
        with self.lock:
            self.active_id = cid
            self.latest_frame = None # 리셋

    def run(self):
        print("🧵 [Input] Capture Thread Started.")
        while self.running:
            # 활성 카메라가 없으면 대기
            if self.active_id is None or self.active_id not in self.caps:
                time.sleep(0.01)
                continue

            # Multi-Cam Strategy:
            # Active Camera -> read() (Decode)
            # Inactive Cameras -> grab() (Buffer flush)
            
            # 1. Grab all (Hardware Sync)
            for cid, cap in self.caps.items():
                if cid == self.active_id:
                    # Active: Read full frame
                    ret, frame = cap.read()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                else:
                    # Inactive: Just flush buffer
                    cap.grab()
            
            # 과도한 CPU 점유 방지 (Sleep removed for max performance, or very small sleep)
            # time.sleep(0.001) 

    def get_latest_frame(self):
        with self.lock:
            if self.new_frame_available and self.latest_frame is not None:
                self.new_frame_available = False
                return self.latest_frame, True
            else:
                return None, False

    def stop(self):
        self.running = False

class InputManager:
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        """
        [수정 v3.0] Threaded Capture 도입
        - camera_indices: 연결할 카메라 ID 리스트 (예: [0, 1])
        - 모든 카메라를 초기에 열어두고(Warm-up), CaptureWorker가 백그라운드에서 관리합니다.
        """
        self.caps = {}
        self.active_id = None
        self.width = width
        self.height = height
        self.fps = fps
        
        unique_ids = sorted(list(set(camera_indices)))
        print(f"📷 [InputManager] 다중 카메라 초기화: {unique_ids}")
        
        for cid in unique_ids:
            print(f"   -> Connecting to Camera {cid}...", end=" ")
            cap = cv2.VideoCapture(cid)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                
                # 워밍업
                for _ in range(5): cap.read()
                
                self.caps[cid] = cap
                print("✅ OK")
                if self.active_id is None: self.active_id = cid
            else:
                print("❌ Failed")

        if not self.caps:
            # [Safety] 강제 종료 대신 예외 발생
            raise RuntimeError("❌ 연결된 카메라가 하나도 없습니다.")

        print(f"✨ [InputManager] 활성 카메라: ID {self.active_id}")

        # [Plan D] Start Capture Thread
        self.worker = CaptureWorker(self.caps)
        self.worker.set_active_camera(self.active_id)
        self.worker.start()

    def select_camera(self, camera_id):
        """활성 카메라 변경 (Instant Switch)"""
        if camera_id in self.caps:
            if self.active_id != camera_id:
                self.active_id = camera_id
                print(f"🔄 [Input] Switched to Camera {camera_id}")
                self.worker.set_active_camera(camera_id)
            return True
        else:
            print(f"⚠️ [Input] Camera {camera_id} not available.")
            return False

    def read(self):
        """
        [Plan D] Non-blocking Read
        - 스레드가 가져온 최신 프레임을 반환합니다.
        - 대기 시간 없이 즉시 반환되므로 메인 루프가 빨라집니다.
        """
        frame_cpu, ret = self.worker.get_latest_frame()
        
        frame_gpu = None
        if ret and frame_cpu is not None:
             # BGR 유지 + GPU 업로드
             frame_gpu = cp.asarray(frame_cpu)
        
        return frame_gpu, ret

    def release(self):
        if self.worker:
            self.worker.stop()
            self.worker.join()
            
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()