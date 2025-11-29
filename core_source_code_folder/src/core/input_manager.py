# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: RTX 3060+ (Mode A Focus)

import cv2
import numpy as np
import time
import sys

# High-Performance GPU Library
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    print("[Critical] CuPy not found. GPU acceleration unavailable.")
    HAS_CUDA = False
    sys.exit(1)

class InputManager:
    def __init__(self, device_id=1, width=1920, height=1080, fps=30):
        """
        [수정 v1.3] Backend Rollback Version
        - 원인 파악: CAP_DSHOW 강제 설정이 C920을 YUY2(5fps) 모드로 빠뜨림.
        - 해결책: cv2.CAP_ANY (Default/MSMF)로 복귀하여 30fps 확보.
        - 최적화: 불필요한 색상 변환 제거 (BGR 유지)
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        
        print(f"📷 [InputManager] 카메라 초기화 (ID: {device_id}, Default Backend)...")
        
        # 1. 백엔드 설정 제거 (기본값 사용)
        # 기존 camera.py와 동일하게 설정하여 MSMF가 자동 최적화하도록 함
        self.cap = cv2.VideoCapture(device_id) 
        
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 카메라(ID:{device_id})를 열 수 없습니다.")

        # 2. 해상도 및 FPS 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 실제 설정 확인
        real_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ [InputManager] 설정 결과: {real_w}x{real_h} @ {real_fps}fps")
        
        # 워밍업
        for _ in range(5):
            self.cap.read()

    def read(self):
        ret, frame_cpu = self.cap.read()
        
        if not ret:
            return None, False

        # [Pipeline Stage 1] Host(CPU) -> Device(GPU) Upload
        # 프레임이 성공적으로 읽혔을 때만 업로드
        frame_gpu = cp.asarray(frame_cpu)

        # [Optimization] BGR 유지
        # OpenCV는 BGR을 줍니다. 가상 카메라도 BGR로 설정하면 변환 비용 0.
        # 따라서 RGB 변환 코드를 제거하고 원본 그대로 반환합니다.
        
        return frame_gpu, True

    def release(self):
        if self.cap:
            self.cap.release()