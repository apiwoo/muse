# Project MUSE - input_manager.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: Multi-Camera Support for Instant Switching

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
    def __init__(self, camera_indices=[0], width=1920, height=1080, fps=30):
        """
        [수정 v2.0] 멀티 카메라 지원
        - camera_indices: 연결할 카메라 ID 리스트 (예: [0, 1])
        - 모든 카메라를 초기에 열어두고(Warm-up), grab()으로 버퍼를 유지합니다.
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
            raise RuntimeError("❌ 연결된 카메라가 하나도 없습니다.")

        print(f"✨ [InputManager] 활성 카메라: ID {self.active_id}")

    def select_camera(self, camera_id):
        """활성 카메라 변경 (Instant Switch)"""
        if camera_id in self.caps:
            if self.active_id != camera_id:
                self.active_id = camera_id
                print(f"🔄 [Input] Switched to Camera {camera_id}")
                # 스위칭 직후 버퍼 플러시 (지연 방지)
                for _ in range(2):
                    self.caps[camera_id].read()
            return True
        else:
            print(f"⚠️ [Input] Camera {camera_id} not available.")
            return False

    def read(self):
        """
        [Multi-Cam Strategy]
        - Active Camera: retrieve()로 실제 데이터 디코딩
        - Inactive Cameras: grab()으로 하드웨어 버퍼만 비움 (비용 절약 + 최신 상태 유지)
        """
        frame_gpu = None
        ret_final = False

        for cid, cap in self.caps.items():
            if cid == self.active_id:
                ret, frame_cpu = cap.read()
                if ret:
                    # BGR 유지 + GPU 업로드
                    frame_gpu = cp.asarray(frame_cpu)
                    ret_final = True
            else:
                # 비활성 카메라도 계속 캡처해야 나중에 전환했을 때 딜레이가 없음
                # decode를 안하므로 CPU 부하 적음
                cap.grab()
        
        return frame_gpu, ret_final

    def release(self):
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()