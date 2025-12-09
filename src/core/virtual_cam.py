# Project MUSE - virtual_cam.py
# (C) 2025 MUSE Corp. All rights reserved.
# Target: RTX 3060+ (Mode A Focus)

import pyvirtualcam
import numpy as np
import sys

# High-Performance GPU Library
try:
    import cupy as cp
except ImportError:
    cp = None

class VirtualCamera:
    def __init__(self, width=1920, height=1080, fps=30):
        """
        [수정 v1.3] BGR Format & Sleep Restore
        - 포맷: RGB -> BGR 변경 (OpenCV 포맷과 통일하여 변환 비용 제거)
        - 동기화: sleep_until_next_frame() 복구 (안정적인 송출 페이싱)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

        print(f"📡 [VirtualCam] OBS 연결... ({width}x{height} @ {fps}fps, BGR)")
        
        try:
            # [PixelFormat 변경] RGB -> BGR
            # OpenCV에서 넘어오는 데이터가 BGR이므로, 여기서도 BGR로 받아야 색이 정상으로 나옴
            self.cam = pyvirtualcam.Camera(
                width=width, 
                height=height, 
                fps=fps, 
                fmt=pyvirtualcam.PixelFormat.BGR
            )
            print(f"✅ [VirtualCam] 연결 성공")
        except Exception as e:
            print(f"❌ [VirtualCam] 연결 실패: {e}")
            sys.exit(1)

    def send(self, frame):
        if self.cam is None:
            return

        # GPU -> CPU Download
        if cp is not None and isinstance(frame, cp.ndarray):
            frame_cpu = frame.get()
        else:
            frame_cpu = frame

        # 프레임 전송
        self.cam.send(frame_cpu)
        
        # [Sleep 복구]
        # 입력이 정상적인 30fps라면, 이 함수는 프레임을 깎아먹지 않고
        # 송출 타이밍을 일정하게 맞춰주는(Jitter 방지) 역할을 합니다.
        # self.cam.sleep_until_next_frame()

    def close(self):
        if self.cam:
            self.cam.close()