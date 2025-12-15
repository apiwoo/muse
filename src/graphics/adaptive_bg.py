# Project MUSE - adaptive_bg.py
# V3.8: Smart Fast-Update for Definite Backgrounds (Ghosting Killer)
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
# [New] For spatial filtering (neighbor check)
import cupyx.scipy.ndimage 
import cv2
import numpy as np
import os

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None # Float32
        
        # [Config] Learning Rates
        self.base_learning_rate = 0.005 # 기본(느린) 업데이트 속도
        self.fast_learning_rate = 0.1   # [New] 확실한 배경일 때의 빠른 속도
        
        # [Critical] Difference Penalty
        # 변화가 클수록 업데이트를 억제하는 계수 (사람 보호용)
        self.diff_penalty = 10.0 
        
        self.is_static_loaded = False

    def load_static_background(self, bg_source):
        frame_bgr = None
        if isinstance(bg_source, str):
            if os.path.exists(bg_source):
                frame_bgr = cv2.imread(bg_source)
                print(f"[BG] Loaded static background from: {bg_source}")
            else:
                print(f"[BG] Background file not found: {bg_source}")
                return False
        elif isinstance(bg_source, np.ndarray):
            frame_bgr = bg_source
        
        if frame_bgr is not None:
            if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
                frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
            self.bg_buffer = cp.asarray(frame_bgr).astype(cp.float32)
            self.is_static_loaded = True
            print("[BG] Static Background Locked. Auto-update disabled.")
            return True
        return False

    def reset(self, frame_gpu):
        if frame_gpu is None: return
        # 정적 배경이 로드되어 있다면 리셋하지 않음 (보호)
        if self.is_static_loaded: return
        
        self.bg_buffer = frame_gpu.astype(cp.float32)
        print("[BG] Background Buffer Reset (Initialized with Live Frame)")

    def update(self, frame_gpu, person_alpha):
        """
        [Corrected Update Logic V3.8]
        1. Static Lock: 캡처된 배경이 있다면 절대 업데이트하지 않음.
        2. [New] Spatial Safety Check:
           - 주변 픽셀(7x7)이 모두 배경이라면 -> "확실한 배경" -> Fast Update (잔상 제거)
           - 주변에 사람이 있다면 -> "경계 영역" -> Slow Update (스탬핑 방지)
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        # [Rule 1] Static Background Lock
        # 사용자가 직접 찍은 배경은 신성불가침 영역으로 둡니다.
        if self.is_static_loaded:
            return

        current_frame_float = frame_gpu.astype(cp.float32)

        # 1. Pixel Difference (0.0 ~ 1.0)
        # 현재 화면과 저장된 배경의 차이 계산
        diff = cp.abs(current_frame_float - self.bg_buffer)
        diff_mean = cp.mean(diff, axis=2, keepdims=True)
        diff_factor = diff_mean / 255.0

        # 2. Spatial Analysis (Neighbor Check)
        # "주변 셀이 전부 배경이면 확실히 배경이다"
        # alpha_2d: (H, W)
        alpha_2d = person_alpha
        if alpha_2d.ndim == 3: alpha_2d = alpha_2d.squeeze()
        
        # 주변 7x7 영역 내의 '최대 알파값'을 찾습니다.
        # 이 값이 0에 가깝다면, 주변 7x7 픽셀 모두가 배경이라는 뜻입니다.
        neighbor_max_alpha = cupyx.scipy.ndimage.maximum_filter(alpha_2d, size=7)
        neighbor_max_alpha = neighbor_max_alpha[..., None] # (H, W, 1)

        # [Condition] Definite Background
        # 주변에 살(Body)이 전혀 감지되지 않음 (Threshold 0.05)
        is_definite_bg = (neighbor_max_alpha < 0.05).astype(cp.float32)

        # 3. Dynamic Learning Rate Calculation
        # Case A: 확실한 배경 (is_definite_bg == 1.0)
        # -> Fast Rate 적용. Penalty 무시. (잔상 즉시 지움)
        # Case B: 불확실한 경계 (is_definite_bg == 0.0)
        # -> Slow Rate 적용. Penalty 적용. (사람 보호)
        
        # Slow Logic (Inverse Diff)
        conservative_rate = self.base_learning_rate / (1.0 + diff_factor * self.diff_penalty)
        
        # Mix Rates
        final_rate = is_definite_bg * self.fast_learning_rate + \
                     (1.0 - is_definite_bg) * conservative_rate
        
        # 4. Apply Update
        self.bg_buffer = self.bg_buffer * (1.0 - final_rate) + current_frame_float * final_rate

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)