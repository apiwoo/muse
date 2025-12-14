# Project MUSE - adaptive_bg.py
# V3.7: Static Lock & Inverse Difference Logic (Fix Background Corruption)
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
import cv2
import numpy as np
import os

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None # Float32
        
        # [Config] Safety Parameters
        self.base_learning_rate = 0.005 
        # [Critical Change] Difference Penalty
        # 변화가 클수록 업데이트를 억제하는 계수입니다.
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
        [Corrected Update Logic]
        1. Static Lock: 캡처된 배경이 있다면 절대 업데이트하지 않음 (완전 고정).
        2. Inverse Diff: 변화가 큰 영역(사람 등)은 업데이트를 거부함.
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        # [FIX 1] Static Background Lock
        # 사용자가 직접 찍은 배경은 신성불가침 영역으로 둡니다.
        # 조명 변화 대응보다 원본 보존이 훨씬 중요합니다.
        if self.is_static_loaded:
            return

        current_frame_float = frame_gpu.astype(cp.float32)

        # 1. Pixel Difference (0.0 ~ 1.0)
        diff = cp.abs(current_frame_float - self.bg_buffer)
        diff_mean = cp.mean(diff, axis=2, keepdims=True)
        diff_factor = diff_mean / 255.0

        # 2. Background Probability
        bg_prob = 1.0 - person_alpha
        bg_prob = bg_prob[..., None]
        
        # Safe Zone: 배경 확률이 매우 높은 곳
        safe_zone_mask = (bg_prob > 0.99).astype(cp.float32)

        # [FIX 2] Inverse Difference Logic (The Ghosting Killer)
        # 이전 로직: Rate + (diff * speed) -> 변화 크면 빨리 업데이트 (최악의 오판)
        # 수정 로직: Rate / (1 + diff * penalty) -> 변화 크면 업데이트 안함 (정상)
        # 변화가 0에 가까울 때만 base_rate로 업데이트되고,
        # 사람이 난입하여 diff가 커지면 learning_rate가 0에 수렴합니다.
        dynamic_rate = self.base_learning_rate / (1.0 + diff_factor * self.diff_penalty)
        
        # 최종 가중치 적용
        final_weight = dynamic_rate * safe_zone_mask
        
        # 3. Apply Update
        self.bg_buffer = self.bg_buffer * (1.0 - final_weight) + current_frame_float * final_weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)