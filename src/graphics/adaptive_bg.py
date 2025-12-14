# Project MUSE - adaptive_bg.py
# V3.5: Conservative & Safe Update (Anti-Ghosting Optimized)
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
import cv2
import numpy as np
import os

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None # Float32 for precision accumulation
        
        # [Config] Anti-Ghosting Parameters (Conservative Mode)
        # 배경 업데이트를 매우 보수적으로 설정하여, 움직이는 사람의 잔상이
        # 배경 버퍼에 기록되는 것을 방지합니다.
        
        self.base_learning_rate = 0.001 # 매우 느린 학습 (기존 0.005 대비 1/5)
        self.adaptation_speed = 0.02    # 급격한 변화에 둔감하게 반응
        self.max_update_rate = 0.05     # 한 번에 변할 수 있는 최대치 제한 (잔상 박제 방지)
        
        self.is_static_loaded = False

    def load_static_background(self, bg_source):
        """
        Load clean plate manually.
        """
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
            print("[BG] Static background applied to VRAM.")
            return True
            
        return False

    def reset(self, frame_gpu):
        if frame_gpu is None: return
        if self.is_static_loaded: return

        self.bg_buffer = frame_gpu.astype(cp.float32)
        print("[BG] Background Buffer Reset (Initialized with Live Frame)")

    def update(self, frame_gpu, person_alpha):
        """
        [Conservative Update Logic]
        배경이 '확실한' 영역만 아주 천천히 업데이트합니다.
        
        - 문제: 사람이 빠르게 지나갈 때 그 잔상이 배경에 남음
        - 해결: 
          1. Learning Rate 대폭 감소
          2. Safe Zone Masking (배경 확률 99% 이상인 곳만 업데이트)
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        current_frame_float = frame_gpu.astype(cp.float32)

        # 1. Pixel Difference
        diff = cp.abs(current_frame_float - self.bg_buffer)
        diff_mean = cp.mean(diff, axis=2, keepdims=True)
        diff_factor = diff_mean / 255.0

        # 2. Background Probability
        bg_prob = 1.0 - person_alpha
        bg_prob = bg_prob[..., None]
        
        # [Critical] Safe Zone Masking
        # 배경일 확률이 99% 이상인 픽셀만 업데이트 후보로 선정
        # 인물 경계선(Semi-transparent)이나 내부가 업데이트되는 것을 원천 차단
        safe_zone_mask = (bg_prob > 0.99).astype(cp.float32)

        # 3. Dynamic Rate Calculation
        boosted_rate = self.base_learning_rate + (diff_factor * self.adaptation_speed)
        boosted_rate = cp.clip(boosted_rate, 0.0, self.max_update_rate)
        
        # 최종 가중치: (계산된 속도) * (안전 구역 마스크)
        # 안전 구역이 아니면 가중치는 0이 되어 업데이트되지 않음
        final_weight = boosted_rate * safe_zone_mask
        
        # 4. Apply
        self.bg_buffer = self.bg_buffer * (1.0 - final_weight) + current_frame_float * final_weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)