# Project MUSE - adaptive_bg.py
# Self-Healing Background Buffer (V3.5: Dynamic Noise Suppression)
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
        
        # [Config] Adaptive Parameters
        self.base_learning_rate = 0.002 # 기본 학습률 (매우 낮춰서 노이즈 억제)
        self.adaptation_speed = 0.5     # 반응성 계수
        self.max_update_rate = 0.4      # 최대 업데이트 제한
        
        self.is_static_loaded = False   # Flag to prevent overwriting static BG

    def load_static_background(self, bg_source):
        """
        [New] Load background from file path or numpy array directly to GPU.
        This fixes the 'Floating Ghost' issue by providing a clean plate.
        """
        frame_bgr = None
        
        # 1. Load from Path
        if isinstance(bg_source, str):
            if os.path.exists(bg_source):
                frame_bgr = cv2.imread(bg_source)
                print(f"[BG] Loaded static background from: {bg_source}")
            else:
                print(f"[BG] Background file not found: {bg_source}")
                return False
        
        # 2. Load from Array
        elif isinstance(bg_source, np.ndarray):
            frame_bgr = bg_source
        
        if frame_bgr is not None:
            # Resize if needed
            if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
                frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
            
            # Upload to GPU
            self.bg_buffer = cp.asarray(frame_bgr).astype(cp.float32)
            self.is_static_loaded = True
            print("[BG] Static background applied to VRAM.")
            return True
            
        return False

    def reset(self, frame_gpu):
        """
        Force reset background to current frame.
        Only used if no static background is available.
        """
        if frame_gpu is None: return
        
        # [Safety] Don't overwrite if we successfully loaded a clean plate file
        if self.is_static_loaded:
            return

        self.bg_buffer = frame_gpu.astype(cp.float32)
        print("[BG] Background Buffer Reset (Initialized with Live Frame)")

    def update(self, frame_gpu, person_alpha):
        """
        Update background based on Reliability and Change Detection.
        Improved: Non-linear response (Square of Diff) to suppress noise.
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        # Ensure frame is float32 for calculation
        current_frame_float = frame_gpu.astype(cp.float32)

        # 1. Calculate Pixel Difference (Change Magnitude)
        diff = cp.abs(current_frame_float - self.bg_buffer)
        diff_mean = cp.mean(diff, axis=2, keepdims=True) # (H, W, 1)
        
        # Normalize diff to 0.0 ~ 1.0
        diff_norm = diff_mean / 255.0

        # 2. Non-linear Dynamic Factor
        # 제곱을 사용하여 작은 변화(노이즈)는 무시하고(0에 수렴),
        # 큰 변화(조명)는 증폭시킵니다.
        # 예: Diff 0.05 -> Factor 0.0025 (무시)
        # 예: Diff 0.30 -> Factor 0.0900 (반영)
        dynamic_factor = diff_norm ** 2

        # 3. Background Probability (Reliability)
        # person_alpha: 1.0 = Person, 0.0 = Background
        bg_prob = 1.0 - person_alpha
        bg_prob = bg_prob[..., None] # Broadcast
        
        # 4. Calculate Dynamic Update Rate
        # 기본 속도 + (동적 팩터 * 반응 속도)
        boosted_rate = self.base_learning_rate + (dynamic_factor * self.adaptation_speed * 5.0)
        
        # 최대 업데이트 속도 제한
        boosted_rate = cp.clip(boosted_rate, 0.0, self.max_update_rate)
        
        # 최종 적용 가중치: 배경 확률이 높은 곳에만 부스팅된 속도 적용
        final_weight = boosted_rate * bg_prob
        
        # 5. Apply Update (Exponential Moving Average)
        self.bg_buffer = self.bg_buffer * (1.0 - final_weight) + current_frame_float * final_weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)