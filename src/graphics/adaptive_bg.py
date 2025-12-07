# Project MUSE - adaptive_bg.py
# Self-Healing Background Buffer (V3: Reliability-Based Adaptive Update)
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
        self.base_learning_rate = 0.005 # 기본 학습률 (안정성 위주)
        self.adaptation_speed = 0.15    # 신뢰도 높은 영역의 변화 적응 가속도
        self.max_update_rate = 0.3      # 프레임당 최대 업데이트 비율 (플리커 방지 클램핑)
        
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
        
        Algorithm:
        1. Calculate Difference: |CurrentFrame - StoredBG|
        2. Calculate Background Confidence: (1.0 - person_alpha)
        3. Dynamic Rate = BaseRate + (Diff * Confidence * Speed)
        
        Result:
        - High Confidence (Sure BG) + High Diff (Lighting Change) -> Fast Update
        - Low Confidence (Person Boundary) -> Slow/No Update
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        # Ensure frame is float32 for calculation
        current_frame_float = frame_gpu.astype(cp.float32)

        # 1. Calculate Pixel Difference (Change Magnitude)
        # RGB 채널 평균 차이를 구하여 조명 변화 등의 강도를 측정합니다.
        diff = cp.abs(current_frame_float - self.bg_buffer)
        diff_mean = cp.mean(diff, axis=2, keepdims=True) # (H, W, 1)
        
        # Normalize diff to 0.0 ~ 1.0 (assuming pixel range 0-255)
        diff_factor = diff_mean / 255.0

        # 2. Background Probability (Reliability)
        # person_alpha: 1.0 = Person, 0.0 = Background
        # bg_prob: 1.0 = Definite Background (High Reliability)
        bg_prob = 1.0 - person_alpha
        bg_prob = bg_prob[..., None] # Broadcast to (H, W, 1)
        
        # 3. Calculate Dynamic Update Rate
        # 기본 학습률에 '변화량'과 '신뢰도'를 곱한 값을 더합니다.
        # 변화가 크고 배경이 확실할수록 업데이트 속도가 빨라집니다.
        boosted_rate = self.base_learning_rate + (diff_factor * self.adaptation_speed)
        
        # 최대 업데이트 속도 제한 (급격한 플리커링 방지)
        boosted_rate = cp.clip(boosted_rate, 0.0, self.max_update_rate)
        
        # 최종 적용 가중치: 배경 확률이 높은 곳에만 부스팅된 속도 적용
        # 배경이 아닌 곳(bg_prob ~ 0)은 업데이트 되지 않습니다.
        final_weight = boosted_rate * bg_prob
        
        # 4. Apply Update (Exponential Moving Average)
        # Buffer = Buffer * (1 - weight) + Frame * weight
        self.bg_buffer = self.bg_buffer * (1.0 - final_weight) + current_frame_float * final_weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)