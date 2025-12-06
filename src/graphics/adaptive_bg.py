# Project MUSE - adaptive_bg.py
# Self-Healing Background Buffer
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
import cv2
import numpy as np

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None # Float32 for precision accumulation
        self.learning_rate = 0.005 # Slow update for stability
        
    def reset(self, frame_gpu):
        """Force reset background to current frame"""
        if frame_gpu is None: return
        self.bg_buffer = frame_gpu.astype(cp.float32)
        print("[BG] Background Buffer Reset")

    def update(self, frame_gpu, person_alpha):
        """
        Update background only where person is NOT present.
        person_alpha: 1.0 = Person, 0.0 = Background
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None: self.reset(frame_gpu)
            return

        # Background probability (Inverse of alpha)
        bg_prob = 1.0 - person_alpha
        bg_prob = bg_prob[..., None] # Broadcast to (H, W, 1)
        
        # Dynamic Update Rule:
        # Buffer = Buffer * (1 - lr*prob) + Frame * (lr*prob)
        # 사람이 있는 곳(prob=0)은 업데이트 0, 배경인 곳(prob=1)은 lr만큼 업데이트
        weight = self.learning_rate * bg_prob
        
        self.bg_buffer = self.bg_buffer * (1.0 - weight) + frame_gpu.astype(cp.float32) * weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)

    def fill_holes(self, warped_frame, warp_field_x, warp_field_y):
        """
        Composites warped frame over the clean background plate.
        This naturally fills holes created by warping (slimming).
        """
        # Note: This logic is usually handled in the composite kernel (CUDA),
        # but here we provide the clean plate for that kernel to use.
        return self.get_background()