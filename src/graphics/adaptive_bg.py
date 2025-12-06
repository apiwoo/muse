# Project MUSE - adaptive_bg.py
# Self-Healing Background Buffer (V2: Static Load Support)
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
        self.learning_rate = 0.005 # Slow update for stability
        self.is_static_loaded = False # [New] Flag to prevent overwriting static BG

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
        weight = self.learning_rate * bg_prob
        
        self.bg_buffer = self.bg_buffer * (1.0 - weight) + frame_gpu.astype(cp.float32) * weight

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)