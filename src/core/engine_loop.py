# Project MUSE - engine_loop.py
# V5 Architecture: The Guided High-Res Flow
# (C) 2025 MUSE Corp. All rights reserved.

import time
import numpy as np
import os
import sys

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QMutexLocker

# [MUSE Modules]
from utils.config import ProfileManager
from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.consensus_engine import ConsensusEngine # [New] Tri-Core
from graphics.adaptive_bg import AdaptiveBackground # [New]
from graphics.beauty_engine import BeautyEngine

try:
    import cupy as cp
except ImportError:
    cp = None

class BeautyWorker(QThread):
    frame_processed = Signal(object)
    slider_sync_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.param_mutex = QMutex()
        
        self.profile_mgr = ProfileManager()
        self.profiles = self.profile_mgr.get_profile_list()
        
        self.current_profile_name = "default"
        if self.profiles: self.current_profile_name = self.profiles[0]
        
        initial_config = self.profile_mgr.get_config(self.current_profile_name)
        self.params = initial_config.get("params", {})
        
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30
        
        # Root Path for loading assets
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def run(self):
        print(f"[ENGINE] Launching V5 High-Fidelity Pipeline...")

        try:
            # 1. Hardware Initialization
            init_config = self.profile_mgr.get_config(self.current_profile_name)
            init_cid = init_config.get("camera_id", 0)
            
            self.input_mgr = InputManager(camera_indices=[init_cid], width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.input_mgr.select_camera(init_cid)
            
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)

            # 2. AI Core Initialization (Tri-Core)
            self.ai_engine = ConsensusEngine(self.root_dir)
            
            # 3. Graphics Core Initialization
            self.bg_manager = AdaptiveBackground(self.WIDTH, self.HEIGHT)
            self.beauty_engine = BeautyEngine(profiles=self.profiles)
            self.beauty_engine.set_profile(self.current_profile_name)
            
            with QMutexLocker(self.param_mutex):
                self.slider_sync_requested.emit(self.params)
            
        except Exception as e:
            print(f"[ERROR] Engine Init Failed: {e}")
            import traceback
            traceback.print_exc()
            return

        frame_count = 0
        prev_time = time.time()

        while self.running:
            # Input (GPU)
            frame_gpu, ret = self.input_mgr.read()
            if not ret or frame_gpu is None:
                self.msleep(1)
                continue
            
            # --- V5 Pipeline Start ---
            
            # 1. Consensus Inference (High-Res Alpha + Skeleton)
            # Returns 1080p Alpha Matte and Pose Keypoints
            alpha_matte, keypoints = self.ai_engine.process(frame_gpu)
            
            # 2. Adaptive Background Update
            # Only update background if we have a valid matte
            if alpha_matte is not None:
                self.bg_manager.update(frame_gpu, alpha_matte)
            else:
                self.bg_manager.reset(frame_gpu) # Init on first frame or failure

            # 3. Prepare Params
            with QMutexLocker(self.param_mutex):
                current_params = self.params.copy()

            # 4. Rendering (Warp + Hole Filling)
            # We pass the clean background from AdaptiveBackground to BeautyEngine
            clean_bg = self.bg_manager.get_background()
            
            # BeautyEngine's process needs update to accept external BG/Mask properly
            # For now, we inject the mask into the tracker slot for compatibility
            # In V5, BeautyEngine should composite using: Frame, Alpha, CleanBG
            
            # (Hack for compatibility with existing BeautyEngine structure)
            # We override the internal BG buffer of BeautyEngine for this frame
            self.beauty_engine.bg_gpu = clean_bg 
            
            frame_out_gpu = self.beauty_engine.process(
                frame_gpu, 
                faces=[], # Face logic temporarily skipped or needs separateFaceMesh if needed
                body_landmarks=keypoints, 
                params=current_params, 
                mask=alpha_matte # High-Res Alpha
            )
            
            # --- V5 Pipeline End ---

            # Output
            self.virtual_cam.send(frame_out_gpu)
            self.frame_processed.emit(frame_out_gpu)

            # FPS Stats
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                # print(f"[V5] FPS: {frame_count}")
                frame_count = 0
                prev_time = curr_time

        self.cleanup()

    def update_params(self, new_params):
        with QMutexLocker(self.param_mutex):
            self.params = new_params.copy()

    def reset_background(self):
        # Trigger reset on next frame in AdaptiveBG
        pass 

    def cleanup(self):
        print("[ENGINE] Cleanup")
        if hasattr(self, 'input_mgr'): self.input_mgr.release()
        if hasattr(self, 'virtual_cam'): self.virtual_cam.close()

    def switch_profile(self, index):
        pass # Implement based on original logic
    
    def stop(self):
        self.running = False