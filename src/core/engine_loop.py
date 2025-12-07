# Project MUSE - engine_loop.py
# V5 Architecture: The Guided High-Res Flow (Debug Enhanced)
# (C) 2025 MUSE Corp. All rights reserved.

import time
import numpy as np
import os
import sys
import cv2

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QMutexLocker

# [MUSE Modules]
from utils.config import ProfileManager
from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.consensus_engine import ConsensusEngine
from graphics.adaptive_bg import AdaptiveBackground
from graphics.beauty_engine import BeautyEngine
from ai.tracking.facemesh import FaceMesh

try:
    import cupy as cp
except ImportError:
    cp = None

class BeautyWorker(QThread):
    frame_processed = Signal(object)
    slider_sync_requested = Signal(dict)

    def __init__(self, start_profile="default"): 
        super().__init__()
        self.running = True
        self.param_mutex = QMutex()
        
        self.profile_mgr = ProfileManager()
        self.profiles = self.profile_mgr.get_profile_list()
        
        self.current_profile_name = start_profile
        if self.current_profile_name not in self.profiles:
            self.current_profile_name = self.profiles[0] if self.profiles else "default"
        
        initial_config = self.profile_mgr.get_config(self.current_profile_name)
        self.params = initial_config.get("params", {})
        
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.pending_profile_index = -1
        self.pending_bg_capture = False
        
        print(f"[ENGINE] Initializing Face Tracker (MediaPipe)...")
        self.face_tracker = FaceMesh(self.root_dir)

    def run(self):
        print(f"[ENGINE] Launching V5 Pipeline (Profile: {self.current_profile_name})...")

        try:
            # 1. Hardware Initialization
            used_cams = set([0])
            for p in self.profiles:
                cfg = self.profile_mgr.get_config(p)
                used_cams.add(cfg.get("camera_id", 0))
            
            current_cfg = self.profile_mgr.get_config(self.current_profile_name)
            used_cams.add(current_cfg.get("camera_id", 0))
            
            print(f"[ENGINE] Initializing Cameras: {list(used_cams)}")
            self.input_mgr = InputManager(camera_indices=list(used_cams), width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            
            print(f"[ENGINE] Selecting Initial Camera ID: {current_cfg.get('camera_id', 0)}")
            self.input_mgr.select_camera(current_cfg.get("camera_id", 0))
            
            print(f"[ENGINE] Initializing Virtual Camera...")
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)

            # 2. AI & Graphics
            print(f"[ENGINE] Initializing AI Engine...")
            self.ai_engine = ConsensusEngine(self.root_dir)
            
            # [Added] Set Initial AI Strategy
            self.ai_engine.set_profile(self.current_profile_name)

            print(f"[ENGINE] Initializing Graphics (Adaptive BG)...")
            self.bg_manager = AdaptiveBackground(self.WIDTH, self.HEIGHT)
            print(f"[ENGINE] Initializing Beauty Engine...")
            self.beauty_engine = BeautyEngine(profiles=self.profiles)
            
            # [Load] Initial Assets
            print(f"[ENGINE] Loading Initial Assets...")
            self._load_profile_assets(self.current_profile_name)
            
            with QMutexLocker(self.param_mutex):
                self.slider_sync_requested.emit(self.params)
            
            print(f"[ENGINE] Initialization Complete. Starting Loop...")
            
        except Exception as e:
            print(f"[ERROR] Engine Init Failed: {e}")
            import traceback
            traceback.print_exc()
            return

        frame_count = 0
        no_frame_tick = 0
        prev_time = time.time()
        
        print("[ENGINE] >>> Entering Main Loop <<<")

        while self.running:
            # [Check Profile Switch]
            if self.pending_profile_index >= 0:
                self._execute_profile_switch(self.pending_profile_index)
                self.pending_profile_index = -1

            # Input
            t_read_start = time.perf_counter()
            frame_gpu, ret = self.input_mgr.read()
            t_read_end = time.perf_counter()
            
            if not ret or frame_gpu is None:
                no_frame_tick += 1
                if no_frame_tick % 60 == 0:
                    print(f"[WARNING] Engine Loop: No Frame (Tick: {no_frame_tick})")
                self.msleep(5) 
                continue
            
            if no_frame_tick > 0:
                print(f"[INFO] Engine Loop: Frame Signal Restored!")
                no_frame_tick = 0

            # BG Capture
            if self.pending_bg_capture:
                self._execute_bg_capture(frame_gpu)
                self.pending_bg_capture = False

            # --- Pipeline ---
            t_ai_start = time.perf_counter()
            
            # AI Engine (Handles Hybrid Strategy internally)
            alpha_matte, keypoints = self.ai_engine.process(frame_gpu)
            
            # Face Tracking (CPU)
            if hasattr(frame_gpu, 'get'):
                frame_bgr_cpu = frame_gpu.get()
            else:
                frame_bgr_cpu = frame_gpu
            
            faces = self.face_tracker.process(frame_bgr_cpu)
            t_ai_end = time.perf_counter()
            
            if alpha_matte is not None:
                self.bg_manager.update(frame_gpu, alpha_matte)
            else:
                self.bg_manager.reset(frame_gpu)

            with QMutexLocker(self.param_mutex):
                current_params = self.params.copy()

            clean_bg = self.bg_manager.get_background()
            self.beauty_engine.bg_gpu = clean_bg 
            
            t_beauty_start = time.perf_counter()
            frame_out_gpu = self.beauty_engine.process(
                frame_gpu, 
                faces=faces, 
                body_landmarks=keypoints, 
                params=current_params, 
                mask=alpha_matte
            )
            t_beauty_end = time.perf_counter()
            
            t_send_start = time.perf_counter()
            self.virtual_cam.send(frame_out_gpu)
            t_send_end = time.perf_counter()
            
            self.frame_processed.emit(frame_out_gpu)

            frame_count += 1
            if time.time() - prev_time >= 1.0:
                # print(f"[FPS] {frame_count} ...")
                frame_count = 0
                prev_time = time.time()

        print("[ENGINE] Loop Finished.")
        self.cleanup()

    def _execute_bg_capture(self, frame_gpu):
        if frame_gpu is None: return
        
        if hasattr(frame_gpu, 'get'):
            frame_bgr = frame_gpu.get()
        else:
            frame_bgr = frame_gpu
            
        profile_path = self.profile_mgr.get_profile_path(self.current_profile_name)
        save_path = os.path.join(profile_path, "background.jpg")
        
        try:
            cv2.imwrite(save_path, frame_bgr)
            print(f"[BG] Saved background to: {save_path}")
            self.bg_manager.load_static_background(frame_bgr)
        except Exception as e:
            print(f"[ERROR] Failed to save background: {e}")

    def _execute_profile_switch(self, index):
        self.profiles = self.profile_mgr.get_profile_list()
        
        target_profile_name = ""
        if index < len(self.profiles):
            target_profile_name = self.profiles[index]
        else:
            target_profile_name = f"profile_{index+1}"
            print(f"[INFO] Creating new profile '{target_profile_name}'...")
            current_cam_id = self.input_mgr.active_id if self.input_mgr.active_id is not None else 0
            self.profile_mgr.create_profile(target_profile_name, camera_id=current_cam_id)
            self.profiles = self.profile_mgr.get_profile_list()

        if target_profile_name == self.current_profile_name:
            return

        print(f"\n>>> [SWITCH] {self.current_profile_name} -> {target_profile_name}")
        
        config = self.profile_mgr.get_config(target_profile_name)
        target_cam_id = config.get("camera_id", 0)
        
        if self.input_mgr.select_camera(target_cam_id):
            print(f"   [CAM] Source: {target_cam_id}")
        
        # [Added] Update AI Strategy
        self.ai_engine.set_profile(target_profile_name)

        self._load_profile_assets(target_profile_name)
        self.current_profile_name = target_profile_name
        print("<<< [SWITCH] Done.\n")

    def _load_profile_assets(self, profile_name):
        bg_path = os.path.join(self.root_dir, "recorded_data", "personal_data", profile_name, "background.jpg")
        if os.path.exists(bg_path):
            self.bg_manager.load_static_background(bg_path)
        else:
            print(f"[WARNING] No background.jpg. Press 'B' to capture.")
            self.bg_manager.is_static_loaded = False 
        
        self.beauty_engine.set_profile(profile_name)
        config = self.profile_mgr.get_config(profile_name)
        
        with QMutexLocker(self.param_mutex):
            self.params = config.get("params", {})
            self.slider_sync_requested.emit(self.params)

    def switch_profile(self, index):
        self.pending_profile_index = index

    def reset_background(self):
        self.pending_bg_capture = True

    def update_params(self, new_params):
        with QMutexLocker(self.param_mutex):
            self.params = new_params.copy()
            self.profile_mgr.update_params(self.current_profile_name, new_params)

    def cleanup(self):
        print("[ENGINE] Cleanup")
        if hasattr(self, 'input_mgr'): self.input_mgr.release()
        if hasattr(self, 'virtual_cam'): self.virtual_cam.close()
    
    def stop(self):
        self.running = False