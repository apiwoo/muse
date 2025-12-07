# Project MUSE - engine_loop.py
# V5 Architecture: The Guided High-Res Flow (Debug Enhanced)
# (C) 2025 MUSE Corp. All rights reserved.
# [Debug] Added extensive logging to trace frame flow from InputManager to UI

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

# [New] Face Model Connection
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
        
        # [New] Face Tracker Initialization
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
            
            # Select Initial Camera
            print(f"[ENGINE] Selecting Initial Camera ID: {current_cfg.get('camera_id', 0)}")
            self.input_mgr.select_camera(current_cfg.get("camera_id", 0))
            
            print(f"[ENGINE] Initializing Virtual Camera...")
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)

            # 2. AI & Graphics
            print(f"[ENGINE] Initializing AI Engine...")
            self.ai_engine = ConsensusEngine(self.root_dir)
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
            # Input (GPU)
            # [Debug] Trace InputManager read
            t_read_start = time.perf_counter()
            frame_gpu, ret = self.input_mgr.read()
            t_read_end = time.perf_counter()
            
            if not ret or frame_gpu is None:
                # [DEBUG LOG] 데이터 수신 실패 모니터링
                no_frame_tick += 1
                if no_frame_tick % 60 == 0: # 약 2초마다 경고 (너무 자주 뜨지 않게)
                    print(f"[WARNING] Engine Loop: No Frame from InputManager (Tick: {no_frame_tick})")
                
                self.msleep(5) 
                continue
            
            # [Debug] Check received frame properties once per second
            if frame_count % 30 == 0:
                frame_type = type(frame_gpu)
                frame_shape = frame_gpu.shape if hasattr(frame_gpu, 'shape') else 'Unknown'
                # print(f"[DEBUG] Frame Received: Type={frame_type}, Shape={frame_shape}, ReadTime={(t_read_end-t_read_start)*1000:.2f}ms")

            # 프레임 수신 성공 시 카운터 리셋 및 복구 로그
            if no_frame_tick > 0:
                print(f"[INFO] Engine Loop: Frame Signal Restored!")
                no_frame_tick = 0

            # [Event] BG Capture
            if self.pending_bg_capture:
                print(f"[DEBUG] BG Capture Triggered")
                self._execute_bg_capture(frame_gpu)
                self.pending_bg_capture = False

            # --- Pipeline ---
            t_ai_start = time.perf_counter()
            alpha_matte, keypoints = self.ai_engine.process(frame_gpu)
            
            # [New] Face Tracking (CPU)
            # MediaPipe는 CPU 이미지를 요구하므로 변환
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
            # [Fix] Pass detected 'faces' instead of empty list
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
            
            # [Debug] Emit Signal Trace
            # print(f"[DEBUG] Emitting frame_processed signal...")
            self.frame_processed.emit(frame_out_gpu)

            frame_count += 1
            if time.time() - prev_time >= 1.0:
                print(f"[FPS] {frame_count} | Read: {(t_read_end - t_read_start)*1000:.1f}ms | AI: {(t_ai_end - t_ai_start)*1000:.1f}ms | Beauty: {(t_beauty_end - t_beauty_start)*1000:.1f}ms | Send: {(t_send_end - t_send_start)*1000:.1f}ms")
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
            print(f"[INFO] Profile '{target_profile_name}' does not exist. Creating...")
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
        else:
            print(f"   [WARNING] Camera {target_cam_id} not ready.")

        self._load_profile_assets(target_profile_name)
        self.current_profile_name = target_profile_name
        print("<<< [SWITCH] Done.\n")

    def _load_profile_assets(self, profile_name):
        bg_path = os.path.join(self.root_dir, "recorded_data", "personal_data", profile_name, "background.jpg")
        if os.path.exists(bg_path):
            if self.bg_manager.load_static_background(bg_path):
                # print(f"[INFO] Static background loaded for {profile_name}")
                pass
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