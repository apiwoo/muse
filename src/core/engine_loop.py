# Project MUSE - engine_loop.py
# V5 Architecture: The Guided High-Res Flow (Debug Enhanced)
# Updated: Phase 3 High-Precision LoRA Mode Support
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

    def __init__(self, start_profile="default", run_mode="STANDARD"): 
        super().__init__()
        self.running = True
        self.param_mutex = QMutex()
        
        self.profile_mgr = ProfileManager()
        self.profiles = self.profile_mgr.get_profile_list()
        
        self.current_profile_name = start_profile
        if self.current_profile_name not in self.profiles:
            self.current_profile_name = self.profiles[0] if self.profiles else "default"
        
        # [New] Runtime Mode (STANDARD, LORA, PERSONAL)
        self.run_mode = run_mode
        
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
        print(f"[ENGINE] Launching V5 Pipeline (Profile: {self.current_profile_name}, Mode: {self.run_mode})...")

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
            
            # [Added] Set Initial AI Strategy with Mode
            self.ai_engine.set_strategy(self.current_profile_name, self.run_mode)

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
        
        # [Performance Monitoring]
        acc_read = 0.0
        acc_ai = 0.0
        acc_beauty = 0.0
        acc_write = 0.0
        
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
            
            # [Added] Skeleton Drawing Logic
            final_output = frame_out_gpu
            if current_params.get('show_body_debug', False) and keypoints is not None:
                try:
                    # [Fix: Sync Required]
                    if hasattr(self.beauty_engine, 'stream'):
                        self.beauty_engine.stream.synchronize()

                    # GPU -> CPU for drawing
                    if hasattr(frame_out_gpu, 'get'):
                        debug_frame_cpu = frame_out_gpu.get()
                    else:
                        debug_frame_cpu = frame_out_gpu.copy()
                    
                    # Draw Skeleton
                    self._draw_skeleton(debug_frame_cpu, keypoints)
                    
                    # CPU -> GPU (Warning: Performance overhead)
                    if hasattr(frame_out_gpu, 'get'):
                        final_output = cp.asarray(debug_frame_cpu)
                    else:
                        final_output = debug_frame_cpu
                        
                except Exception as e:
                    print(f"[WARN] Skeleton draw failed: {e}")

            t_send_start = time.perf_counter()
            self.virtual_cam.send(final_output)
            t_send_end = time.perf_counter()
            
            self.frame_processed.emit(final_output)

            # [Metrics Calculation]
            t_read_ms = (t_read_end - t_read_start) * 1000.0
            t_ai_ms = (t_ai_end - t_ai_start) * 1000.0
            t_beauty_ms = (t_beauty_end - t_beauty_start) * 1000.0
            t_write_ms = (t_send_end - t_send_start) * 1000.0
            
            acc_read += t_read_ms
            acc_ai += t_ai_ms
            acc_beauty += t_beauty_ms
            acc_write += t_write_ms

            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                elapsed = curr_time - prev_time
                fps = frame_count / elapsed
                
                avg_read = acc_read / frame_count
                avg_ai = acc_ai / frame_count
                avg_beauty = acc_beauty / frame_count
                avg_write = acc_write / frame_count
                
                # 순수 처리 지연 시간 (Read + AI + Beauty) + Write(대기 포함)
                total_process = avg_read + avg_ai + avg_beauty
                
                print(f"[FPS: {fps:.1f}] Latency: {total_process:.1f}ms (+Write {avg_write:.1f}ms) | "
                      f"R: {avg_read:.1f}, AI: {avg_ai:.1f}, B: {avg_beauty:.1f}, W: {avg_write:.1f}")
                
                frame_count = 0
                acc_read = 0.0
                acc_ai = 0.0
                acc_beauty = 0.0
                acc_write = 0.0
                prev_time = curr_time

        print("[ENGINE] Loop Finished.")
        self.cleanup()

    def _draw_skeleton(self, img, keypoints):
        """
        Draw COCO 17 Keypoints skeleton on image.
        keypoints: (17, 3) [x, y, conf]
        """
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4), # Face
            (5, 6), (5, 7), (7, 9), # Left Arm
            (6, 8), (8, 10), # Right Arm
            (5, 11), (6, 12), (11, 12), # Body
            (11, 13), (13, 15), # Left Leg
            (12, 14), (14, 16) # Right Leg
        ]
        
        c_point = (0, 255, 0)
        c_line = (255, 255, 0)
        
        if keypoints is None: return

        # Draw Points
        for i in range(len(keypoints)):
            x, y, conf = keypoints[i]
            if conf > 0.4:
                cv2.circle(img, (int(x), int(y)), 4, c_point, -1)

        # Draw Lines
        for i, j in edges:
            if i >= len(keypoints) or j >= len(keypoints): continue
            
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            
            if c1 > 0.4 and c2 > 0.4:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c_line, 2)

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
        
        # [Added] Update AI Strategy (Preserve Current Mode)
        self.ai_engine.set_strategy(target_profile_name, self.run_mode)

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