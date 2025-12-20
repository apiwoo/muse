# Project MUSE - engine_loop.py
# V5 Architecture: The Guided High-Res Flow (Debug Enhanced)
# Updated: Phase 3 High-Precision LoRA Mode Support
# Updated: Simplified Debug Skeleton (Torso Only)
# Updated: Verbose Debug Logging for Pipeline Analysis
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

            # [V39] Initialize AI Skin Parser for hybrid masking
            print(f"[ENGINE] Initializing Skin Parser...")
            self.beauty_engine._init_skin_parser()

            # [V35] AI 마스크 지연 프레임 수 설정 (잔상 제거의 핵심)
            # - 일반적으로 1~2프레임 (GPU 성능에 따라 조절)
            # - 지연이 클수록 더 과거 프레임과 매칭하여 잔상 제거
            # - 너무 크면 반응이 느려지므로 1~2 권장
            self.beauty_engine.ai_latency_frames = 1
            
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
            loop_start_ts = time.perf_counter()

            # [Check Profile Switch]
            if self.pending_profile_index >= 0:
                self._execute_profile_switch(self.pending_profile_index)
                self.pending_profile_index = -1

            # Input
            # [V44] 최신 프레임만 획득 (버퍼 플러시)
            t_read_start = time.perf_counter()
            frame_gpu, ret = self.input_mgr.read_latest()
            t_read_end = time.perf_counter()

            # [V44 LEGACY] 기존 read() 호출 (롤백용)
            # frame_gpu, ret = self.input_mgr.read()
            
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

            # [V6] 배경 안정성 확인 및 슬리밍 사용 여부 체크
            bg_is_stable = self.bg_manager.is_background_stable()
            any_slimming = (
                current_params.get('waist_slim', 0.0) > 0.0 or
                current_params.get('shoulder_narrow', 0.0) > 0.0 or
                current_params.get('ribcage_slim', 0.0) > 0.0 or
                current_params.get('hip_widen', 0.0) > 0.0
            )

            t_beauty_start = time.perf_counter()
            # [DEBUG] Beauty Engine Inputs Logging (Throttled)
            if frame_count % 30 == 0:
                skin_val = current_params.get('skin_smooth', 0.0)
                face_detected = "YES" if faces else "NO"
                kp_detected = "YES" if keypoints is not None else "NO"
                print(f"[DEBUG-B] Param(Skin): {skin_val:.2f}, Face: {face_detected}, Body: {kp_detected}")

            frame_out_gpu = self.beauty_engine.process(
                frame_gpu,
                faces=faces,
                body_landmarks=keypoints,
                params=current_params,
                mask=alpha_matte,
                frame_cpu=frame_bgr_cpu,  # Pass CPU frame to avoid GPU->CPU transfer
                bg_stable=bg_is_stable    # [V6] 배경 안정성 플래그 전달
            )
            t_beauty_end = time.perf_counter()
            
            # [Added] Skeleton Drawing Logic
            final_output = frame_out_gpu

            # [V6] 배경 미캡처 + 슬리밍 활성화 시 경고 표시
            if not bg_is_stable and any_slimming:
                try:
                    if hasattr(final_output, 'get'):
                        warn_frame = final_output.get()
                    else:
                        warn_frame = final_output.copy()

                    # 빨간색 경고 메시지 오버레이
                    warning_text = "Press 'B' to capture background for slimming"
                    cv2.putText(warn_frame, warning_text, (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                    if hasattr(final_output, 'get'):
                        final_output = cp.asarray(warn_frame)
                    else:
                        final_output = warn_frame
                except Exception as e:
                    if frame_count % 60 == 0:
                        print(f"[WARN] Warning overlay failed: {e}")

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
                    
                    # Draw Skeleton (Torso Only)
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

            # [DEBUG] Latency Warning
            loop_end_ts = time.perf_counter()
            total_loop_ms = (loop_end_ts - loop_start_ts) * 1000.0
            if total_loop_ms > 100.0 and frame_count % 30 == 0:
                 print(f"[WARN-LAG] High Latency Detected: {total_loop_ms:.1f}ms (Beauty: {t_beauty_ms:.1f}ms)")

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
        Draw COCO 17 Keypoints (Simplified: Torso Only)
        keypoints: (17, 3) [x, y, conf]
        """
        # [Modified] Only draw Shoulder (5,6) and Hip (11,12) connections
        edges = [
            (5, 6),   # Shoulders
            (11, 12), # Hips
            (5, 11),  # Left Torso
            (6, 12)   # Right Torso
        ]
        
        # Only these points will be drawn
        target_indices = [5, 6, 11, 12]
        
        c_point = (0, 255, 0)
        c_line = (255, 255, 0)
        
        if keypoints is None: return

        # Draw Points
        for i in target_indices:
            if i >= len(keypoints): continue
            x, y, conf = keypoints[i]
            if conf > 0.4:
                cv2.circle(img, (int(x), int(y)), 6, c_point, -1)

        # Draw Lines
        for i, j in edges:
            if i >= len(keypoints) or j >= len(keypoints): continue
            
            x1, y1, c1 = keypoints[i]
            x2, y2, c2 = keypoints[j]
            
            if c1 > 0.4 and c2 > 0.4:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c_line, 3)

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