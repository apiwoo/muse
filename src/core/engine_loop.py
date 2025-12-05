# Project MUSE - engine_loop.py
# Orchestrator of the Zero-Copy Pipeline
# (C) 2025 MUSE Corp. All rights reserved.

import time
import numpy as np
import cv2

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QMutexLocker

# [MUSE Modules]
from utils.config import ProfileManager
from core.input_manager import InputManager
from core.virtual_cam import VirtualCamera
from ai.tracking.facemesh import FaceMesh
from ai.tracking.body_tracker import BodyTracker 
from graphics.beauty_engine import BeautyEngine

try:
    import cupy as cp
except ImportError:
    cp = None

class BeautyWorker(QThread):
    """
    [Background Thread]
    Fully GPU-Resident Pipeline
    Input(GPU) -> AI(CPU/GPU Hybrid) -> Render(GPU) -> Output(GPU/GL)
    """
    frame_processed = Signal(object) # Changed to object to carry CuPy array
    slider_sync_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.should_reset_bg = False
        self.param_mutex = QMutex()
        
        self.profile_mgr = ProfileManager()
        self.profiles = self.profile_mgr.get_profile_list()
        
        self.current_profile_name = "default"
        if self.profiles:
            self.current_profile_name = self.profiles[0]
        
        initial_config = self.profile_mgr.get_config(self.current_profile_name)
        self.params = initial_config.get("params", {})
        
        self.WIDTH = 1920
        self.HEIGHT = 1080
        self.FPS = 30
        
        # Pinned Memory for Fast Download (GPU -> CPU for AI)
        self.pinned_mem_ai = None

    def run(self):
        print(f"[THREAD] [Worker] Zero-Copy Engine Start (Active: {self.current_profile_name})")

        try:
            required_cams = []
            for p in self.profiles:
                cfg = self.profile_mgr.get_config(p)
                required_cams.append(cfg.get("camera_id", 0))
            
            self.input_mgr = InputManager(camera_indices=required_cams, width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            
            init_cid = self.profile_mgr.get_config(self.current_profile_name).get("camera_id", 0)
            self.input_mgr.select_camera(init_cid)

            self.tracker = FaceMesh(root_dir="assets")
            self.body_tracker = BodyTracker(profiles=self.profiles)
            self.beauty_engine = BeautyEngine(profiles=self.profiles)
            self.virtual_cam = VirtualCamera(width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)

            self.body_tracker.set_profile(self.current_profile_name)
            self.beauty_engine.set_profile(self.current_profile_name)
            
            with QMutexLocker(self.param_mutex):
                self.slider_sync_requested.emit(self.params)
            
        except Exception as e:
            print(f"[ERROR] [Worker] Init Failed: {e}")
            return

        frame_count = 0
        prev_time = time.time()

        while self.running:
            # [Step 1] Input (GPU Resident)
            # frame_gpu is CuPy array
            frame_gpu, ret = self.input_mgr.read()
            
            if not ret or frame_gpu is None:
                self.msleep(1)
                continue
            
            # [Event] Background Reset
            if self.should_reset_bg:
                self.beauty_engine.reset_background(frame_gpu)
                self.should_reset_bg = False

            # [Step 2] AI Processing (Hybrid)
            # AI needs CPU data (currently). Use fast download.
            frame_cpu_ai = self._download_for_ai(frame_gpu)
            
            faces = self.tracker.process(frame_cpu_ai) if self.tracker else []
            body_landmarks = self.body_tracker.process(frame_cpu_ai) if self.body_tracker else None

            current_params = {}
            with QMutexLocker(self.param_mutex):
                current_params = self.params.copy()

            # [Step 3] Rendering (GPU)
            # All params are passed to BeautyEngine which uses CuPy Kernels
            if self.beauty_engine:
                frame_out_gpu = self.beauty_engine.process(
                    frame_gpu, faces, body_landmarks, current_params, 
                    mask=self.body_tracker.get_mask()
                )
            else:
                frame_out_gpu = frame_gpu

            # [Step 4] Output
            # OBS (Virtual Cam) handles GPU-CPU internally or we pass CPU if needed
            # For max performance, VirtualCam should ideally accept GPU, but standard pyvirtualcam needs CPU.
            # We do it here or inside virtual_cam.send
            if self.virtual_cam:
                self.virtual_cam.send(frame_out_gpu)

            # UI Display
            # Emit GPU handle directly. GLWidget will handle PBO upload.
            self.frame_processed.emit(frame_out_gpu)

            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                # print(f"[FAST] FPS: {frame_count}")
                frame_count = 0
                prev_time = curr_time

        self.cleanup()

    def _download_for_ai(self, frame_gpu):
        """
        Fast GPU -> CPU download using Pinned Memory
        """
        if not cp or not hasattr(frame_gpu, 'device'):
            return frame_gpu # Already CPU

        h, w, c = frame_gpu.shape
        nbytes = frame_gpu.nbytes
        
        if self.pinned_mem_ai is None or self.pinned_mem_ai.nbytes != nbytes:
            self.pinned_mem_ai = cp.cuda.alloc_pinned_memory(nbytes)
            
        frame_cpu = np.frombuffer(self.pinned_mem_ai, frame_gpu.dtype, frame_gpu.size).reshape((h, w, c))
        frame_gpu.get(out=frame_cpu)
        return frame_cpu

    def cleanup(self):
        print("[CLEAN] [Worker] Cleanup")
        self.save_current_config()
        if self.input_mgr: self.input_mgr.release()
        if self.virtual_cam: self.virtual_cam.close()
        # Free Pinned Memory
        self.pinned_mem_ai = None

    def save_current_config(self):
        with QMutexLocker(self.param_mutex):
            self.profile_mgr.update_params(self.current_profile_name, self.params)

    @Slot(dict)
    def update_params(self, new_params):
        with QMutexLocker(self.param_mutex):
            self.params = new_params.copy()

    @Slot()
    def reset_background(self):
        self.should_reset_bg = True

    @Slot(int)
    def switch_profile(self, index):
        if index < 0 or index >= len(self.profiles): return
        target_profile = self.profiles[index]
        if target_profile == self.current_profile_name: return

        print(f"\n[LOOP] [Switch] -> {target_profile}")
        self.save_current_config()
        self.current_profile_name = target_profile
        new_config = self.profile_mgr.get_config(target_profile)
        
        with QMutexLocker(self.param_mutex):
            self.params = new_config.get("params", {}).copy()
        
        target_cam_id = new_config.get("camera_id", 0)
        self.input_mgr.select_camera(target_cam_id)
        self.body_tracker.set_profile(target_profile)
        self.beauty_engine.set_profile(target_profile)
        
        self.slider_sync_requested.emit(self.params)

    def stop(self):
        self.running = False