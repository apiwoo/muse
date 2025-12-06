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
        self.pinned_mem_size = 0  # [Fix] 메모리 크기 추적용 변수 추가

        # Profile Switching Request
        self.pending_profile_request = None

    def run(self):
        print(f"[THREAD] [Worker] Zero-Copy Engine Start (Active: {self.current_profile_name})")

        try:
            # [Fix 1] 카메라 찔러보기 방지: 현재 프로필의 카메라만 단독으로 초기화
            init_config = self.profile_mgr.get_config(self.current_profile_name)
            init_cid = init_config.get("camera_id", 0)
            
            print(f"[CAM] Initializing ONLY active source: {init_cid}")
            # 리스트에 현재 카메라 ID 하나만 전달하여 불필요한 장치 접근 차단
            self.input_mgr = InputManager(camera_indices=[init_cid], width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
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
            # [Step 0] Handle Profile Switch Request (In-Loop Safety)
            if self.pending_profile_request is not None:
                self._handle_profile_switch(self.pending_profile_request)
                self.pending_profile_request = None

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
            if self.virtual_cam:
                self.virtual_cam.send(frame_out_gpu)

            # UI Display
            # Emit GPU handle directly. Viewport needs to handle conversion.
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
        [Fix 2] PinnedMemoryPointer 속성 오류 해결 (nbytes -> self.pinned_mem_size)
        """
        if not cp or not hasattr(frame_gpu, 'device'):
            return frame_gpu # Already CPU

        h, w, c = frame_gpu.shape
        nbytes = frame_gpu.nbytes
        
        # [Fix] 객체 속성 대신 별도 변수로 크기 비교
        if self.pinned_mem_ai is None or self.pinned_mem_size != nbytes:
            self.pinned_mem_ai = cp.cuda.alloc_pinned_memory(nbytes)
            self.pinned_mem_size = nbytes
            
        frame_cpu = np.frombuffer(self.pinned_mem_ai, frame_gpu.dtype, frame_gpu.size).reshape((h, w, c))
        frame_gpu.get(out=frame_cpu)
        return frame_cpu

    def _handle_profile_switch(self, target_profile):
        """
        메인 루프 내에서 안전하게 실행되는 프로파일 전환 로직
        """
        print(f"\n[LOOP] [Switch] Processing switch to -> {target_profile}")
        
        # 1. 기존 설정 저장
        self.save_current_config()
        
        # 2. 새 설정 로드
        self.current_profile_name = target_profile
        new_config = self.profile_mgr.get_config(target_profile)
        
        with QMutexLocker(self.param_mutex):
            self.params = new_config.get("params", {}).copy()
        
        # 3. 카메라 전환 (필요한 경우에만 재연결)
        target_cam_id = new_config.get("camera_id", 0)
        
        print(f"[CAM] Switching source to Camera {target_cam_id}...")
        
        # 기존 카메라 해제 후 새 카메라만 연결 (단독 연결 유지)
        if self.input_mgr:
            self.input_mgr.release()
        
        try:
            self.input_mgr = InputManager(camera_indices=[target_cam_id], width=self.WIDTH, height=self.HEIGHT, fps=self.FPS)
            self.input_mgr.select_camera(target_cam_id)
        except Exception as e:
            print(f"[ERROR] Camera switch failed: {e}")

        # 4. 엔진 상태 업데이트
        self.body_tracker.set_profile(target_profile)
        self.beauty_engine.set_profile(target_profile)
        
        # 5. UI 싱크
        self.slider_sync_requested.emit(self.params)

    def cleanup(self):
        print("[CLEAN] [Worker] Cleanup")
        self.save_current_config()
        if hasattr(self, 'input_mgr') and self.input_mgr: self.input_mgr.release()
        if hasattr(self, 'virtual_cam') and self.virtual_cam: self.virtual_cam.close()
        # Free Pinned Memory
        self.pinned_mem_ai = None
        self.pinned_mem_size = 0

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
        """
        UI 스레드에서 호출됨. 실제 전환은 run() 루프에서 처리하도록 요청만 남김.
        """
        if index < 0 or index >= len(self.profiles): return
        target_profile = self.profiles[index]
        if target_profile == self.current_profile_name: return

        # 요청 대기열에 등록 (Loop에서 처리)
        self.pending_profile_request = target_profile

    def stop(self):
        self.running = False