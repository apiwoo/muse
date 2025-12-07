# Project MUSE - beauty_engine.py
# V16.0: Dynamic Smoothing & Bokeh Background Logic
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import glob
from ai.tracking.facemesh import FaceMesh

# Import Kernels & Logic
from graphics.kernels.cuda_kernels import WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE
from graphics.processors.morph_logic import MorphLogic

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[WARNING] [BeautyEngine] CuPy not found. Fallback to CPU Mode.")

class BeautyEngine:
    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V16.0 Dynamic Smoothing Ready")
        self.map_scale = 0.25 
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        self.prev_gpu_dx = None
        self.prev_gpu_dy = None
        
        self.bg_buffers = {}
        self.active_profile = 'default'
        self.bg_gpu = None
        self.bg_blur_gpu = None # [New] Blurred Background for Hole Filling
        self.has_bg = False
        
        self.morph_logic = MorphLogic()
        self.current_alpha = 0.85
        
        # [New] Temporal Smoothing Variables
        self.gpu_mask_prev = None
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")
        
        if HAS_CUDA:
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self._load_all_backgrounds(profiles)

    def _load_all_backgrounds(self, profiles):
        if not HAS_CUDA: return
        for p in profiles:
            bg_path = os.path.join(self.data_dir, p, "background.jpg")
            if os.path.exists(bg_path):
                img = cv2.imread(bg_path)
                if img is not None:
                    self.bg_buffers[p] = {'cpu': img, 'gpu': None}
                    print(f"[BEAUTY] Loaded background for profile: {p}")
            else:
                self.bg_buffers[p] = {'cpu': None, 'gpu': None}

    def set_profile(self, profile_name):
        if profile_name in self.bg_buffers:
            self.active_profile = profile_name
            if self.bg_buffers[profile_name]['gpu'] is not None:
                self.bg_gpu = self.bg_buffers[profile_name]['gpu']
                self.has_bg = True
            else:
                self.has_bg = False
        else:
            self.active_profile = profile_name
            self.bg_buffers[profile_name] = {'cpu': None, 'gpu': None}
            self.has_bg = False
        
        # Reset Smoothing States
        self.gpu_mask_prev = None 
        self.bg_blur_gpu = None

    def reset_background(self, frame):
        if not HAS_CUDA or frame is None: return
        with self.stream:
            new_bg_gpu = cp.array(frame) if not hasattr(frame, 'device') else cp.copy(frame)
            self.bg_gpu = new_bg_gpu
            self.bg_blur_gpu = None # Reset blur buffer to force regeneration
            
            if self.active_profile not in self.bg_buffers:
                self.bg_buffers[self.active_profile] = {'cpu': None, 'gpu': None}
            self.bg_buffers[self.active_profile]['gpu'] = new_bg_gpu
            self.has_bg = True
        
        self.stream.synchronize()
        frame_cpu = cp.asnumpy(new_bg_gpu)
        save_path = os.path.join(self.data_dir, self.active_profile, "background.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, frame_cpu)

    def process(self, frame, faces, body_landmarks=None, params=None, mask=None):
        if frame is None or not HAS_CUDA: return frame
        if params is None: params = {}

        is_gpu_input = hasattr(frame, 'device')
        with self.stream:
            if is_gpu_input:
                frame_gpu = frame 
                h, w = frame.shape[:2]
            else:
                h, w = frame.shape[:2]
                frame_gpu = cp.asarray(frame)

            # Init Buffers
            if self.cache_w != w or self.cache_h != h:
                self.cache_w, self.cache_h = w, h
                self.gpu_initialized = False
                self._init_bg_buffers(w, h, frame_gpu)
                self.gpu_mask_prev = None # Reset mask buffer on resize
                
                if self.active_profile in self.bg_buffers:
                    gpu_bg = self.bg_buffers[self.active_profile]['gpu']
                    if gpu_bg is not None:
                        self.bg_gpu = gpu_bg
                        self.has_bg = True

            sw, sh = int(w * self.map_scale), int(h * self.map_scale)
            if not self.gpu_initialized:
                self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.prev_gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.prev_gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_initialized = True

            if self.bg_gpu is None:
                self.bg_gpu = cp.copy(frame_gpu)
                self.has_bg = True

            # ==================================================================
            # [1] Mask Processing: Smoothing & Dynamics
            # ==================================================================
            mask_kernel_input = None
            use_bg = 0

            if self.has_bg and mask is not None:
                # Convert to float32 (0.0 ~ 1.0) for processing
                if hasattr(mask, 'device'): 
                    mask_gpu = mask.astype(cp.float32)
                else: 
                    mask_gpu = cp.asarray(mask).astype(cp.float32)
                
                # Normalize if input is 0~255
                if mask_gpu.max() > 1.0:
                    mask_gpu /= 255.0

                # A. Temporal Smoothing (Dynamic Response)
                # "빠르면 빠르게, 느리면 느리게"
                if self.gpu_mask_prev is None:
                    self.gpu_mask_prev = mask_gpu
                else:
                    # 마스크 변화량 계산 (전체 평균)
                    # 움직임이 크면 diff 값이 커짐
                    mask_diff = cp.abs(mask_gpu - self.gpu_mask_prev).mean()
                    
                    # Diff 맵핑: 0.0(정지) -> Alpha 0.1, 0.05(빠름) -> Alpha 0.9
                    # 작은 떨림(노이즈)은 무시하고, 큰 움직임은 즉시 반영
                    target_alpha = 0.1 + (mask_diff * 20.0)
                    target_alpha = cp.clip(target_alpha, 0.1, 0.9)
                    
                    # Exponential Moving Average (EMA)
                    self.gpu_mask_prev = self.gpu_mask_prev * (1.0 - target_alpha) + mask_gpu * target_alpha
                    mask_gpu = self.gpu_mask_prev

                # B. Spatial Smoothing (Soft Edges)
                # 경계선을 흐릿하게 만들어 합성이 자연스럽게 함
                mask_gpu = cupyx.scipy.ndimage.gaussian_filter(mask_gpu, sigma=1.5)

                # Kernel expects uint8 (0~255)
                mask_kernel_input = (mask_gpu * 255.0).astype(cp.uint8)
                use_bg = 1
            else:
                mask_kernel_input = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

            # ==================================================================
            # [2] Background Blur Logic (Bokeh Effect)
            # ==================================================================
            bg_for_kernel = self.bg_gpu
            
            if self.has_bg:
                # 배경이 업데이트 되었거나 블러 버퍼가 없으면 생성
                # 매 프레임 생성하면 Adaptive Update로 인한 조명 변화도 블러 배경에 반영됨
                # 성능 고려: 5ms 이내면 허용 (RTX 3060 기준 1080p Gaussian은 빠름)
                
                bg_float = self.bg_gpu.astype(cp.float32)
                # Sigma=5.0: 꽤 흐릿하게 처리하여 왜곡을 감춤
                self.bg_blur_gpu = cupyx.scipy.ndimage.gaussian_filter(bg_float, sigma=5.0)
                self.bg_blur_gpu = self.bg_blur_gpu.astype(cp.uint8)
                bg_for_kernel = self.bg_blur_gpu

            # ==================================================================
            # [3] Warping Logic
            # ==================================================================
            self.morph_logic.clear()
            has_deformation = False
            
            body_cpu = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if body_cpu is not None:
                scaled_body = body_cpu[:, :2] * self.map_scale
                
                if params.get('shoulder_narrow', 0) > 0:
                    self.morph_logic.collect_shoulder_params(scaled_body, params['shoulder_narrow'])
                    has_deformation = True
                
                if params.get('ribcage_slim', 0) > 0:
                    self.morph_logic.collect_ribcage_params(scaled_body, params['ribcage_slim'])
                    has_deformation = True
                    
                if params.get('waist_slim', 0) > 0:
                    self.morph_logic.collect_waist_params(scaled_body, params['waist_slim'])
                    has_deformation = True
                    
                if params.get('hip_widen', 0) > 0:
                    self.morph_logic.collect_hip_params(scaled_body, params['hip_widen'])
                    has_deformation = True
            
            if faces:
                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                
                for face in faces:
                    lm_small = face.landmarks * self.map_scale
                    if face_v > 0: 
                        self.morph_logic.collect_face_contour_params(lm_small, face_v)
                        has_deformation = True
                    if eye_scale > 0: 
                        self.morph_logic.collect_eyes_params(lm_small, eye_scale)
                        has_deformation = True
                    if head_scale != 0: 
                        self.morph_logic.collect_head_params(lm_small, head_scale)
                        has_deformation = True

            self.gpu_dx.fill(0)
            self.gpu_dy.fill(0)

            warp_params = self.morph_logic.get_params()
            
            if len(warp_params) > 0:
                params_arr = np.array(warp_params, dtype=np.float32)
                params_gpu = cp.asarray(params_arr)
                
                block_dim = (16, 16)
                grid_dim = ((sw + block_dim[0] - 1) // block_dim[0], (sh + block_dim[1] - 1) // block_dim[1])
                self.warp_kernel(grid_dim, block_dim, (self.gpu_dx, self.gpu_dy, params_gpu, len(warp_params), sw, sh))
                
            if has_deformation or (self.prev_gpu_dx is not None):
                self._apply_temporal_smoothing_fast(self.current_alpha)
                
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=5, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=5, output=self.gpu_dy)

                result_gpu = cp.empty_like(frame_gpu)
                block_dim = (32, 32)
                grid_dim = ((w + block_dim[0] - 1) // block_dim[0], (h + block_dim[1] - 1) // block_dim[1])
                scale = int(1.0 / self.map_scale)
                
                # Pass 'bg_for_kernel' (Blurred BG) instead of sharp bg_gpu
                self.composite_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, mask_kernel_input, bg_for_kernel, result_gpu, self.gpu_dx, self.gpu_dy, 
                     w, h, sw, sh, scale, use_bg)
                )

                if is_gpu_input: return result_gpu
                else: return result_gpu.get()
            
            return frame_gpu if is_gpu_input else frame

    def _init_bg_buffers(self, w, h, tmpl):
        for p, data in self.bg_buffers.items():
            if data['gpu'] is None or (data['gpu'].shape[1] != w or data['gpu'].shape[0] != h):
                if data['cpu'] is not None:
                    rz = cv2.resize(data['cpu'], (w, h))
                    data['gpu'] = cp.asarray(rz)
                else:
                    data['gpu'] = cp.zeros_like(tmpl)

    def _apply_temporal_smoothing_fast(self, alpha):
        if self.prev_gpu_dx is None:
            self.prev_gpu_dx = self.gpu_dx.copy()
            self.prev_gpu_dy = self.gpu_dy.copy()
            return

        beta = 1.0 - alpha
        self.gpu_dx *= beta
        self.gpu_dx += self.prev_gpu_dx * alpha
        self.gpu_dy *= beta
        self.gpu_dy += self.prev_gpu_dy * alpha
        
        self.prev_gpu_dx[:] = self.gpu_dx
        self.prev_gpu_dy[:] = self.gpu_dy