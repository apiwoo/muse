# Project MUSE - beauty_engine.py
# V15.2: Float32 Mask Compatibility Update
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
        print("[BEAUTY] [BeautyEngine] V15.2 Alpha Blending Ready")
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
        self.has_bg = False
        
        self.morph_logic = MorphLogic()
        self.current_alpha = 0.85
        
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

    def reset_background(self, frame):
        if not HAS_CUDA or frame is None: return
        with self.stream:
            new_bg_gpu = cp.array(frame) if not hasattr(frame, 'device') else cp.copy(frame)
            self.bg_gpu = new_bg_gpu
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

            if self.cache_w != w or self.cache_h != h:
                self.cache_w, self.cache_h = w, h
                self.gpu_initialized = False
                self._init_bg_buffers(w, h, frame_gpu)
                
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

            # [CRITICAL FIX] Type Safety for Mask (Float32 -> Uint8)
            # MODNet returns 0.0~1.0 float32, but Kernel expects 0~255 uint8
            if self.has_bg and mask is not None:
                if hasattr(mask, 'device'): 
                    mask_gpu = mask
                else: 
                    mask_gpu = cp.asarray(mask)
                
                # Auto-conversion if float
                if mask_gpu.dtype == cp.float32 or mask_gpu.dtype == cp.float16:
                    mask_gpu = (mask_gpu * 255.0).astype(cp.uint8)
                
                use_bg = 1
            else:
                mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

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
                
                self.composite_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, mask_gpu, self.bg_gpu, result_gpu, self.gpu_dx, self.gpu_dy, 
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