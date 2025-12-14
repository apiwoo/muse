# Project MUSE - beauty_engine.py
# V18.0: Gradual Response Stabilizer (Smooth Transition)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import time
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

class LandmarkStabilizer:
    """
    [Input Stabilizer V18 - Gradual Response]
    - 속도에 비례하여 Beta 값을 동적으로 조절합니다.
    - 갑작스러운 ON/OFF 스위칭 대신 부드러운 전환을 유도합니다.
    - 빠른 움직임: Beta 증가 -> Lag 감소 (번개 방지)
    - 느린 움직임: Beta 감소 -> Jitter 제거 (떨림 방지)
    """
    def __init__(self, min_cutoff=0.01, base_beta=5.0, high_speed_beta=50.0):
        self.min_cutoff = min_cutoff
        self.base_beta = base_beta
        self.high_speed_beta = high_speed_beta # 최대 속도일 때의 Beta 한계치
        
        self.prev_val = None
        self.prev_trend = None
        self.last_time = None

    def update(self, val_array):
        now = time.time()
        if self.prev_val is None:
            self.prev_val = val_array
            self.prev_trend = np.zeros_like(val_array)
            self.last_time = now
            return val_array

        dt = now - self.last_time
        if dt <= 0: return self.prev_val
        self.last_time = now

        # 1. 속도(Trend) 계산
        dx = val_array - self.prev_val
        trend = dx / dt
        
        # Trend Smoothing (급격한 속도 변화 완화)
        trend_hat = 0.5 * dx + 0.5 * self.prev_trend
        
        # 2. 동적 Beta 계산 (Dynamic Beta)
        # 움직임의 크기(abs_trend)에 따라 Beta를 선형적으로 증가시킵니다.
        # 움직임이 클수록(빠를수록) Beta가 커져서, Cutoff 주파수가 높아지고, 결과적으로 Lag가 줄어듭니다.
        abs_trend = np.abs(trend_hat)
        
        # [TUNING POINT]
        # 속도가 0에 가까우면 base_beta 사용 (떨림 방지)
        # 속도가 빨라질수록 high_speed_beta 쪽으로 가중치 이동
        # 하지만 무한정 커지지 않게 high_speed_beta로 클램핑하는 효과를 줍니다.
        
        # 간단한 선형 모델: beta = base + (speed * factor)
        # 여기서 factor는 반응성을 결정합니다.
        dynamic_beta = self.base_beta + (abs_trend * 0.5) 
        
        # 최대값 제한 (안전장치)
        dynamic_beta = np.minimum(dynamic_beta, self.high_speed_beta)

        # 3. Cutoff Frequency 계산
        cutoff = self.min_cutoff + dynamic_beta * abs_trend
        
        # 4. Alpha 계산 (Low Pass Filter 계수)
        tau = 1.0 / (2 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        
        # Alpha 클램핑 (0.0 ~ 1.0)
        # alpha가 1.0에 가까울수록 원본을 그대로 반영 (Lag 없음)
        alpha = np.clip(alpha, 0.0, 1.0)

        # 5. 최종 값 계산
        curr_val = alpha * val_array + (1.0 - alpha) * self.prev_val
        
        self.prev_val = curr_val
        self.prev_trend = trend_hat
        
        return curr_val

class BeautyEngine:
    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V18.0 Gradual Stabilizer Ready")
        self.map_scale = 0.25 
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        
        # [TUNING] Stabilizer Parameters
        # Body: 
        # - base_beta=1.0: 정지 시 적당한 스무딩
        # - high_speed_beta=100.0: 빠를 땐 거의 원본에 가깝게 (Lag 제거)
        self.body_stabilizer = LandmarkStabilizer(min_cutoff=0.01, base_beta=1.0, high_speed_beta=100.0)
        
        # Face:
        # - base_beta=5.0: 얼굴은 미세 떨림에 민감하므로 기본값을 좀 더 높게
        # - high_speed_beta=50.0: 너무 빠르면 오히려 어색할 수 있어 제한
        self.face_stabilizer = LandmarkStabilizer(min_cutoff=0.5, base_beta=5.0, high_speed_beta=50.0)
        
        self.bg_buffers = {}
        self.active_profile = 'default'
        self.bg_gpu = None
        self.has_bg = False
        
        self.morph_logic = MorphLogic()
        
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
            # 1. Frame Setup
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
                
                # Reset Stabilizers
                self.body_stabilizer = LandmarkStabilizer(min_cutoff=0.01, base_beta=1.0, high_speed_beta=100.0)
                self.face_stabilizer = LandmarkStabilizer(min_cutoff=0.5, base_beta=5.0, high_speed_beta=50.0)
                
                if self.active_profile in self.bg_buffers and self.bg_buffers[self.active_profile]['gpu'] is not None:
                    self.bg_gpu = self.bg_buffers[self.active_profile]['gpu']
                    self.has_bg = True

            sw, sh = int(w * self.map_scale), int(h * self.map_scale)
            if not self.gpu_initialized:
                self.gpu_dx = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_dy = cp.zeros((sh, sw), dtype=cp.float32)
                self.gpu_initialized = True

            if self.bg_gpu is None:
                self.bg_gpu = cp.copy(frame_gpu)
                self.has_bg = True

            if self.has_bg and mask is not None:
                if hasattr(mask, 'device'): 
                    mask_gpu = mask
                else: 
                    mask_gpu = cp.asarray(mask)
                if mask_gpu.dtype == cp.float32 or mask_gpu.dtype == cp.float16:
                    mask_gpu = (mask_gpu * 255.0).astype(cp.uint8)
                use_bg = 1
            else:
                mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

            # 2. Logic: Input Stabilization & Collection
            self.morph_logic.clear()
            
            # (A) Body Processing
            raw_body = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if raw_body is not None:
                # [CORE FIX] Stabilize Input Landmarks
                kpts_xy = raw_body[:, :2]
                stable_kpts = self.body_stabilizer.update(kpts_xy)
                
                scaled_body = stable_kpts * self.map_scale
                
                if params.get('shoulder_narrow', 0) > 0:
                    self.morph_logic.collect_shoulder_params(scaled_body, params['shoulder_narrow'])
                
                if params.get('ribcage_slim', 0) > 0:
                    self.morph_logic.collect_ribcage_params(scaled_body, params['ribcage_slim'])
                    
                if params.get('waist_slim', 0) > 0:
                    self.morph_logic.collect_waist_params(scaled_body, params['waist_slim'])
                    
                if params.get('hip_widen', 0) > 0:
                    self.morph_logic.collect_hip_params(scaled_body, params['hip_widen'])
            
            # (B) Face Processing
            if faces:
                raw_face = faces[0].landmarks
                stable_face = self.face_stabilizer.update(raw_face)
                
                lm_small = stable_face * self.map_scale
                
                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                
                if face_v > 0: 
                    self.morph_logic.collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0: 
                    self.morph_logic.collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0: 
                    self.morph_logic.collect_head_params(lm_small, head_scale)

            # 3. Rendering: Zero-Lag Warping
            self.gpu_dx.fill(0)
            self.gpu_dy.fill(0)

            warp_params = self.morph_logic.get_params()
            
            if len(warp_params) > 0:
                params_arr = np.array(warp_params, dtype=np.float32)
                params_gpu = cp.asarray(params_arr)
                
                block_dim = (16, 16)
                grid_dim = ((sw + block_dim[0] - 1) // block_dim[0], (sh + block_dim[1] - 1) // block_dim[1])
                self.warp_kernel(grid_dim, block_dim, (self.gpu_dx, self.gpu_dy, params_gpu, len(warp_params), sw, sh))
                
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=1, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=1, output=self.gpu_dy)

            # 4. Composite
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

    def _init_bg_buffers(self, w, h, tmpl):
        for p, data in self.bg_buffers.items():
            if data['gpu'] is None or (data['gpu'].shape[1] != w or data['gpu'].shape[0] != h):
                if data['cpu'] is not None:
                    rz = cv2.resize(data['cpu'], (w, h))
                    data['gpu'] = cp.asarray(rz)
                else:
                    data['gpu'] = cp.zeros_like(tmpl)