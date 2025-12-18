# Project MUSE - beauty_engine.py
# V20.0: Precise Protection & Tone Control
# Updated: Tight Exclusion Zones (No Circles Debug) & Skin Tone Blending
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import time
from ai.tracking.facemesh import FaceMesh

# Import Kernels & Logic
from graphics.kernels.cuda_kernels import WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE, SKIN_SMOOTH_KERNEL_CODE
from graphics.processors.morph_logic import MorphLogic

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[WARNING] [BeautyEngine] CuPy not found. Fallback to CPU Mode.")

class LandmarkStabilizer:
    def __init__(self, min_cutoff=0.01, base_beta=5.0, high_speed_beta=50.0):
        self.min_cutoff = min_cutoff
        self.base_beta = base_beta
        self.high_speed_beta = high_speed_beta 
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

        dx = val_array - self.prev_val
        trend = dx / dt
        
        trend_hat = 0.5 * dx + 0.5 * self.prev_trend
        abs_trend = np.abs(trend_hat)
        
        dynamic_beta = self.base_beta + (abs_trend * 0.5) 
        dynamic_beta = np.minimum(dynamic_beta, self.high_speed_beta)

        cutoff = self.min_cutoff + dynamic_beta * abs_trend
        tau = 1.0 / (2 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        alpha = np.clip(alpha, 0.0, 1.0)

        curr_val = alpha * val_array + (1.0 - alpha) * self.prev_val
        self.prev_val = curr_val
        self.prev_trend = trend_hat
        return curr_val

class BeautyEngine:
    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V20.0 Precise Feature & Tone Ready")
        self.map_scale = 0.25 
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False
        
        self.gpu_dx = None
        self.gpu_dy = None
        
        self.body_stabilizer = LandmarkStabilizer(min_cutoff=0.01, base_beta=1.0, high_speed_beta=100.0)
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
            self.skin_kernel = cp.RawKernel(SKIN_SMOOTH_KERNEL_CODE, 'skin_smooth_kernel')
            self._warmup_kernels()
            self._load_all_backgrounds(profiles)

    def _warmup_kernels(self):
        print("   [INIT] Warming up CUDA Kernels...")
        try:
            h, w = 64, 64
            dummy_src = cp.zeros((h, w, 3), dtype=cp.uint8)
            dummy_dst = cp.zeros_like(dummy_src)
            dummy_exclusion = cp.zeros(15, dtype=cp.float32)
            
            # Updated signature with tone params
            self.skin_kernel(
                (2, 2), (32, 32), 
                (dummy_src, dummy_dst, w, h, cp.float32(0.5),
                 cp.float32(32), cp.float32(32), cp.float32(10),
                 cp.float32(128), cp.float32(128), cp.float32(128),
                 cp.float32(0.0), # tone_val (Neutral)
                 dummy_exclusion)
            )
            cp.cuda.Stream.null.synchronize()
            print("   [INIT] Kernels Compiled.")
        except Exception as e:
            print(f"   [WARNING] Warm-up failed: {e}")

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

    def _calculate_exclusion_zones(self, lm):
        """
        Calculate TIGHT protection zones.
        We stick to circular bounds for GPU efficiency but use very tight radii
        based on the actual mesh points to ensure we only cover the feature itself.
        """
        zones = []
        target_keys = ['EYE_L', 'EYE_R', 'BROW_L', 'BROW_R', 'LIPS']
        
        for key in target_keys:
            if key in FaceMesh.FACE_INDICES:
                indices = FaceMesh.FACE_INDICES[key]
                pts = lm[indices] 
                
                center = np.mean(pts, axis=0)
                cx, cy = center[0], center[1]
                
                # Use max distance to define the tightest enclosing circle
                dists = np.sqrt(np.sum((pts - center)**2, axis=1))
                max_dist = np.max(dists)
                
                # [Tuning] Very Tight Padding
                # 0.9 means we slightly undercut the outer boundary to blend the edges,
                # but protect the core feature.
                padding = 0.9 
                
                radius = max_dist * padding
                
                zones.extend([cx, cy, radius])
            else:
                zones.extend([0, 0, 0])
                
        return np.array(zones, dtype=np.float32)

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
                if hasattr(mask, 'device'): mask_gpu = mask
                else: mask_gpu = cp.asarray(mask)
                if mask_gpu.dtype == cp.float32 or mask_gpu.dtype == cp.float16:
                    mask_gpu = (mask_gpu * 255.0).astype(cp.uint8)
                use_bg = 1
            else:
                mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                use_bg = 0

            # ----------------------------------------------------
            # [Step 1.5] Skin Smoothing & Tone Correction
            # ----------------------------------------------------
            skin_strength = params.get('skin_smooth', 0.0)
            skin_tone_val = params.get('skin_tone', 0.0) # -1.0 ~ 1.0
            
            # Variables for debug drawing
            face_cx, face_cy, face_rad = 0.0, 0.0, 0.0
            target_r, target_g, target_b = 128.0, 128.0, 128.0
            exclusion_cpu = np.zeros(15, dtype=np.float32)
            face_detected = False

            source_for_warp = frame_gpu 

            if skin_strength > 0.01 or abs(skin_tone_val) > 0.01:
                # Default Params
                face_cx, face_cy, face_rad = w/2, h/2, 0.0
                
                if faces:
                    try:
                        lm = faces[0].landmarks 
                        min_xy = np.min(lm, axis=0)
                        max_xy = np.max(lm, axis=0)
                        
                        # 1. Face Region (0.8x covers full face)
                        face_center = (min_xy + max_xy) * 0.5
                        face_cx, face_cy = face_center[0], face_center[1]
                        face_rad = np.max(max_xy - min_xy) * 0.8 
                        
                        # 2. Dominant Skin Color (Center Crop)
                        roi_w = int((max_xy[0] - min_xy[0]) * 0.2)
                        roi_h = int((max_xy[1] - min_xy[1]) * 0.2)
                        roi_x = int(face_cx - roi_w/2)
                        roi_y = int(face_cy - roi_h/2)
                        
                        roi_x = max(0, min(roi_x, w-1))
                        roi_y = max(0, min(roi_y, h-1))
                        roi_w = max(1, min(roi_w, w-roi_x))
                        roi_h = max(1, min(roi_h, h-roi_y))
                        
                        if roi_w > 0 and roi_h > 0:
                            roi = frame_gpu[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                            roi_b = float(cp.median(roi[:,:,0]))
                            roi_g = float(cp.median(roi[:,:,1]))
                            roi_r = float(cp.median(roi[:,:,2]))
                            target_r, target_g, target_b = roi_r, roi_g, roi_b
                        
                        # 3. Exclusion Zones (Now Re-enabled with Tight Bounds)
                        exclusion_cpu = self._calculate_exclusion_zones(lm)
                        face_detected = True
                        
                    except Exception as e:
                        print(f"[WARN] Skin Analysis Failed: {e}")

                if face_detected:
                    smoothed_frame = cp.empty_like(frame_gpu)
                    exclusion_gpu = cp.asarray(exclusion_cpu)
                    
                    block_dim = (32, 32)
                    grid_dim = ((w + block_dim[0] - 1) // block_dim[0], (h + block_dim[1] - 1) // block_dim[1])
                    
                    try:
                        self.skin_kernel(
                            grid_dim, block_dim, 
                            (frame_gpu, smoothed_frame, w, h, cp.float32(skin_strength),
                             cp.float32(face_cx), cp.float32(face_cy), cp.float32(face_rad),
                             cp.float32(target_r), cp.float32(target_g), cp.float32(target_b),
                             cp.float32(skin_tone_val), # Pass Tone Value directly
                             exclusion_gpu)
                        )
                        source_for_warp = smoothed_frame
                    except Exception as e:
                        print(f"[ERR] Kernel Exec Failed: {e}")
                        source_for_warp = frame_gpu 

            # ----------------------------------------------------
            # [Step 2] Morph Logic
            # ----------------------------------------------------
            self.morph_logic.clear()
            raw_body = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if raw_body is not None:
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
            
            if faces:
                raw_face = faces[0].landmarks
                stable_face = self.face_stabilizer.update(raw_face)
                lm_small = stable_face * self.map_scale
                
                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                nose_slim = params.get('nose_slim', 0) 
                
                if face_v > 0: self.morph_logic.collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0: self.morph_logic.collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0: self.morph_logic.collect_head_params(lm_small, head_scale)
                if nose_slim > 0: self.morph_logic.collect_nose_params(lm_small, nose_slim)

            # 3. Warping
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
                (source_for_warp, mask_gpu, self.bg_gpu, result_gpu, self.gpu_dx, self.gpu_dy, 
                 w, h, sw, sh, scale, use_bg)
            )

            # [Visual Debugging] - NO Red Circles (Per User Request)
            if params.get('show_body_debug', False) and skin_strength > 0.01 and face_detected:
                if hasattr(result_gpu, 'get'): debug_img = result_gpu.get()
                else: debug_img = result_gpu.copy()
                
                # 1. Face ROI (Blue Circle) - Shows Skin Detection Limit
                cv2.circle(debug_img, (int(face_cx), int(face_cy)), int(face_rad), (255, 0, 0), 2)
                
                # 2. Target Skin Color (Swatch)
                cv2.rectangle(debug_img, (10, 10), (60, 60), (int(target_b), int(target_g), int(target_r)), -1)
                cv2.rectangle(debug_img, (10, 10), (60, 60), (255, 255, 255), 2)
                cv2.putText(debug_img, "Skin", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                result_gpu = cp.asarray(debug_img)

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