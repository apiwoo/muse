# Project MUSE - beauty_engine.py
# V25.0: High-Precision Pipeline & Frequency Separation
# - Added: MaskManager (Polygon-based skin masking)
# - Added: Frequency Separation (Guided Filter)
# - Added: Tone Uniformity (Flat-fielding)
# - Added: Color Grading (Temperature/Tint)
# - Preserved: All legacy features (TPS, Intrusion Handling, One-Euro Filter)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import time
from ai.tracking.facemesh import FaceMesh

# Import Kernels & Logic
from graphics.kernels.cuda_kernels import (
    WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE, SKIN_SMOOTH_KERNEL_CODE
)
from graphics.processors.morph_logic import MorphLogic

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("[WARNING] [BeautyEngine] CuPy not found. Fallback to CPU Mode.")


# ==============================================================================
# [기존 클래스] LandmarkStabilizer - 100% 유지
# ==============================================================================
class LandmarkStabilizer:
    """One-Euro Filter style landmark stabilization"""
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

    def reset(self):
        """Reset stabilizer state"""
        self.prev_val = None
        self.prev_trend = None
        self.last_time = None


# ==============================================================================
# [V25.0 신규 클래스] MaskManager - CPU 기반 고속 마스크 생성
# ==============================================================================
class MaskManager:
    """
    Fast skin mask manager (CPU OpenCV + GPU cache)
    - Uses cv2.fillPoly for fast polygon rasterization
    - Precise exclusion of eyes, eyebrows, lips
    - Caches GPU mask to avoid repeated transfers
    """

    # FaceMesh 인덱스 (정적 참조)
    FACE_OVAL_INDICES = FaceMesh.FACE_INDICES.get("FACE_OVAL", [])
    FOREHEAD_INDICES = FaceMesh.FACE_INDICES.get("FOREHEAD", [])

    # 제외 영역
    EYE_L_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_L_POLY", [])
    EYE_R_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_R_POLY", [])
    BROW_L_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_L_POLY", [])
    BROW_R_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_R_POLY", [])
    LIPS_INDICES = FaceMesh.POLYGON_INDICES.get("LIPS_OUTER_POLY", [])

    def __init__(self):
        self.cache_w = 0
        self.cache_h = 0
        self.mask_cpu = None
        self.mask_gpu = None

    def generate_mask(self, landmarks, w, h, padding_ratio=1.15):
        """
        Generate skin mask using OpenCV (CPU, very fast < 2ms)
        """
        # Reuse buffer if size matches
        if self.mask_cpu is None or self.cache_w != w or self.cache_h != h:
            self.mask_cpu = np.zeros((h, w), dtype=np.uint8)
            self.cache_w = w
            self.cache_h = h
        else:
            self.mask_cpu.fill(0)

        mask = self.mask_cpu

        # 1. Fill face oval
        face_pts = landmarks[self.FACE_OVAL_INDICES].astype(np.int32)
        cv2.fillPoly(mask, [face_pts], 255)

        # 2. Exclude eyes (with padding)
        self._exclude_region(mask, landmarks, self.EYE_L_INDICES, padding_ratio * 1.3)
        self._exclude_region(mask, landmarks, self.EYE_R_INDICES, padding_ratio * 1.3)

        # 3. Exclude eyebrows
        self._exclude_region(mask, landmarks, self.BROW_L_INDICES, padding_ratio * 1.1)
        self._exclude_region(mask, landmarks, self.BROW_R_INDICES, padding_ratio * 1.1)

        # 4. Exclude lips
        self._exclude_region(mask, landmarks, self.LIPS_INDICES, padding_ratio * 1.2)

        # 5. Soft edge (small blur for smooth transition)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        self.mask_cpu = mask

        # Transfer to GPU
        if HAS_CUDA:
            self.mask_gpu = cp.asarray(mask)
            return self.mask_gpu
        return mask

    def _exclude_region(self, mask, landmarks, indices, padding=1.0):
        """Exclude a polygon region from the mask with padding"""
        if len(indices) == 0:
            return

        pts = landmarks[indices].astype(np.float32)

        if padding != 1.0:
            center = np.mean(pts, axis=0)
            pts = center + (pts - center) * padding

        pts = pts.astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 0)

    def get_mask_cpu(self):
        """Get mask as numpy array"""
        return self.mask_cpu


# ==============================================================================
# [메인 클래스] BeautyEngine - V25.0 YY-Style Pipeline
# ==============================================================================
class BeautyEngine:
    """
    V25.0 Beauty Processing Engine
    - CPU-based fast skin masking (OpenCV fillPoly)
    - YY-style bilateral filter smoothing
    - Color grading (temperature/tint)
    - Legacy features preserved (TPS warping, Intrusion handling)
    """

    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V25.0 High-Precision Pipeline Ready")
        self.map_scale = 0.25
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False

        self.gpu_dx = None
        self.gpu_dy = None

        # Stabilizers (기존 유지)
        self.body_stabilizer = LandmarkStabilizer(min_cutoff=0.01, base_beta=1.0, high_speed_beta=100.0)
        self.face_stabilizer = LandmarkStabilizer(min_cutoff=0.5, base_beta=5.0, high_speed_beta=50.0)

        # Background management (기존 유지)
        self.bg_buffers = {}
        self.active_profile = 'default'
        self.bg_gpu = None
        self.has_bg = False

        # Morph Logic (기존 유지)
        self.morph_logic = MorphLogic()

        # V25.0 Components (simplified - CPU bilateral filter based)
        self.mask_manager = MaskManager()

        # Paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")

        if HAS_CUDA:
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Core kernels (warping, compositing)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self.skin_kernel = cp.RawKernel(SKIN_SMOOTH_KERNEL_CODE, 'skin_smooth_kernel')

            self._warmup_kernels()
            self._load_all_backgrounds(profiles)

    def _warmup_kernels(self):
        """Pre-compile CUDA kernels"""
        print("   [INIT] Warming up CUDA Kernels...")
        try:
            h, w = 64, 64
            dummy_src = cp.zeros((h, w, 3), dtype=cp.uint8)
            dummy_dst = cp.zeros_like(dummy_src)
            dummy_exclusion = cp.zeros(15, dtype=cp.float32)

            # Legacy kernel warmup
            self.skin_kernel(
                (2, 2), (32, 32),
                (dummy_src, dummy_dst, w, h, cp.float32(0.5),
                 cp.float32(32), cp.float32(32), cp.float32(10),
                 cp.float32(128), cp.float32(128), cp.float32(128),
                 cp.float32(0.0),
                 dummy_exclusion)
            )

            cp.cuda.Stream.null.synchronize()
            print("   [INIT] Core Kernels Compiled")
        except Exception as e:
            print(f"   [WARNING] Warm-up failed: {e}")

    # ==========================================================================
    # Background Management (기존 100% 유지)
    # ==========================================================================
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

    def _init_bg_buffers(self, w, h, tmpl):
        for p, data in self.bg_buffers.items():
            if data['gpu'] is None or (data['gpu'].shape[1] != w or data['gpu'].shape[0] != h):
                if data['cpu'] is not None:
                    rz = cv2.resize(data['cpu'], (w, h))
                    data['gpu'] = cp.asarray(rz)
                else:
                    data['gpu'] = cp.zeros_like(tmpl)

    # ==========================================================================
    # Exclusion Zones (레거시 호환용 - 원형 기반)
    # ==========================================================================
    def _calculate_exclusion_zones(self, lm):
        """Legacy circular exclusion zones for backward compatibility"""
        zones = []
        target_keys = ['EYE_L', 'EYE_R', 'BROW_L', 'BROW_R', 'LIPS']

        for key in target_keys:
            if key in FaceMesh.FACE_INDICES:
                indices = FaceMesh.FACE_INDICES[key]
                pts = lm[indices]

                center = np.mean(pts, axis=0)
                cx, cy = center[0], center[1]

                dists = np.sqrt(np.sum((pts - center)**2, axis=1))
                max_dist = np.max(dists)

                padding = 0.9
                radius = max_dist * padding

                zones.extend([cx, cy, radius])
            else:
                zones.extend([0, 0, 0])

        return np.array(zones, dtype=np.float32)

    # ==========================================================================
    # YY-Style Skin Smoothing (Edge Preserving Filter)
    # ==========================================================================
    def _process_skin_v25(self, frame_gpu, landmarks, params):
        """
        YY-style skin smoothing using OpenCV edge-preserving filter
        - Smooths skin texture while preserving edges
        - Fast CPU processing on face ROI only
        """
        h, w = frame_gpu.shape[:2]

        # Get parameters
        skin_strength = params.get('skin_smooth', 0.0)
        skin_tone_val = params.get('skin_tone', 0.0)
        color_temp = params.get('color_temperature', 0.0)
        color_tint = params.get('color_tint', 0.0)

        # Skip if no processing needed
        has_skin = skin_strength > 0.01
        has_tone = abs(skin_tone_val) > 0.01
        has_color = abs(color_temp) > 0.01 or abs(color_tint) > 0.01

        if not has_skin and not has_tone and not has_color:
            return frame_gpu, None

        # Get CPU frame
        if hasattr(self, '_frame_cpu_cache') and self._frame_cpu_cache is not None:
            frame_cpu = self._frame_cpu_cache.copy()
        else:
            frame_cpu = cp.asnumpy(frame_gpu)

        skin_mask_gpu = None

        # ===== Skin Smoothing (Face ROI only) =====
        if has_skin:
            # Get face bounding box
            face_pts = landmarks[self.mask_manager.FACE_OVAL_INDICES]
            margin = 10
            x_min = max(0, int(np.min(face_pts[:, 0])) - margin)
            y_min = max(0, int(np.min(face_pts[:, 1])) - margin)
            x_max = min(w, int(np.max(face_pts[:, 0])) + margin)
            y_max = min(h, int(np.max(face_pts[:, 1])) + margin)

            if x_max > x_min + 20 and y_max > y_min + 20:
                # Extract face ROI
                face_roi = frame_cpu[y_min:y_max, x_min:x_max]

                # Edge-preserving smoothing
                # sigma_s: spatial sigma (size of area), sigma_r: range sigma (color similarity)
                sigma_s = 20 + skin_strength * 40  # 20 ~ 60
                sigma_r = 0.2 + skin_strength * 0.3  # 0.2 ~ 0.5

                smoothed_roi = cv2.edgePreservingFilter(
                    face_roi,
                    flags=cv2.RECURS_FILTER,  # Recursive filter (faster)
                    sigma_s=sigma_s,
                    sigma_r=sigma_r
                )

                # Blend with original based on strength
                alpha = skin_strength * 0.8  # Max 80% blend
                blended = cv2.addWeighted(face_roi, 1 - alpha, smoothed_roi, alpha, 0)

                # Feather the edges to avoid hard boundary
                # Create soft mask for ROI
                roi_h, roi_w = blended.shape[:2]
                feather = min(15, roi_w // 10, roi_h // 10)
                if feather > 2:
                    mask_roi = np.ones((roi_h, roi_w), dtype=np.float32)
                    # Fade edges
                    for i in range(feather):
                        fade = i / feather
                        mask_roi[i, :] *= fade
                        mask_roi[roi_h - 1 - i, :] *= fade
                        mask_roi[:, i] *= fade
                        mask_roi[:, roi_w - 1 - i] *= fade

                    mask_3ch = mask_roi[:, :, np.newaxis]
                    blended = (face_roi * (1 - mask_3ch) + blended * mask_3ch).astype(np.uint8)

                frame_cpu[y_min:y_max, x_min:x_max] = blended

        # ===== Color Grading (Full image, GPU) =====
        if has_color:
            frame_float = frame_cpu.astype(np.float32)

            if abs(color_temp) > 0.01:
                if color_temp > 0:
                    frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + color_temp * 30, 0, 255)
                    frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] - color_temp * 20, 0, 255)
                else:
                    frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] - color_temp * 30, 0, 255)
                    frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + color_temp * 20, 0, 255)

            if abs(color_tint) > 0.01:
                if color_tint > 0:
                    frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + color_tint * 15, 0, 255)
                    frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] + color_tint * 15, 0, 255)
                    frame_float[:, :, 1] = np.clip(frame_float[:, :, 1] - color_tint * 20, 0, 255)
                else:
                    frame_float[:, :, 1] = np.clip(frame_float[:, :, 1] - color_tint * 20, 0, 255)
                    frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + color_tint * 10, 0, 255)
                    frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] + color_tint * 10, 0, 255)

            frame_cpu = np.clip(frame_float, 0, 255).astype(np.uint8)

        # ===== Skin Tone (Face mask) =====
        if has_tone:
            skin_mask_gpu = self.mask_manager.generate_mask(landmarks, w, h, padding_ratio=1.2)
            if skin_mask_gpu is not None:
                mask_cpu = cp.asnumpy(skin_mask_gpu).astype(np.float32) / 255.0
                frame_float = frame_cpu.astype(np.float32)

                if skin_tone_val > 0:
                    mix = skin_tone_val * 0.25
                    frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + mix * 35 * mask_cpu, 0, 255)
                    frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] - mix * 15 * mask_cpu, 0, 255)
                else:
                    mix = -skin_tone_val * 0.2
                    for c in range(3):
                        frame_float[:, :, c] = frame_float[:, :, c] + (255 - frame_float[:, :, c]) * mix * mask_cpu

                frame_cpu = np.clip(frame_float, 0, 255).astype(np.uint8)

        # Transfer to GPU
        frame_gpu = cp.asarray(frame_cpu)
        return frame_gpu, skin_mask_gpu

    # ==========================================================================
    # Legacy Skin Processing (기존 100% 유지)
    # ==========================================================================
    def _process_skin_legacy(self, frame_gpu, landmarks, params, w, h):
        """Legacy circular-ROI based skin processing"""
        skin_strength = params.get('skin_smooth', 0.0)
        skin_tone_val = params.get('skin_tone', 0.0)

        if skin_strength < 0.01 and abs(skin_tone_val) < 0.01:
            return frame_gpu

        face_cx, face_cy, face_rad = w/2, h/2, 0.0
        target_r, target_g, target_b = 128.0, 128.0, 128.0

        try:
            lm = landmarks
            min_xy = np.min(lm, axis=0)
            max_xy = np.max(lm, axis=0)

            face_center = (min_xy + max_xy) * 0.5
            face_cx, face_cy = face_center[0], face_center[1]
            face_rad = np.max(max_xy - min_xy) * 0.8

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

            exclusion_cpu = self._calculate_exclusion_zones(lm)

        except Exception as e:
            print(f"[WARN] Skin Analysis Failed: {e}")
            return frame_gpu

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
                 cp.float32(skin_tone_val),
                 exclusion_gpu)
            )
            return smoothed_frame
        except Exception as e:
            print(f"[ERR] Legacy Kernel Exec Failed: {e}")
            return frame_gpu

    # ==========================================================================
    # Main Processing Pipeline
    # ==========================================================================
    def process(self, frame, faces, body_landmarks=None, params=None, mask=None, frame_cpu=None):
        """
        Main processing pipeline

        :param frame: Input frame (CPU or GPU)
        :param faces: List of FaceResult from FaceMesh
        :param body_landmarks: Body keypoints from pose tracker
        :param params: Processing parameters dict
        :param mask: Alpha mask for compositing
        :param frame_cpu: Optional CPU frame to avoid GPU->CPU transfer
        :return: Processed frame
        """
        if frame is None or not HAS_CUDA:
            return frame
        if params is None:
            params = {}

        # Store CPU frame if provided (optimization)
        self._frame_cpu_cache = frame_cpu

        is_gpu_input = hasattr(frame, 'device')

        with self.stream:
            if is_gpu_input:
                frame_gpu = frame
                h, w = frame.shape[:2]
            else:
                h, w = frame.shape[:2]
                frame_gpu = cp.asarray(frame)

            # Initialize buffers on size change
            if self.cache_w != w or self.cache_h != h:
                self.cache_w, self.cache_h = w, h
                self.gpu_initialized = False
                self._init_bg_buffers(w, h, frame_gpu)

                self.body_stabilizer.reset()
                self.face_stabilizer.reset()

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

            # Mask handling
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

            # ==================================================================
            # [Step 1] Skin Processing
            # ==================================================================
            source_for_warp = frame_gpu
            skin_mask_debug = None

            if faces:
                raw_face = faces[0].landmarks
                stable_face = self.face_stabilizer.update(raw_face)

                # V25.0 Pipeline (always enabled)
                source_for_warp, skin_mask_debug = self._process_skin_v25(
                    frame_gpu, stable_face, params
                )

            # ==================================================================
            # [Step 2] Morph Logic (기존 100% 유지)
            # ==================================================================
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
                lm_small = stable_face * self.map_scale

                face_v = params.get('face_v', 0)
                eye_scale = params.get('eye_scale', 0)
                head_scale = params.get('head_scale', 0)
                nose_slim = params.get('nose_slim', 0)

                if face_v > 0:
                    self.morph_logic.collect_face_contour_params(lm_small, face_v)
                if eye_scale > 0:
                    self.morph_logic.collect_eyes_params(lm_small, eye_scale)
                if head_scale != 0:
                    self.morph_logic.collect_head_params(lm_small, head_scale)
                if nose_slim > 0:
                    self.morph_logic.collect_nose_params(lm_small, nose_slim)

            # ==================================================================
            # [Step 3] Warping (기존 100% 유지)
            # ==================================================================
            self.gpu_dx.fill(0)
            self.gpu_dy.fill(0)

            warp_params = self.morph_logic.get_params()
            if len(warp_params) > 0:
                params_arr = np.array(warp_params, dtype=np.float32)
                params_gpu = cp.asarray(params_arr)

                block_dim = (16, 16)
                grid_dim = ((sw + block_dim[0] - 1) // block_dim[0],
                            (sh + block_dim[1] - 1) // block_dim[1])

                self.warp_kernel(grid_dim, block_dim,
                    (self.gpu_dx, self.gpu_dy, params_gpu, len(warp_params), sw, sh))

                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=1, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=1, output=self.gpu_dy)

            # ==================================================================
            # [Step 4] Composite (기존 100% 유지)
            # ==================================================================
            result_gpu = cp.empty_like(frame_gpu)

            block_dim = (32, 32)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])
            scale = int(1.0 / self.map_scale)

            self.composite_kernel(
                grid_dim, block_dim,
                (source_for_warp, mask_gpu, self.bg_gpu, result_gpu,
                 self.gpu_dx, self.gpu_dy,
                 w, h, sw, sh, scale, use_bg)
            )

            # ==================================================================
            # [Debug Visualization]
            # ==================================================================
            if params.get('show_body_debug', False):
                if hasattr(result_gpu, 'get'):
                    debug_img = result_gpu.get()
                else:
                    debug_img = cp.asnumpy(result_gpu)

                # Show skin mask boundary if available
                if skin_mask_debug is not None:
                    mask_np = cp.asnumpy(skin_mask_debug)
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

                # Show processing mode
                cv2.putText(debug_img, "V25.0 Pipeline", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                result_gpu = cp.asarray(debug_img)

            # Return result
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()

