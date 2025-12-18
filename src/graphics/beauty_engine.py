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
    WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE, SKIN_SMOOTH_KERNEL_CODE,
    # V25.0 New Kernels
    POLYGON_MASK_KERNEL_CODE, SKIN_MASK_KERNEL_CODE,
    GUIDED_FILTER_KERNEL_CODE, TONE_UNIFORMITY_KERNEL_CODE,
    COLOR_GRADING_KERNEL_CODE, SKIN_SMOOTH_V2_KERNEL_CODE,
    MASK_BLUR_KERNEL_CODE
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
# [V25.0 신규 클래스] MaskManager - 폴리곤 기반 피부 마스크 관리
# ==============================================================================
class MaskManager:
    """
    Polygon-based skin mask manager
    - Generates precise masks from FaceMesh landmarks
    - Handles face oval and exclusion zones (eyes, brows, lips)
    - GPU-accelerated mask generation
    """

    # 클래스 레벨 인덱스 (FaceMesh 인스턴스 생성 없이 직접 참조)
    FACE_OVAL_INDICES = FaceMesh.FACE_INDICES.get("FACE_OVAL", [])
    EYE_L_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_L_POLY", FaceMesh.FACE_INDICES.get("EYE_L", []))
    EYE_R_INDICES = FaceMesh.POLYGON_INDICES.get("EYE_R_POLY", FaceMesh.FACE_INDICES.get("EYE_R", []))
    BROW_L_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_L_POLY", FaceMesh.FACE_INDICES.get("BROW_L", []))
    BROW_R_INDICES = FaceMesh.POLYGON_INDICES.get("BROW_R_POLY", FaceMesh.FACE_INDICES.get("BROW_R", []))
    LIPS_INDICES = FaceMesh.POLYGON_INDICES.get("LIPS_OUTER_POLY", FaceMesh.FACE_INDICES.get("LIPS", []))

    def __init__(self):
        self.skin_mask_gpu = None
        self.soft_mask_gpu = None
        self.cache_w = 0
        self.cache_h = 0

        # Kernel compilation (lazy)
        self._skin_mask_kernel = None
        self._mask_blur_kernel = None

    def _compile_kernels(self):
        """Lazy kernel compilation"""
        if self._skin_mask_kernel is None:
            self._skin_mask_kernel = cp.RawKernel(SKIN_MASK_KERNEL_CODE, 'skin_mask_kernel')
            self._mask_blur_kernel = cp.RawKernel(MASK_BLUR_KERNEL_CODE, 'mask_blur_kernel')

    def generate_mask(self, landmarks, w, h, feather_radius=3):
        """
        Generate skin mask from landmarks

        :param landmarks: (478, 2) numpy array from FaceMesh
        :param w: Image width
        :param h: Image height
        :param feather_radius: Edge softening radius
        :return: GPU mask array (soft edges)
        """
        if not HAS_CUDA:
            return None

        self._compile_kernels()

        # Reinitialize buffers if size changed
        if self.cache_w != w or self.cache_h != h:
            self.skin_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            self.soft_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            self.cache_w = w
            self.cache_h = h
        else:
            self.skin_mask_gpu.fill(0)

        # Get polygon data using class-level indices (no FaceMesh instantiation)
        face_oval = landmarks[self.FACE_OVAL_INDICES].astype(np.float32)
        eye_l = landmarks[self.EYE_L_INDICES].astype(np.float32)
        eye_r = landmarks[self.EYE_R_INDICES].astype(np.float32)
        brow_l = landmarks[self.BROW_L_INDICES].astype(np.float32)
        brow_r = landmarks[self.BROW_R_INDICES].astype(np.float32)
        lips = landmarks[self.LIPS_INDICES].astype(np.float32)

        # Flatten vertex arrays for GPU
        face_verts = face_oval.flatten()
        eye_l_verts = eye_l.flatten()
        eye_r_verts = eye_r.flatten()
        brow_l_verts = brow_l.flatten()
        brow_r_verts = brow_r.flatten()
        lips_verts = lips.flatten()

        # Transfer to GPU
        face_gpu = cp.asarray(face_verts)
        eye_l_gpu = cp.asarray(eye_l_verts)
        eye_r_gpu = cp.asarray(eye_r_verts)
        brow_l_gpu = cp.asarray(brow_l_verts)
        brow_r_gpu = cp.asarray(brow_r_verts)
        lips_gpu = cp.asarray(lips_verts)

        # Execute skin mask kernel
        block_dim = (32, 32)
        grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                    (h + block_dim[1] - 1) // block_dim[1])

        self._skin_mask_kernel(
            grid_dim, block_dim,
            (self.skin_mask_gpu,
             face_gpu, len(face_oval),
             eye_l_gpu, len(eye_l),
             eye_r_gpu, len(eye_r),
             brow_l_gpu, len(brow_l),
             brow_r_gpu, len(brow_r),
             lips_gpu, len(lips),
             w, h, cp.float32(1.1))  # 1.1 = 10% padding for exclusions
        )

        # Apply feathering for soft edges
        if feather_radius > 0:
            self._mask_blur_kernel(
                grid_dim, block_dim,
                (self.skin_mask_gpu, self.soft_mask_gpu, w, h, feather_radius)
            )
            return self.soft_mask_gpu
        else:
            return self.skin_mask_gpu

    def get_mask_cpu(self):
        """Get current mask as CPU numpy array"""
        if self.soft_mask_gpu is not None:
            return cp.asnumpy(self.soft_mask_gpu)
        elif self.skin_mask_gpu is not None:
            return cp.asnumpy(self.skin_mask_gpu)
        return None


# ==============================================================================
# [V25.0 신규 클래스] FrequencySeparator - 주파수 분리 처리
# ==============================================================================
class FrequencySeparator:
    """
    Frequency separation for skin processing
    - Guided Filter based edge-aware low-pass
    - Preserves high-frequency detail (skin texture, fine lines)
    - Enables flat-fielding for tone uniformity
    """
    def __init__(self):
        self.low_freq_gpu = None
        self.cache_w = 0
        self.cache_h = 0

        self._guided_kernel = None
        self._tone_kernel = None

    def _compile_kernels(self):
        if self._guided_kernel is None:
            self._guided_kernel = cp.RawKernel(GUIDED_FILTER_KERNEL_CODE, 'guided_filter_kernel')
            self._tone_kernel = cp.RawKernel(TONE_UNIFORMITY_KERNEL_CODE, 'tone_uniformity_kernel')

    def extract_low_frequency(self, frame_gpu, mask_gpu, radius=8, epsilon=0.04):
        """
        Extract low-frequency component using Guided Filter

        :param frame_gpu: Input frame (GPU)
        :param mask_gpu: Skin mask (GPU)
        :param radius: Filter radius
        :param epsilon: Edge preservation factor
        :return: Low-frequency component (GPU)
        """
        if not HAS_CUDA:
            return frame_gpu

        self._compile_kernels()

        h, w = frame_gpu.shape[:2]

        if self.cache_w != w or self.cache_h != h:
            self.low_freq_gpu = cp.empty_like(frame_gpu)
            self.cache_w = w
            self.cache_h = h

        block_dim = (16, 16)
        grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                    (h + block_dim[1] - 1) // block_dim[1])

        self._guided_kernel(
            grid_dim, block_dim,
            (frame_gpu, frame_gpu, self.low_freq_gpu, mask_gpu,
             w, h, radius, cp.float32(epsilon))
        )

        return self.low_freq_gpu

    def apply_tone_uniformity(self, frame_gpu, low_freq_gpu, mask_gpu,
                              mean_color, flatten_strength=0.3, detail_preserve=0.7):
        """
        Apply tone uniformity (flat-fielding)

        :param frame_gpu: Original frame
        :param low_freq_gpu: Low-frequency from guided filter
        :param mask_gpu: Skin mask
        :param mean_color: Target mean skin color (B, G, R)
        :param flatten_strength: How much to push towards mean (0-1)
        :param detail_preserve: High-frequency preservation (0-1)
        :return: Processed frame (GPU)
        """
        if not HAS_CUDA:
            return frame_gpu

        self._compile_kernels()

        h, w = frame_gpu.shape[:2]
        result_gpu = cp.empty_like(frame_gpu)

        block_dim = (16, 16)
        grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                    (h + block_dim[1] - 1) // block_dim[1])

        self._tone_kernel(
            grid_dim, block_dim,
            (frame_gpu, low_freq_gpu, result_gpu, mask_gpu,
             w, h,
             cp.float32(mean_color[0]), cp.float32(mean_color[1]), cp.float32(mean_color[2]),
             cp.float32(flatten_strength), cp.float32(detail_preserve))
        )

        return result_gpu


# ==============================================================================
# [V25.0 신규 클래스] ColorGrader - 색상 그레이딩
# ==============================================================================
class ColorGrader:
    """
    Color grading processor
    - Temperature: Cool (Blue) ↔ Warm (Yellow)
    - Tint: Green ↔ Magenta
    - HSL-based efficient implementation
    """
    def __init__(self):
        self._grading_kernel = None

    def _compile_kernels(self):
        if self._grading_kernel is None:
            self._grading_kernel = cp.RawKernel(COLOR_GRADING_KERNEL_CODE, 'color_grading_kernel')

    def apply(self, frame_gpu, mask_gpu=None, temperature=0.0, tint=0.0, use_mask=True):
        """
        Apply color grading

        :param frame_gpu: Input frame (GPU)
        :param mask_gpu: Optional mask (apply only to masked areas)
        :param temperature: -1.0 (cool) to 1.0 (warm)
        :param tint: -1.0 (green) to 1.0 (magenta)
        :param use_mask: Whether to use mask
        :return: Graded frame (GPU)
        """
        if not HAS_CUDA:
            return frame_gpu

        # Skip if no adjustment needed
        if abs(temperature) < 0.01 and abs(tint) < 0.01:
            return frame_gpu

        self._compile_kernels()

        h, w = frame_gpu.shape[:2]
        result_gpu = cp.empty_like(frame_gpu)

        block_dim = (16, 16)
        grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                    (h + block_dim[1] - 1) // block_dim[1])

        # Handle null mask
        if mask_gpu is None:
            mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            use_mask_flag = 0
        else:
            use_mask_flag = 1 if use_mask else 0

        self._grading_kernel(
            grid_dim, block_dim,
            (frame_gpu, result_gpu, mask_gpu,
             w, h,
             cp.float32(temperature), cp.float32(tint),
             use_mask_flag)
        )

        return result_gpu


# ==============================================================================
# [메인 클래스] BeautyEngine - V25.0 통합 파이프라인
# ==============================================================================
class BeautyEngine:
    """
    V25.0 Beauty Processing Engine
    - Polygon-based precise skin masking
    - Frequency separation for YY-style porcelain skin
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

        # V25.0 New Components
        self.mask_manager = MaskManager()
        self.freq_separator = FrequencySeparator()
        self.color_grader = ColorGrader()

        # Processing mode
        self.use_v25_pipeline = True  # Set to False to use legacy pipeline

        # Paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(self.root_dir, "recorded_data", "personal_data")

        if HAS_CUDA:
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Legacy kernels (유지)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self.skin_kernel = cp.RawKernel(SKIN_SMOOTH_KERNEL_CODE, 'skin_smooth_kernel')
            # V25.0 kernels
            self.skin_v2_kernel = cp.RawKernel(SKIN_SMOOTH_V2_KERNEL_CODE, 'skin_smooth_v2_kernel')

            self._warmup_kernels()
            self._load_all_backgrounds(profiles)

    def _warmup_kernels(self):
        """Pre-compile CUDA kernels"""
        print("   [INIT] Warming up CUDA Kernels (V25.0)...")
        try:
            h, w = 64, 64
            dummy_src = cp.zeros((h, w, 3), dtype=cp.uint8)
            dummy_dst = cp.zeros_like(dummy_src)
            dummy_mask = cp.zeros((h, w), dtype=cp.uint8)
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

            # V25.0 kernel warmup
            self.skin_v2_kernel(
                (2, 2), (32, 32),
                (dummy_src, dummy_src, dummy_dst, dummy_mask,
                 w, h, cp.float32(0.5), 4, cp.float32(0.04),
                 cp.float32(128), cp.float32(128), cp.float32(128),
                 cp.float32(0.0))
            )

            cp.cuda.Stream.null.synchronize()
            print("   [INIT] All Kernels Compiled (Legacy + V25.0)")
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
    # V25.0 Advanced Skin Processing
    # ==========================================================================
    def _process_skin_v25(self, frame_gpu, landmarks, params):
        """
        V25.0 Advanced skin processing pipeline

        1. Generate polygon mask
        2. Extract low-frequency (Guided Filter)
        3. Apply tone uniformity (Flat-fielding)
        4. Apply color grading
        5. Final skin smoothing with Guided Filter
        """
        h, w = frame_gpu.shape[:2]

        # Get parameters
        skin_strength = params.get('skin_smooth', 0.0)
        skin_tone_val = params.get('skin_tone', 0.0)
        flatten_strength = params.get('flatten_strength', 0.3)
        detail_preserve = params.get('detail_preserve', 0.7)
        gf_radius = params.get('gf_radius', 8)
        gf_epsilon = params.get('gf_epsilon', 0.04)
        temperature = params.get('color_temperature', 0.0)
        tint = params.get('color_tint', 0.0)

        # Skip if no processing needed
        if skin_strength < 0.01 and abs(skin_tone_val) < 0.01 and abs(temperature) < 0.01 and abs(tint) < 0.01:
            return frame_gpu, None

        # Step 1: Generate polygon mask
        skin_mask_gpu = self.mask_manager.generate_mask(landmarks, w, h, feather_radius=3)
        if skin_mask_gpu is None:
            return frame_gpu, None

        # Calculate mean skin color from mask region
        mask_np = cp.asnumpy(skin_mask_gpu)
        frame_np = cp.asnumpy(frame_gpu) if hasattr(frame_gpu, 'get') else frame_gpu.get()
        skin_pixels = frame_np[mask_np > 128]
        if len(skin_pixels) > 0:
            mean_color = np.median(skin_pixels, axis=0)  # B, G, R
        else:
            mean_color = np.array([128.0, 128.0, 128.0])

        current_frame = frame_gpu

        # Step 2: Frequency separation + Tone uniformity (if flatten_strength > 0)
        if flatten_strength > 0.01 and skin_strength > 0.01:
            low_freq = self.freq_separator.extract_low_frequency(
                current_frame, skin_mask_gpu, radius=gf_radius, epsilon=gf_epsilon
            )
            current_frame = self.freq_separator.apply_tone_uniformity(
                current_frame, low_freq, skin_mask_gpu,
                mean_color, flatten_strength, detail_preserve
            )

        # Step 3: Color grading
        if abs(temperature) > 0.01 or abs(tint) > 0.01:
            current_frame = self.color_grader.apply(
                current_frame, skin_mask_gpu,
                temperature, tint, use_mask=True
            )

        # Step 4: Final skin smoothing with V2 kernel
        if skin_strength > 0.01:
            result_gpu = cp.empty_like(current_frame)

            block_dim = (16, 16)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])

            self.skin_v2_kernel(
                grid_dim, block_dim,
                (current_frame, current_frame, result_gpu, skin_mask_gpu,
                 w, h, cp.float32(skin_strength),
                 gf_radius, cp.float32(gf_epsilon),
                 cp.float32(mean_color[2]), cp.float32(mean_color[1]), cp.float32(mean_color[0]),
                 cp.float32(skin_tone_val))
            )
            current_frame = result_gpu

        return current_frame, skin_mask_gpu

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
    def process(self, frame, faces, body_landmarks=None, params=None, mask=None):
        """
        Main processing pipeline

        :param frame: Input frame (CPU or GPU)
        :param faces: List of FaceResult from FaceMesh
        :param body_landmarks: Body keypoints from pose tracker
        :param params: Processing parameters dict
        :param mask: Alpha mask for compositing
        :return: Processed frame
        """
        if frame is None or not HAS_CUDA:
            return frame
        if params is None:
            params = {}

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

                # Choose pipeline based on setting
                if self.use_v25_pipeline:
                    source_for_warp, skin_mask_debug = self._process_skin_v25(
                        frame_gpu, stable_face, params
                    )
                else:
                    source_for_warp = self._process_skin_legacy(
                        frame_gpu, stable_face, params, w, h
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
                mode_text = "V25.0 Pipeline" if self.use_v25_pipeline else "Legacy Pipeline"
                cv2.putText(debug_img, mode_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                result_gpu = cp.asarray(debug_img)

            # Return result
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()

    # ==========================================================================
    # Pipeline Mode Control
    # ==========================================================================
    def set_pipeline_mode(self, use_v25=True):
        """
        Switch between V25.0 and legacy pipeline

        :param use_v25: True for V25.0, False for legacy
        """
        self.use_v25_pipeline = use_v25
        mode = "V25.0 High-Precision" if use_v25 else "Legacy Bilateral"
        print(f"[BEAUTY] Pipeline Mode: {mode}")
