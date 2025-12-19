# Project MUSE - beauty_engine.py
# V36.0: Warp Grid Based Mask - 근본적 재설계
# - [V36] 워핑 그리드에서 마스크 직접 생성 (시간 동기화 문제 근본 해결)
# - [V36] AI 마스크는 use_bg 플래그 결정에만 사용
# - [V36] 배경에 src 절대 미사용으로 휘어짐 완전 방지
# - [V36] 단순화된 2레이어 합성: 워핑된 사람 + 정적 배경
# - Preserved: V34 Grid Modulation, V33 FrameSyncBuffer, V31 Dual-Pass
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os
import time
from ai.tracking.facemesh import FaceMesh

# Import Kernels & Logic
from graphics.kernels.cuda_kernels import (
    WARP_KERNEL_CODE, COMPOSITE_KERNEL_CODE, SKIN_SMOOTH_KERNEL_CODE,
    BILATERAL_SMOOTH_KERNEL_CODE, GPU_RESIZE_KERNEL_CODE,
    GPU_MASK_RESIZE_KERNEL_CODE, FINAL_BLEND_KERNEL_CODE,
    # V4: Forward Mask based Composite
    FORWARD_WARP_MASK_KERNEL_CODE, MASK_DILATE_KERNEL_CODE, SIMPLE_COMPOSITE_KERNEL_CODE,
    # V5: Void Fill Composite (동기화된 트리플 레이어)
    VOID_FILL_COMPOSITE_KERNEL_CODE,
    # V29: High-Fidelity Skin Smoothing (Guided Filter + LAB Color Space)
    GUIDED_FILTER_KERNEL_CODE, FAST_SKIN_SMOOTH_KERNEL_CODE, LAB_SKIN_SMOOTH_KERNEL_CODE,
    # V31: Dual-Pass Smooth Kernel (Wide/Fine 합성)
    DUAL_PASS_SMOOTH_KERNEL_CODE,
    # V34: Background Warp Prevention & Clean 2-Layer Composite
    MODULATE_DISPLACEMENT_KERNEL_CODE, LAYERED_COMPOSITE_KERNEL_CODE,
    # V36: Warp Grid Based Mask (근본적 재설계)
    WARP_MASK_FROM_GRID_KERNEL_CODE, CLEAN_COMPOSITE_KERNEL_CODE
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
# [V33 신규 클래스] FrameSyncBuffer - AI 결과와 프레임 시점 동기화
# - AI 마스크 생성 지연(1~3프레임)만큼 원본 프레임을 버퍼링
# - 마스크가 생성된 시점의 프레임과 정확히 매칭하여 잔상 원천 차단
# ==============================================================================
class FrameSyncBuffer:
    """
    Frame Synchronization Buffer for AI latency compensation.

    Problem: AI mask has 1-3 frame latency, causing "trailing ghost" artifacts.
    Solution: Buffer original frames and match them with AI results by timestamp.

    [V33] 프레임-마스크 시점 동기화로 Trailing 현상 근본 해결
    """
    def __init__(self, max_size=3):
        """
        Args:
            max_size: Maximum number of frames to buffer (default: 3)
                      Higher = more latency tolerance, more VRAM usage (~15MB/frame)
        """
        self.max_size = max_size
        self.buffer = []  # List of (frame_id, frame_gpu) tuples
        self.frame_counter = 0

    def push(self, frame_gpu):
        """
        Add a new frame to the buffer.

        Args:
            frame_gpu: Current frame (CuPy array)
        Returns:
            frame_id: Unique identifier for this frame
        """
        self.frame_counter += 1
        frame_id = self.frame_counter

        # Store a copy to prevent external modifications
        if HAS_CUDA:
            frame_copy = frame_gpu.copy()
        else:
            frame_copy = frame_gpu.copy() if hasattr(frame_gpu, 'copy') else frame_gpu

        self.buffer.append((frame_id, frame_copy))

        # Remove oldest frame if buffer is full
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

        return frame_id

    def get_synced_frame(self, delay_frames=1):
        """
        Get the frame that was captured 'delay_frames' ago.

        Args:
            delay_frames: How many frames back to retrieve (1 = previous frame)
        Returns:
            Synced frame or None if not available
        """
        if len(self.buffer) == 0:
            return None

        # Calculate target index (0 = oldest, -1 = newest)
        target_idx = len(self.buffer) - 1 - delay_frames

        if target_idx < 0:
            # Not enough frames buffered yet, return oldest available
            return self.buffer[0][1]

        return self.buffer[target_idx][1]

    def get_latest(self):
        """Get the most recent frame in buffer."""
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1][1]

    def reset(self):
        """Clear all buffered frames."""
        self.buffer.clear()
        self.frame_counter = 0

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# [V28.0 신규 클래스] MaskStabilizer - 마스크 시간 동기화
# - 워핑 그리드와 마스크의 시간적 불일치(번개 현상) 해결
# - LandmarkStabilizer와 유사한 지연으로 마스크 안정화
# ==============================================================================
class MaskStabilizer:
    """
    Mask Temporal Stabilizer for synchronizing mask with warping grid.

    Problem: Warping grid has latency from LandmarkStabilizer, but mask is realtime.
    Solution: Apply similar smoothing to mask so they move together.
    """
    def __init__(self, alpha=0.15):
        """
        Args:
            alpha: Smoothing factor (0.0 = full lag, 1.0 = no smoothing)
                   [V31] 0.15로 조정하여 WarpGridStabilizer와 동기화
        """
        self.alpha = alpha
        self.prev_mask = None

    def update(self, mask_gpu):
        """
        Apply temporal smoothing to mask.

        Args:
            mask_gpu: Current frame mask (CuPy array, uint8)
        Returns:
            Stabilized mask (CuPy array, uint8)
        """
        if self.prev_mask is None or self.prev_mask.shape != mask_gpu.shape:
            self.prev_mask = mask_gpu.astype(cp.float32)
            return mask_gpu

        # Exponential moving average
        curr_float = mask_gpu.astype(cp.float32)
        stabilized = self.alpha * curr_float + (1.0 - self.alpha) * self.prev_mask
        self.prev_mask = stabilized

        return stabilized.astype(cp.uint8)

    def reset(self):
        """Reset stabilizer state"""
        self.prev_mask = None


# ==============================================================================
# [V33 업그레이드] WarpGridStabilizer - 동적 적응형 EMA
# - 마스크 안정화만으로 부족했던 잔상 문제를 워핑 그리드(dx, dy) 자체에서 해결
# - [V33] 움직임 속도에 따라 alpha 동적 조절 (느림: 0.15, 빠름: 0.8)
# - "뒤늦게 따라오는 느낌" 제거를 위한 적응형 스냅 로직
# ==============================================================================
class WarpGridStabilizer:
    """
    Warp Grid Temporal Stabilizer with Adaptive EMA.

    Problem: Fixed alpha causes "lagging behind" feeling during fast movements.
    Solution: Dynamically adjust alpha based on grid velocity.

    [V33] 동적 alpha로 빠른 움직임에는 즉각 반응, 느린 움직임에는 안정화
    - 정지/느림: alpha=0.15 (강한 안정화)
    - 빠른 움직임: alpha=0.8 (즉각 스냅)
    """
    def __init__(self, base_alpha=0.15, snap_alpha=0.8, velocity_threshold=3.0):
        """
        Args:
            base_alpha: Base smoothing factor for slow movements (default: 0.15)
            snap_alpha: Snap factor for fast movements (default: 0.8)
            velocity_threshold: Grid velocity threshold for snap (pixels/frame)
        """
        self.base_alpha = base_alpha
        self.snap_alpha = snap_alpha
        self.velocity_threshold = velocity_threshold
        self.prev_dx = None
        self.prev_dy = None

    def update(self, dx_gpu, dy_gpu):
        """
        Apply adaptive temporal smoothing to warp grid.

        Args:
            dx_gpu: Current frame dx grid (CuPy array, float32)
            dy_gpu: Current frame dy grid (CuPy array, float32)
        Returns:
            Tuple of stabilized (dx, dy) grids
        """
        if self.prev_dx is None or self.prev_dx.shape != dx_gpu.shape:
            self.prev_dx = dx_gpu.copy()
            self.prev_dy = dy_gpu.copy()
            return dx_gpu, dy_gpu

        # [V33] 그리드 변화량(속도) 계산
        delta_dx = dx_gpu - self.prev_dx
        delta_dy = dy_gpu - self.prev_dy

        # 전체 그리드의 평균 속도 계산
        velocity = float(cp.mean(cp.sqrt(delta_dx**2 + delta_dy**2)))

        # [V33] 속도 기반 동적 alpha 계산
        # velocity가 threshold 이상이면 snap_alpha로 전환 (즉각 반응)
        # velocity가 낮으면 base_alpha 유지 (강한 안정화)
        if velocity > self.velocity_threshold:
            # 빠른 움직임: 점진적으로 snap_alpha로 전환
            speed_factor = min((velocity - self.velocity_threshold) / self.velocity_threshold, 1.0)
            adaptive_alpha = self.base_alpha + (self.snap_alpha - self.base_alpha) * speed_factor
        else:
            adaptive_alpha = self.base_alpha

        # Exponential moving average with adaptive alpha
        stabilized_dx = adaptive_alpha * dx_gpu + (1.0 - adaptive_alpha) * self.prev_dx
        stabilized_dy = adaptive_alpha * dy_gpu + (1.0 - adaptive_alpha) * self.prev_dy

        self.prev_dx = stabilized_dx.copy()
        self.prev_dy = stabilized_dy.copy()

        return stabilized_dx, stabilized_dy

    def reset(self):
        """Reset stabilizer state"""
        self.prev_dx = None
        self.prev_dy = None


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

    def generate_mask(self, landmarks, w, h, padding_ratio=1.15, exclude_features=False):
        """
        Generate skin mask using OpenCV (CPU, very fast < 2ms)

        Args:
            landmarks: Face landmarks array
            w, h: Image dimensions
            padding_ratio: Padding for exclusion zones
            exclude_features: If False, no exclusion zones (YY-style)
                            If True, exclude eyes/brows/lips (legacy)
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

        # 2-4. Exclude features only if requested (legacy mode)
        # YY-Style: Bilateral filter handles edge preservation automatically
        if exclude_features:
            self._exclude_region(mask, landmarks, self.EYE_L_INDICES, padding_ratio * 1.3)
            self._exclude_region(mask, landmarks, self.EYE_R_INDICES, padding_ratio * 1.3)
            self._exclude_region(mask, landmarks, self.BROW_L_INDICES, padding_ratio * 1.1)
            self._exclude_region(mask, landmarks, self.BROW_R_INDICES, padding_ratio * 1.1)
            self._exclude_region(mask, landmarks, self.LIPS_INDICES, padding_ratio * 1.2)

        # 5. Soft edge - increased sigma for smoother blending (5 -> 17)
        mask = cv2.GaussianBlur(mask, (35, 35), 17)
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
# [메인 클래스] BeautyEngine - V36.0 Warp Grid Based Mask
# ==============================================================================
class BeautyEngine:
    """
    V36.0 Beauty Processing Engine - Warp Grid Based Mask

    Key Changes (V36.0):
    - 워핑 그리드에서 마스크 직접 생성 (시간 동기화 문제 근본 해결)
    - AI 마스크와 워핑 그리드의 "시점 불일치" 문제 완전 해결
    - 배경에 src 절대 미사용으로 휘어짐 완전 방지
    - 단순화된 2레이어 합성: 워핑된 사람 + 정적 배경

    Preserved from V34.0:
    - Grid Modulation: 인물 마스크로 dx/dy 변위 그리드 마스킹

    Preserved from V33.0:
    - FrameSyncBuffer: AI 마스크 지연 보상을 위한 프레임 버퍼링
    - Adaptive WarpGridStabilizer: 속도 기반 동적 alpha

    Result: 일렁임 완전 해결 + 배경 휘어짐 방지 + 예측 가능한 동작
    """

    def __init__(self, profiles=[]):
        print("[BEAUTY] [BeautyEngine] V36.0 Warp Grid Based Mask Ready")
        self.map_scale = 0.25
        self.cache_w = 0
        self.cache_h = 0
        self.gpu_initialized = False

        self.gpu_dx = None
        self.gpu_dy = None

        # Stabilizers (기존 유지)
        self.body_stabilizer = LandmarkStabilizer(min_cutoff=0.01, base_beta=1.0, high_speed_beta=100.0)
        self.face_stabilizer = LandmarkStabilizer(min_cutoff=0.5, base_beta=5.0, high_speed_beta=50.0)

        # V28.0: Mask Stabilizer (마스크-워핑 시간 동기화)
        self.mask_stabilizer = None  # Initialized when HAS_CUDA

        # V33: Warp Grid Stabilizer with Adaptive EMA (동적 alpha)
        self.warp_grid_stabilizer = None  # Initialized when HAS_CUDA

        # V33: Frame Sync Buffer (AI 지연 보상)
        self.frame_sync_buffer = None  # Initialized when HAS_CUDA
        self.ai_latency_frames = 1  # AI 마스크 지연 프레임 수 (조절 가능)

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

        # V29/V31: High-Fidelity processing buffers (원본 해상도, 다운스케일 제거)
        self.guided_result = None  # Guided Filter 결과 (저주파)
        self.freq_sep_result = None  # Frequency Separation 결과
        self.v29_frame_count = 0  # 디버그 로그용 프레임 카운터

        # V31: Dual-Pass 피부 보정 버퍼
        self.wide_smooth_result = None  # Wide Pass 결과 (radius 15)
        self.fine_smooth_result = None  # Fine Pass 결과 (radius 5)

        # V4: Forward Mask buffers (순방향 마스크 기반 합성)
        self.forward_mask_gpu = None
        self.forward_mask_dilated_gpu = None

        # V28.0: Stabilized mask buffer (동기화된 마스크)
        self.stabilized_mask_gpu = None

        # V36: Warp Grid Based Mask buffers
        self.warp_grid_mask_gpu = None           # 워핑 강도 기반 마스크
        self.warp_grid_mask_forward_gpu = None   # 순방향 워핑된 마스크

        if HAS_CUDA:
            # V33: Initialize Stabilizers with adaptive alpha
            self.mask_stabilizer = MaskStabilizer(alpha=0.15)
            # [V33] 동적 alpha: 느린 움직임 0.15, 빠른 움직임 0.8
            self.warp_grid_stabilizer = WarpGridStabilizer(
                base_alpha=0.15, snap_alpha=0.8, velocity_threshold=3.0
            )
            # [V33] Frame Sync Buffer for AI latency compensation
            self.frame_sync_buffer = FrameSyncBuffer(max_size=3)
            self.stream = cp.cuda.Stream(non_blocking=True)
            # Core kernels (warping, compositing)
            self.warp_kernel = cp.RawKernel(WARP_KERNEL_CODE, 'warp_kernel')
            self.composite_kernel = cp.RawKernel(COMPOSITE_KERNEL_CODE, 'composite_kernel')
            self.skin_kernel = cp.RawKernel(SKIN_SMOOTH_KERNEL_CODE, 'skin_smooth_kernel')

            # YY-Style kernels (bilateral smoothing) - Legacy
            self.bilateral_kernel = cp.RawKernel(BILATERAL_SMOOTH_KERNEL_CODE, 'bilateral_smooth_kernel')
            self.resize_kernel = cp.RawKernel(GPU_RESIZE_KERNEL_CODE, 'gpu_resize_kernel')
            self.mask_resize_kernel = cp.RawKernel(GPU_MASK_RESIZE_KERNEL_CODE, 'gpu_mask_resize_kernel')
            self.blend_kernel = cp.RawKernel(FINAL_BLEND_KERNEL_CODE, 'final_blend_kernel')

            # V29: High-Fidelity kernels (Guided Filter + LAB Color Space)
            self.guided_filter_kernel = cp.RawKernel(GUIDED_FILTER_KERNEL_CODE, 'guided_filter_kernel')
            self.freq_separation_kernel = cp.RawKernel(FAST_SKIN_SMOOTH_KERNEL_CODE, 'fast_skin_smooth_kernel')
            self.lab_smooth_kernel = cp.RawKernel(LAB_SKIN_SMOOTH_KERNEL_CODE, 'lab_skin_smooth_kernel')

            # V31: Dual-Pass Smooth kernel (Wide/Fine 합성)
            self.dual_pass_smooth_kernel = cp.RawKernel(DUAL_PASS_SMOOTH_KERNEL_CODE, 'dual_pass_smooth_kernel')

            # V4: Forward Mask based Composite (순방향 마스크 기반 합성)
            self.forward_warp_mask_kernel = cp.RawKernel(FORWARD_WARP_MASK_KERNEL_CODE, 'forward_warp_mask_kernel')
            self.mask_dilate_kernel = cp.RawKernel(MASK_DILATE_KERNEL_CODE, 'mask_dilate_kernel')
            self.simple_composite_kernel = cp.RawKernel(SIMPLE_COMPOSITE_KERNEL_CODE, 'simple_composite_kernel')

            # V28.0: Void Fill Composite (동기화된 트리플 레이어)
            self.void_fill_composite_kernel = cp.RawKernel(VOID_FILL_COMPOSITE_KERNEL_CODE, 'void_fill_composite_kernel')

            # V34: Background Warp Prevention & Clean 2-Layer Composite
            self.modulate_displacement_kernel = cp.RawKernel(MODULATE_DISPLACEMENT_KERNEL_CODE, 'modulate_displacement_kernel')
            self.layered_composite_kernel = cp.RawKernel(LAYERED_COMPOSITE_KERNEL_CODE, 'layered_composite_kernel')

            # V36: Warp Grid Based Mask (근본적 재설계)
            self.warp_mask_from_grid_kernel = cp.RawKernel(WARP_MASK_FROM_GRID_KERNEL_CODE, 'warp_mask_from_grid_kernel')
            self.clean_composite_kernel = cp.RawKernel(CLEAN_COMPOSITE_KERNEL_CODE, 'clean_composite_kernel')

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
    # V31: Dual-Pass High-Fidelity Skin Smoothing
    # - Wide Pass (radius 15, epsilon 0.02): 피부톤 전체를 도자기처럼 균일화
    # - Fine Pass (radius 5, epsilon 0.008): 미세 디테일 보존
    # - 두 결과를 비선형 곡선(pow 0.3)으로 합성하여 엣지 보존 극대화
    # ==========================================================================
    def _process_skin_yy_style(self, frame_gpu, landmarks, params):
        """
        V31: Dual-Pass High-Fidelity skin smoothing

        Key improvements over V30 (Single-Pass):
        - Wide Pass: 넓은 반경으로 피부톤 전체를 균일화 (도자기 질감)
        - Fine Pass: 좁은 반경으로 눈썹/속눈썹 등 미세 디테일 유지
        - Non-linear Detail Curve: powf(edge_factor, 0.3f)로 엣지 복원력 기하급수적 강화
        - Contrast Boost: 1.05f로 투명감 있는 피부 표현

        Result: 피부는 매끈, 눈코입은 초선명
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

        # Generate skin mask (no exclusion zones - Dual-Pass가 엣지 자동 보존)
        skin_mask_gpu = self.mask_manager.generate_mask(
            landmarks, w, h, padding_ratio=1.25, exclude_features=False
        )

        if skin_mask_gpu is None:
            return frame_gpu, None

        # 디버그 로그 (60프레임마다 1회)
        self.v29_frame_count += 1
        should_log = (self.v29_frame_count % 60 == 1)

        # ===== V31 Dual-Pass Skin Smoothing =====
        if has_skin:
            # 원본 평균 저장 (디버그용)
            orig_mean = float(cp.mean(frame_gpu)) if should_log else 0

            # Initialize or resize buffers (원본 해상도)
            if (self.wide_smooth_result is None or
                self.wide_smooth_result.shape[0] != h or
                self.wide_smooth_result.shape[1] != w):
                self.wide_smooth_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                self.fine_smooth_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                self.freq_sep_result = cp.zeros((h, w, 3), dtype=cp.uint8)
                print(f"[V31] Dual-Pass 버퍼 초기화: {w}x{h}")

            block_dim = (16, 16)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])

            # ============================================
            # [Step 1] Wide Pass: 넓은 반경으로 피부톤 균일화
            # - radius 15, epsilon 0.02
            # - 피부 전체를 부드럽게 밀어버림 (도자기 효과)
            # ============================================
            wide_radius = 15
            wide_epsilon = 0.02

            try:
                self.guided_filter_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, frame_gpu, self.wide_smooth_result, skin_mask_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.int32(wide_radius),
                     cp.float32(wide_epsilon))
                )
            except Exception as e:
                print(f"[V31 ERROR] Wide Pass 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # ============================================
            # [Step 2] Fine Pass: 좁은 반경으로 디테일 보존
            # - radius 5, epsilon 0.008
            # - 속눈썹, 눈동자 등 미세 엣지 유지
            # ============================================
            fine_radius = 5
            fine_epsilon = 0.008

            try:
                self.guided_filter_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, frame_gpu, self.fine_smooth_result, skin_mask_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.int32(fine_radius),
                     cp.float32(fine_epsilon))
                )
            except Exception as e:
                print(f"[V31 ERROR] Fine Pass 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # ============================================
            # [Step 3] Dual-Pass 합성 (비선형 디테일 곡선)
            # - Wide/Fine 결과를 입력받아 합성
            # - powf(edge_factor, 0.3f)로 엣지 영역 복원력 강화
            # - Contrast 부스팅 1.05f로 투명감 표현
            # ============================================
            blend_strength = min(skin_strength * 1.5, 0.90)  # 최대 90%

            try:
                self.dual_pass_smooth_kernel(
                    grid_dim, block_dim,
                    (frame_gpu, self.wide_smooth_result, self.fine_smooth_result,
                     skin_mask_gpu, self.freq_sep_result,
                     cp.int32(w), cp.int32(h),
                     cp.float32(skin_strength),
                     cp.float32(blend_strength))
                )
            except Exception as e:
                print(f"[V31 ERROR] Dual-Pass Smooth 실패: {e}")
                return frame_gpu, skin_mask_gpu

            # Step 4: Color Grading 적용 (선택적)
            if has_color:
                result_gpu = cp.empty_like(frame_gpu)
                self.blend_kernel(
                    grid_dim, block_dim,
                    (self.freq_sep_result, self.freq_sep_result, skin_mask_gpu, result_gpu,
                     cp.int32(w), cp.int32(h),
                     cp.float32(0.0),
                     cp.float32(color_temp),
                     cp.float32(color_tint))
                )
                frame_gpu = result_gpu
            else:
                frame_gpu = cp.asarray(self.freq_sep_result)

            # 디버그 로그 (60프레임마다)
            if should_log:
                result_mean = float(cp.mean(frame_gpu))
                diff = abs(result_mean - orig_mean)
                print(f"[V31-DP] str={skin_strength:.2f}, blend={blend_strength:.2f}, diff={diff:.2f}")

        # ===== Skin Tone / Color Only (No Smoothing) =====
        elif has_tone or has_color:
            result_gpu = cp.empty_like(frame_gpu)
            grid_dim = ((w + 15) // 16, (h + 15) // 16)

            self.blend_kernel(
                grid_dim, (16, 16),
                (frame_gpu, frame_gpu, skin_mask_gpu, result_gpu,
                 w, h,
                 cp.float32(0.0),  # No smoothing blend
                 cp.float32(color_temp),
                 cp.float32(color_tint))
            )
            frame_gpu = result_gpu

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
    # V36: Warp Grid Based Mask Generation
    # ==========================================================================
    def _generate_warp_mask_from_grid(self, w, h, sw, sh, scale, threshold=0.5):
        """
        [V36] 워핑 그리드에서 마스크 생성

        워핑 강도(|dx|+|dy|)가 threshold 이상인 영역을 마스크로 추출.
        워핑 그리드와 완벽히 동기화되어 시간 지연 문제 해결.

        Args:
            w, h: 전체 해상도
            sw, sh: 워핑 그리드 해상도
            scale: 스케일 배율
            threshold: 워핑 강도 임계값 (픽셀 단위)

        Returns:
            warp_mask_gpu: 워핑 영역 마스크 (CuPy array, uint8)
        """
        # 버퍼 초기화
        if self.warp_grid_mask_gpu is None or self.warp_grid_mask_gpu.shape != (h, w):
            self.warp_grid_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
            self.warp_grid_mask_forward_gpu = cp.zeros((h, w), dtype=cp.uint8)

        block_dim = (16, 16)
        grid_dim = ((w + 15) // 16, (h + 15) // 16)

        # Step 1: 워핑 그리드에서 마스크 생성
        self.warp_mask_from_grid_kernel(
            grid_dim, block_dim,
            (self.gpu_dx, self.gpu_dy, self.warp_grid_mask_gpu,
             w, h, sw, sh, scale, cp.float32(threshold))
        )

        # Step 2: 가우시안 블러로 경계 부드럽게
        smoothed = cupyx.scipy.ndimage.gaussian_filter(
            self.warp_grid_mask_gpu.astype(cp.float32), sigma=3.0
        )
        self.warp_grid_mask_gpu = cp.clip(smoothed, 0, 255).astype(cp.uint8)

        # Step 3: 순방향 워핑으로 최종 사람 영역 계산
        self.warp_grid_mask_forward_gpu.fill(0)
        self.forward_warp_mask_kernel(
            grid_dim, block_dim,
            (self.warp_grid_mask_gpu, self.warp_grid_mask_forward_gpu,
             self.gpu_dx, self.gpu_dy,
             w, h, sw, sh, scale)
        )

        # Step 4: 홀 채우기 (Dilate)
        temp_mask = cp.zeros_like(self.warp_grid_mask_forward_gpu)
        self.mask_dilate_kernel(
            grid_dim, block_dim,
            (self.warp_grid_mask_forward_gpu, temp_mask,
             w, h, 2)
        )

        # Step 5: 최종 블러
        final_mask = cupyx.scipy.ndimage.gaussian_filter(
            temp_mask.astype(cp.float32), sigma=2.0
        )
        final_mask = cp.clip(final_mask, 0, 255).astype(cp.uint8)

        return final_mask

    # ==========================================================================
    # Main Processing Pipeline
    # ==========================================================================
    def process(self, frame, faces, body_landmarks=None, params=None, mask=None, frame_cpu=None, bg_stable=False):
        """
        Main processing pipeline

        :param frame: Input frame (CPU or GPU)
        :param faces: List of FaceResult from FaceMesh
        :param body_landmarks: Body keypoints from pose tracker
        :param params: Processing parameters dict
        :param mask: Alpha mask for compositing
        :param frame_cpu: Optional CPU frame to avoid GPU->CPU transfer
        :param bg_stable: [V6] 정적 배경 안정성 플래그 (True: 슬리밍 합성 가능)
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
                # V28.0: Reset MaskStabilizer on size change
                if self.mask_stabilizer is not None:
                    self.mask_stabilizer.reset()
                # V31: Reset WarpGridStabilizer on size change
                if self.warp_grid_stabilizer is not None:
                    self.warp_grid_stabilizer.reset()
                # V33: Reset FrameSyncBuffer on size change
                if self.frame_sync_buffer is not None:
                    self.frame_sync_buffer.reset()

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

            # ==================================================================
            # [V35] Frame Synchronization - 잔상 제거의 핵심
            # - 현재 프레임을 버퍼에 저장
            # - 모든 연산(워핑, 마스크, 합성)을 동기화된 과거 프레임으로 통일
            # - 이것이 잔상(Ghosting) 해결의 핵심
            # ==================================================================
            synced_frame = frame_gpu  # 기본값: 현재 프레임

            if self.frame_sync_buffer is not None:
                # 현재 프레임을 버퍼에 저장
                self.frame_sync_buffer.push(frame_gpu)

                # AI 마스크가 있을 때 동기화된 프레임 사용
                if mask is not None and bg_stable:
                    # AI 지연만큼 과거 프레임 가져오기
                    _synced = self.frame_sync_buffer.get_synced_frame(
                        delay_frames=self.ai_latency_frames
                    )
                    if _synced is not None:
                        synced_frame = _synced

            # Mask handling
            # [V6] 정적 배경이 안정적일 때만 슬리밍 합성 활성화 (일렁임 방지)
            if self.has_bg and mask is not None and bg_stable:
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
            # [V35] 동기화된 프레임(synced_frame)을 베이스로 사용
            # ==================================================================
            source_for_warp = synced_frame
            skin_mask_debug = None

            if faces:
                raw_face = faces[0].landmarks
                stable_face = self.face_stabilizer.update(raw_face)

                # YY-Style Pipeline (Bilateral Filter based)
                # [V35] synced_frame을 입력으로 사용하여 시점 통일
                source_for_warp, skin_mask_debug = self._process_skin_yy_style(
                    synced_frame, stable_face, params
                )

            # ==================================================================
            # [Step 2] Morph Logic (기존 100% 유지)
            # ==================================================================
            self.morph_logic.clear()

            # [V30] bg_stable 감쇄 계수: 배경이 불안정하면 슬리밍 강도 50% 감쇄
            slim_damping = 1.0 if bg_stable else 0.5

            raw_body = body_landmarks.get() if hasattr(body_landmarks, 'get') else body_landmarks
            if raw_body is not None:
                kpts_xy = raw_body[:, :2]
                stable_kpts = self.body_stabilizer.update(kpts_xy)
                scaled_body = stable_kpts * self.map_scale

                # [V30] 슬리밍 파라미터에 damping 적용
                if params.get('shoulder_narrow', 0) > 0:
                    self.morph_logic.collect_shoulder_params(scaled_body, params['shoulder_narrow'] * slim_damping)
                if params.get('ribcage_slim', 0) > 0:
                    self.morph_logic.collect_ribcage_params(scaled_body, params['ribcage_slim'] * slim_damping)
                if params.get('waist_slim', 0) > 0:
                    self.morph_logic.collect_waist_params(scaled_body, params['waist_slim'] * slim_damping)
                if params.get('hip_widen', 0) > 0:
                    self.morph_logic.collect_hip_params(scaled_body, params['hip_widen'] * slim_damping)

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

                # [V31] 그리드 블러 강화: sigma 1.0 → 2.0 (경계면 부드러움 향상)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dx, sigma=2.0, output=self.gpu_dx)
                cupyx.scipy.ndimage.gaussian_filter(self.gpu_dy, sigma=2.0, output=self.gpu_dy)

                # [V31] WarpGridStabilizer로 시간적 평활화 (잔상 근본 해결)
                if self.warp_grid_stabilizer is not None:
                    self.gpu_dx, self.gpu_dy = self.warp_grid_stabilizer.update(self.gpu_dx, self.gpu_dy)

                # ==============================================================
                # [V34] Grid Modulation - 배경 워핑 원천 차단
                # - 인물 마스크를 small scale로 리사이즈
                # - dx, dy 그리드에 마스크를 곱하여 배경 영역 변위 = 0
                # ==============================================================
                if use_bg and mask_gpu is not None:
                    # 마스크를 small scale로 리사이즈
                    mask_small_gpu = cupyx.scipy.ndimage.zoom(
                        mask_gpu.astype(cp.float32),
                        (sh / h, sw / w),
                        order=1
                    ).astype(cp.uint8)

                    # [주의사항] 마스크 경계면 블러링 (너무 칼같으면 인물 외곽 잘림 방지)
                    mask_small_blurred = cupyx.scipy.ndimage.gaussian_filter(
                        mask_small_gpu.astype(cp.float32), sigma=2.0
                    ).astype(cp.uint8)

                    # 그리드 모듈레이션 적용
                    small_block_dim = (16, 16)
                    small_grid_dim = ((sw + small_block_dim[0] - 1) // small_block_dim[0],
                                      (sh + small_block_dim[1] - 1) // small_block_dim[1])

                    self.modulate_displacement_kernel(
                        small_grid_dim, small_block_dim,
                        (self.gpu_dx, self.gpu_dy, mask_small_blurred, sw, sh)
                    )

            # ==================================================================
            # [Step 4] V36: Void Only Fill 합성
            # - 기본: 워핑된 원본 프레임
            # - Void 영역만: 저장된 배경으로 패치 (슬리밍으로 비어진 공간만)
            # - 배경 전체 교체 X, Void만 채움 O
            # ==================================================================
            block_dim = (16, 16)
            grid_dim = ((w + block_dim[0] - 1) // block_dim[0],
                        (h + block_dim[1] - 1) // block_dim[1])
            scale = int(1.0 / self.map_scale)

            # 4-1. 버퍼 초기화
            if self.forward_mask_gpu is None or self.forward_mask_gpu.shape != (h, w):
                self.forward_mask_gpu = cp.zeros((h, w), dtype=cp.uint8)
                self.forward_mask_dilated_gpu = cp.zeros((h, w), dtype=cp.uint8)

            # 4-2. 원본 마스크 준비 (mask_orig: 사람이 "있었던" 영역)
            # AI 마스크가 있으면 사용, 없으면 워핑 그리드 기반 마스크 생성
            if use_bg and mask_gpu is not None:
                # AI 마스크 경계 블러 처리
                mask_orig = cupyx.scipy.ndimage.gaussian_filter(
                    mask_gpu.astype(cp.float32), sigma=1.5
                ).astype(cp.uint8)
            else:
                # AI 마스크가 없으면 워핑 그리드에서 생성
                mask_orig = self._generate_warp_mask_from_grid(
                    w, h, sw, sh, scale, threshold=0.5
                )

            # 4-3. 순방향 워핑된 마스크 (mask_fwd: 사람이 "현재 있는" 영역)
            # 슬리밍으로 인해 수축된 영역
            self.forward_mask_gpu.fill(0)

            self.forward_warp_mask_kernel(
                grid_dim, block_dim,
                (mask_orig, self.forward_mask_gpu,
                 self.gpu_dx, self.gpu_dy,
                 w, h, sw, sh, scale)
            )

            # 4-4. 마스크 홀 채우기 (Dilate)
            self.mask_dilate_kernel(
                grid_dim, block_dim,
                (self.forward_mask_gpu, self.forward_mask_dilated_gpu,
                 w, h, 2)
            )

            # 4-5. 순방향 마스크 평활화
            mask_fwd = cupyx.scipy.ndimage.gaussian_filter(
                self.forward_mask_dilated_gpu.astype(cp.float32), sigma=1.5
            ).astype(cp.uint8)

            # 4-6. V36 Void Only Fill 합성
            # - mask_orig: 원래 사람 위치
            # - mask_fwd: 현재 사람 위치 (슬리밍 후)
            # - Void = mask_orig > 0 && mask_fwd == 0 인 영역만 배경으로 채움
            result_gpu = cp.empty_like(frame_gpu)

            self.clean_composite_kernel(
                grid_dim, block_dim,
                (source_for_warp, self.bg_gpu,
                 mask_orig, mask_fwd,
                 result_gpu,
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
                cv2.putText(debug_img, "V36 Void Only Fill", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                result_gpu = cp.asarray(debug_img)

            # Return result
            if is_gpu_input:
                return result_gpu
            else:
                return result_gpu.get()

