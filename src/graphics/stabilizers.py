# Project MUSE - stabilizers.py
# Temporal Stabilization Components for Beauty Processing Pipeline
# Extracted from beauty_engine.py for modular architecture
# (C) 2025 MUSE Corp. All rights reserved.

"""
Stabilizer modules for temporal smoothing of landmarks, masks, and warp grids.

This module contains:
- LandmarkStabilizer: One-Euro Filter style landmark stabilization
- MaskStabilizer: Multi-Frame Median Consensus for mask temporal smoothing
- WarpGridStabilizer: Hysteresis + Alpha Smoothing for warp grid stabilization
"""

import time
import numpy as np

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class LandmarkStabilizer:
    """
    One-Euro Filter style landmark stabilization.

    Provides adaptive low-pass filtering for landmark coordinates.
    Uses dynamic beta to balance smoothness and responsiveness based on movement speed.
    """

    def __init__(self, min_cutoff=0.01, base_beta=5.0, high_speed_beta=50.0, bypass=False):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            base_beta: Base beta for slow movements
            high_speed_beta: Maximum beta for fast movements
            bypass: [V44] If True, skip stabilization and return input directly
        """
        self.min_cutoff = min_cutoff
        self.base_beta = base_beta
        self.high_speed_beta = high_speed_beta
        self.prev_val = None
        self.prev_trend = None
        self.last_time = None
        self.bypass = bypass  # [V44] Frame-independent mode

    def update(self, val_array):
        """
        Update stabilizer with new landmark values.

        Args:
            val_array: New landmark coordinates (numpy array)
        Returns:
            Stabilized landmark coordinates
        """
        # [V44] bypass mode: return input directly (no stabilization)
        if self.bypass:
            return val_array

        now = time.time()
        if self.prev_val is None:
            self.prev_val = val_array
            self.prev_trend = np.zeros_like(val_array)
            self.last_time = now
            return val_array

        dt = now - self.last_time
        if dt <= 0:
            return self.prev_val
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
        """Reset stabilizer state."""
        self.prev_val = None
        self.prev_trend = None
        self.last_time = None


"""
[V37 Original - Preserved for Rollback]
class MaskStabilizer:
    # [V37] Mask Temporal Stabilizer with Multi-Frame Median Consensus.
    # - 5프레임 Median 합의 방식
    # - 문제점: 5프레임(0.17초) 지연으로 인한 잔상 발생

    def __init__(self, alpha=0.10, consensus_frames=5):
        self.alpha = alpha
        self.consensus_frames = consensus_frames
        self.prev_mask = None
        self.mask_history = []
        self.stacked_buffer = None

    def update(self, mask_gpu):
        if not HAS_CUDA:
            return mask_gpu
        if self.prev_mask is None or self.prev_mask.shape != mask_gpu.shape:
            self.prev_mask = mask_gpu.astype(cp.float32)
            self.mask_history = [mask_gpu.copy() for _ in range(self.consensus_frames)]
            h, w = mask_gpu.shape
            self.stacked_buffer = cp.zeros((self.consensus_frames, h, w), dtype=cp.uint8)
            return mask_gpu
        self.mask_history.pop(0)
        self.mask_history.append(mask_gpu.copy())
        for i, m in enumerate(self.mask_history):
            self.stacked_buffer[i] = m
        consensus_mask = cp.median(self.stacked_buffer, axis=0)
        curr_float = consensus_mask.astype(cp.float32)
        stabilized = self.alpha * curr_float + (1.0 - self.alpha) * self.prev_mask
        self.prev_mask = stabilized
        return stabilized.astype(cp.uint8)

    def reset(self):
        self.prev_mask = None
        self.mask_history = []
        self.stacked_buffer = None
"""


"""
[V37.1 Original - Preserved for Rollback]
class MaskStabilizer:
    # [V37.1] Mask Temporal Stabilizer - Simple EMA Only.
    # - 고정 alpha EMA 방식
    # - 문제점: 움직임에 관계없이 동일한 alpha로 잔상 발생

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.prev_mask = None

    def update(self, mask_gpu):
        if self.prev_mask is None or self.prev_mask.shape != mask_gpu.shape:
            self.prev_mask = mask_gpu.astype(cp.float32)
            return mask_gpu
        curr_float = mask_gpu.astype(cp.float32)
        stabilized = self.alpha * curr_float + (1.0 - self.alpha) * self.prev_mask
        self.prev_mask = stabilized
        return stabilized.astype(cp.uint8)

    def reset(self):
        self.prev_mask = None
V37.1 LEGACY */
"""


class MaskStabilizer:
    """
    [V40] Adaptive Mask Temporal Stabilizer

    움직임 속도(Diff)에 따라 alpha를 동적으로 조절:
    - 빠른 움직임: alpha=0.85 (즉각 반응 → 잔상 제거)
    - 정지 상태: alpha=0.12 (강한 스무딩 → 떨림 억제)
    """

    def __init__(self, base_alpha=0.12, fast_alpha=0.85, diff_threshold=0.04, bypass=False):
        """
        Args:
            base_alpha: 정지 시 alpha (강한 스무딩) - V40: 0.12
            fast_alpha: 움직임 시 alpha (즉각 반응) - V40: 0.85
            diff_threshold: 움직임 감지 임계값 - V40: 0.04
            bypass: [V44] If True, skip stabilization and return input directly
        """
        self.base_alpha = base_alpha
        self.fast_alpha = fast_alpha
        self.diff_threshold = diff_threshold
        self.prev_mask = None
        self.bypass = bypass  # [V44] Frame-independent mode

    def update(self, mask_gpu):
        """
        마스크 안정화 (적응형 EMA)

        Args:
            mask_gpu: 현재 프레임 마스크 (CuPy array)

        Returns:
            stabilized_mask: 안정화된 마스크
        """
        # [V44] bypass mode: return input directly (no stabilization)
        if self.bypass:
            return mask_gpu

        if self.prev_mask is None or self.prev_mask.shape != mask_gpu.shape:
            self.prev_mask = mask_gpu.astype(cp.float32)
            return mask_gpu

        curr_float = mask_gpu.astype(cp.float32)

        # [V39 Adaptive] 움직임 감지
        diff = cp.mean(cp.abs(curr_float - self.prev_mask)) / 255.0
        diff_val = float(diff)

        # 동적 alpha 결정
        if diff_val > self.diff_threshold:
            # 움직임 발생 → 즉각 반응 (잔상 제거)
            alpha = self.fast_alpha
        else:
            # 정지 상태 → 강한 스무딩 (떨림 억제)
            alpha = self.base_alpha

        # EMA 적용
        stabilized = alpha * curr_float + (1.0 - alpha) * self.prev_mask
        self.prev_mask = stabilized

        return stabilized.astype(cp.uint8)

    def reset(self):
        """상태 초기화"""
        self.prev_mask = None


"""
[V42 LEGACY] WarpGridStabilizer - Hysteresis + Alpha Smoothing
class WarpGridStabilizer:
    def __init__(self,
                 base_alpha=0.08,
                 snap_alpha=0.6,
                 low_threshold=2.0,
                 high_threshold=5.0,
                 alpha_smooth=0.3):
        self.base_alpha = base_alpha
        self.snap_alpha = snap_alpha
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.alpha_smooth = alpha_smooth
        self.prev_dx = None
        self.prev_dy = None
        self.prev_velocity = 0.0
        self.current_alpha = base_alpha
        self.is_fast_mode = False

    def update(self, dx_gpu, dy_gpu):
        if not HAS_CUDA:
            return dx_gpu, dy_gpu
        if self.prev_dx is None or self.prev_dx.shape != dx_gpu.shape:
            self.prev_dx = dx_gpu.copy()
            self.prev_dy = dy_gpu.copy()
            return dx_gpu, dy_gpu
        delta_dx = dx_gpu - self.prev_dx
        delta_dy = dy_gpu - self.prev_dy
        raw_velocity = float(cp.mean(cp.sqrt(delta_dx**2 + delta_dy**2)))
        self.prev_velocity = 0.7 * self.prev_velocity + 0.3 * raw_velocity
        velocity = self.prev_velocity
        if not self.is_fast_mode and velocity > self.high_threshold:
            self.is_fast_mode = True
        elif self.is_fast_mode and velocity < self.low_threshold:
            self.is_fast_mode = False
        if self.is_fast_mode:
            speed_factor = min((velocity - self.low_threshold) /
                              (self.high_threshold - self.low_threshold), 1.0)
            speed_factor = max(0.0, speed_factor)
            target_alpha = self.base_alpha + (self.snap_alpha - self.base_alpha) * speed_factor
        else:
            target_alpha = self.base_alpha
        self.current_alpha = (self.alpha_smooth * target_alpha +
                             (1.0 - self.alpha_smooth) * self.current_alpha)
        stabilized_dx = self.current_alpha * dx_gpu + (1.0 - self.current_alpha) * self.prev_dx
        stabilized_dy = self.current_alpha * dy_gpu + (1.0 - self.current_alpha) * self.prev_dy
        self.prev_dx = stabilized_dx.copy()
        self.prev_dy = stabilized_dy.copy()
        return stabilized_dx, stabilized_dy

    def reset(self):
        self.prev_dx = None
        self.prev_dy = None
        self.prev_velocity = 0.0
        self.current_alpha = self.base_alpha
        self.is_fast_mode = False
V42 LEGACY */
"""


class WarpGridStabilizer:
    """
    V43: Delta-Adaptive Warp Grid Stabilizer
    - 움직임 발생 즉시 alpha 상승 (지연 없음)
    - 정지 시 자연스럽게 안정화
    """

    def __init__(self, base_alpha=0.05, max_alpha=0.98, delta_scale=15.0, bypass=False):
        """
        Args:
            base_alpha: 정지 시 최소 alpha (강한 안정화)
            max_alpha: 움직임 시 최대 alpha (즉각 반응)
            delta_scale: delta → alpha 변환 스케일
            bypass: [V44] If True, skip stabilization and return input directly
        """
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        self.delta_scale = delta_scale
        self.prev_dx = None
        self.prev_dy = None
        self.bypass = bypass  # [V44] Frame-independent mode

    def update(self, dx_gpu, dy_gpu):
        """
        V43 Delta-Adaptive 업데이트
        """
        # [V44] bypass mode: return input directly (no stabilization)
        if self.bypass:
            return dx_gpu, dy_gpu

        if not HAS_CUDA:
            return dx_gpu, dy_gpu

        if self.prev_dx is None or self.prev_dx.shape != dx_gpu.shape:
            self.prev_dx = dx_gpu.copy()
            self.prev_dy = dy_gpu.copy()
            return dx_gpu, dy_gpu

        # 1. 프레임 간 변화량 계산
        delta_dx = dx_gpu - self.prev_dx
        delta_dy = dy_gpu - self.prev_dy
        delta_magnitude = float(cp.sqrt(cp.mean(delta_dx**2 + delta_dy**2)))

        # 2. delta → alpha 변환 (연속적)
        alpha = self.base_alpha + delta_magnitude * self.delta_scale
        alpha = min(self.max_alpha, max(self.base_alpha, alpha))

        # 3. EMA 적용
        smoothed_dx = alpha * dx_gpu + (1.0 - alpha) * self.prev_dx
        smoothed_dy = alpha * dy_gpu + (1.0 - alpha) * self.prev_dy

        # 4. 상태 업데이트
        self.prev_dx = smoothed_dx.copy()
        self.prev_dy = smoothed_dy.copy()

        return smoothed_dx, smoothed_dy

    def reset(self):
        self.prev_dx = None
        self.prev_dy = None
