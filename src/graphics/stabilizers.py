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

    def __init__(self, min_cutoff=0.01, base_beta=5.0, high_speed_beta=50.0):
        """
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            base_beta: Base beta for slow movements
            high_speed_beta: Maximum beta for fast movements
        """
        self.min_cutoff = min_cutoff
        self.base_beta = base_beta
        self.high_speed_beta = high_speed_beta
        self.prev_val = None
        self.prev_trend = None
        self.last_time = None

    def update(self, val_array):
        """
        Update stabilizer with new landmark values.

        Args:
            val_array: New landmark coordinates (numpy array)
        Returns:
            Stabilized landmark coordinates
        """
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


class MaskStabilizer:
    """
    [V37] Mask Temporal Stabilizer with Multi-Frame Median Consensus.

    Key features:
    - Multi-Frame Consensus: Uses median of recent N frames
    - Reusable buffer: Minimizes memory allocation overhead
    - EMA smoothing for final output
    """

    def __init__(self, alpha=0.10, consensus_frames=5):
        """
        Args:
            alpha: EMA smoothing factor (lower = more smoothing)
            consensus_frames: Number of frames for median consensus
        """
        self.alpha = alpha
        self.consensus_frames = consensus_frames
        self.prev_mask = None
        self.mask_history = []
        self.stacked_buffer = None

    def update(self, mask_gpu):
        """
        Update stabilizer with new mask.

        Args:
            mask_gpu: New mask (CuPy array, uint8)
        Returns:
            Stabilized mask (CuPy array, uint8)
        """
        if not HAS_CUDA:
            return mask_gpu

        if self.prev_mask is None or self.prev_mask.shape != mask_gpu.shape:
            self.prev_mask = mask_gpu.astype(cp.float32)
            self.mask_history = [mask_gpu.copy() for _ in range(self.consensus_frames)]
            # Initialize stack buffer
            h, w = mask_gpu.shape
            self.stacked_buffer = cp.zeros((self.consensus_frames, h, w), dtype=cp.uint8)
            return mask_gpu

        # Update history (FIFO)
        self.mask_history.pop(0)
        self.mask_history.append(mask_gpu.copy())

        # Stack into reusable buffer (minimize memory allocation)
        for i, m in enumerate(self.mask_history):
            self.stacked_buffer[i] = m

        # Multi-Frame Median Consensus
        consensus_mask = cp.median(self.stacked_buffer, axis=0)

        # EMA smoothing
        curr_float = consensus_mask.astype(cp.float32)
        stabilized = self.alpha * curr_float + (1.0 - self.alpha) * self.prev_mask
        self.prev_mask = stabilized

        return stabilized.astype(cp.uint8)

    def reset(self):
        """Reset stabilizer state."""
        self.prev_mask = None
        self.mask_history = []
        self.stacked_buffer = None


class WarpGridStabilizer:
    """
    [V37] Warp Grid Temporal Stabilizer with Hysteresis + Alpha Smoothing.

    Key features:
    - Hysteresis dual-threshold: Prevents jitter during state transitions
    - Alpha Smoothing: Smooths the alpha value itself via EMA
    - Velocity EMA: Less sensitive to instantaneous speed changes
    """

    def __init__(self,
                 base_alpha=0.08,
                 snap_alpha=0.6,
                 low_threshold=2.0,
                 high_threshold=5.0,
                 alpha_smooth=0.3):
        """
        Args:
            base_alpha: Alpha for stationary state (strong smoothing)
            snap_alpha: Alpha for fast movement (responsive)
            low_threshold: Lower threshold for hysteresis
            high_threshold: Upper threshold for hysteresis
            alpha_smooth: EMA factor for alpha smoothing
        """
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
        """
        Update stabilizer with new warp grids.

        Args:
            dx_gpu: X displacement grid (CuPy array)
            dy_gpu: Y displacement grid (CuPy array)
        Returns:
            Tuple of stabilized (dx_gpu, dy_gpu)
        """
        if not HAS_CUDA:
            return dx_gpu, dy_gpu

        if self.prev_dx is None or self.prev_dx.shape != dx_gpu.shape:
            self.prev_dx = dx_gpu.copy()
            self.prev_dy = dy_gpu.copy()
            return dx_gpu, dy_gpu

        # Calculate grid change
        delta_dx = dx_gpu - self.prev_dx
        delta_dy = dy_gpu - self.prev_dy
        raw_velocity = float(cp.mean(cp.sqrt(delta_dx**2 + delta_dy**2)))

        # Velocity EMA (smooth instantaneous changes)
        self.prev_velocity = 0.7 * self.prev_velocity + 0.3 * raw_velocity
        velocity = self.prev_velocity

        # Hysteresis state machine
        if not self.is_fast_mode and velocity > self.high_threshold:
            self.is_fast_mode = True
        elif self.is_fast_mode and velocity < self.low_threshold:
            self.is_fast_mode = False

        # Determine target alpha
        if self.is_fast_mode:
            speed_factor = min((velocity - self.low_threshold) /
                              (self.high_threshold - self.low_threshold), 1.0)
            speed_factor = max(0.0, speed_factor)
            target_alpha = self.base_alpha + (self.snap_alpha - self.base_alpha) * speed_factor
        else:
            target_alpha = self.base_alpha

        # Alpha smoothing
        self.current_alpha = (self.alpha_smooth * target_alpha +
                             (1.0 - self.alpha_smooth) * self.current_alpha)

        # Apply EMA
        stabilized_dx = self.current_alpha * dx_gpu + (1.0 - self.current_alpha) * self.prev_dx
        stabilized_dy = self.current_alpha * dy_gpu + (1.0 - self.current_alpha) * self.prev_dy

        self.prev_dx = stabilized_dx.copy()
        self.prev_dy = stabilized_dy.copy()

        return stabilized_dx, stabilized_dy

    def reset(self):
        """Reset stabilizer state."""
        self.prev_dx = None
        self.prev_dy = None
        self.prev_velocity = 0.0
        self.current_alpha = self.base_alpha
        self.is_fast_mode = False
