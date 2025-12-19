# Project MUSE - buffers.py
# Frame Buffer Components for AI Latency Compensation
# Extracted from beauty_engine.py for modular architecture
# (C) 2025 MUSE Corp. All rights reserved.

"""
Buffer modules for frame synchronization and AI latency compensation.

This module contains:
- FrameSyncBuffer: AI mask latency compensation buffer
"""

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class FrameSyncBuffer:
    """
    [V33] Frame Synchronization Buffer for AI latency compensation.

    Problem: AI mask has 1-3 frame latency, causing "trailing ghost" artifacts.
    Solution: Buffer original frames and match them with AI results by timestamp.

    Key features:
    - Stores recent frames to match with delayed AI masks
    - Compensates for variable AI processing latency
    - Prevents ghosting artifacts in compositing
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
