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
    [V41] Frame Synchronization Buffer for AI latency compensation.
    (Updated from V33 for Time-Locked Sync)

    Problem: AI mask has 1-3 frame latency, causing "trailing ghost" artifacts.
    Solution: Buffer original frames and match them with AI results by timestamp.

    Key features:
    - Stores recent frames to match with delayed AI masks
    - Compensates for variable AI processing latency
    - Prevents ghosting artifacts in compositing
    - [V41] get_exact_frame(): ID 기반 정확한 프레임 회수 지원
    - [V41] push(): 랜드마크 동시 저장 지원 (Optional)
    """

    def __init__(self, max_size=3):
        """
        Args:
            max_size: Maximum number of frames to buffer (default: 3)
                      Higher = more latency tolerance, more VRAM usage (~15MB/frame)
        """
        self.max_size = max_size
        self.buffer = []  # [V41] List of (frame_id, frame_gpu, landmarks) tuples
        self.frame_counter = 0

    def push(self, frame_gpu, landmarks=None):
        """
        [V41 강화] Add a new frame to the buffer with optional landmarks.

        Args:
            frame_gpu: Current frame (CuPy array)
            landmarks: [V41] Optional landmarks for this frame (동기화용)
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

        # [V41] 랜드마크도 함께 저장 (있으면)
        self.buffer.append((frame_id, frame_copy, landmarks))

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
            # [V41] 튜플 인덱스 1 = frame_gpu
            return self.buffer[0][1]

        # [V41] 튜플 인덱스 1 = frame_gpu
        return self.buffer[target_idx][1]

    def get_exact_frame(self, frame_id):
        """
        [V41] 특정 ID의 프레임을 정확히 회수.

        AI 추론 결과와 함께 넘어온 frame_id로 해당 프레임을 정확히 찾아 반환.
        Time-Lock 동기화의 핵심.

        Args:
            frame_id: push() 시 반환받은 프레임 ID

        Returns:
            해당 프레임 또는 None (버퍼에서 이미 제거된 경우)
        """
        for stored_id, stored_frame, _ in self.buffer:
            if stored_id == frame_id:
                return stored_frame

        # ID가 없으면 (이미 제거됨) 가장 오래된 프레임 반환
        # 또는 None 반환하여 호출부에서 처리
        return self.buffer[0][1] if len(self.buffer) > 0 else None

    def get_exact_frame_with_landmarks(self, frame_id):
        """
        [V41] 특정 ID의 프레임과 랜드마크를 함께 회수.

        Args:
            frame_id: push() 시 반환받은 프레임 ID

        Returns:
            (frame_gpu, landmarks) 튜플 또는 (None, None)
        """
        for stored_id, stored_frame, stored_landmarks in self.buffer:
            if stored_id == frame_id:
                return stored_frame, stored_landmarks

        # ID가 없으면 가장 오래된 프레임 반환
        if len(self.buffer) > 0:
            return self.buffer[0][1], self.buffer[0][2]
        return None, None

    def get_latest(self):
        """Get the most recent frame in buffer."""
        if len(self.buffer) == 0:
            return None
        # [V41] 튜플 인덱스 1 = frame_gpu
        return self.buffer[-1][1]

    def get_latest_with_id(self):
        """
        [V41] Get the most recent frame with its ID.

        Returns:
            (frame_id, frame_gpu) tuple or (None, None) if buffer is empty
        """
        if len(self.buffer) == 0:
            return None, None
        return self.buffer[-1][0], self.buffer[-1][1]

    def reset(self):
        """Clear all buffered frames."""
        self.buffer.clear()
        self.frame_counter = 0

    def __len__(self):
        return len(self.buffer)
