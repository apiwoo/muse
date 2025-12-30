# Project MUSE - adaptive_bg.py
# V5.2: MODNet-Based Background Update with Temporal + Spatial Verification
# (C) 2025 MUSE Corp. All rights reserved.

import cupy as cp
import cupyx.scipy.ndimage
import cv2
import numpy as np
import os
import threading

class AdaptiveBackground:
    def __init__(self, width=1920, height=1080):
        self.w = width
        self.h = height
        self.bg_buffer = None  # Float32

        self.is_static_loaded = False

        # [V5.0] 파일 자동 저장 설정
        self.bg_file_path = None
        self.update_counter = 0
        self.file_save_interval = 600  # 약 20초마다 파일 저장 (30fps 기준)

        # [V5.2] 연속 프레임 검증 + 공간 마진
        self.temporal_counter = None  # (H, W) 각 픽셀의 연속 배경 프레임 수
        self.TEMPORAL_THRESHOLD = 15  # 15프레임 연속 배경이어야 업데이트 (0.5초)
        self.ALPHA_THRESHOLD = 0.005  # alpha < 0.005 = 확실한 배경
        self.SPATIAL_SIZE = 7         # 7x7 이웃 검사

    def load_static_background(self, bg_source):
        frame_bgr = None
        if isinstance(bg_source, str):
            if os.path.exists(bg_source):
                frame_bgr = cv2.imread(bg_source)
                self.bg_file_path = bg_source  # [V5.0] 파일 경로 저장
                print(f"[BG] Loaded static background from: {bg_source}")
            else:
                print(f"[BG] Background file not found: {bg_source}")
                return False
        elif isinstance(bg_source, np.ndarray):
            frame_bgr = bg_source

        if frame_bgr is not None:
            if frame_bgr.shape[1] != self.w or frame_bgr.shape[0] != self.h:
                frame_bgr = cv2.resize(frame_bgr, (self.w, self.h))
            self.bg_buffer = cp.asarray(frame_bgr).astype(cp.float32)
            self.is_static_loaded = True
            self.temporal_counter = None  # [V5.1] 리셋
            print("[BG] Static Background Loaded. Real-time update enabled for BG areas.")
            return True
        return False

    def set_file_path(self, path):
        """[V5.0] 배경 파일 경로 설정"""
        self.bg_file_path = path

    def reset(self, frame_gpu):
        if frame_gpu is None: return
        # 정적 배경이 로드되어 있다면 리셋하지 않음 (보호)
        if self.is_static_loaded: return
        
        self.bg_buffer = frame_gpu.astype(cp.float32)
        print("[BG] Background Buffer Reset (Initialized with Live Frame)")

    def update(self, frame_gpu, person_alpha):
        """
        [V5.2] MODNet 기반 배경 업데이트 (Temporal + Spatial 검증)

        원칙:
        1. 공간 검증: 7x7 이웃 중 최대 alpha가 임계값 미만이어야 함
        2. 시간 검증: N프레임 연속으로 배경이어야 업데이트
        3. 1프레임이라도 사람이면 카운터 리셋
        """
        if self.bg_buffer is None or frame_gpu is None:
            if frame_gpu is not None:
                self.reset(frame_gpu)
            return

        # Static 배경이 없으면 업데이트 안 함
        if not self.is_static_loaded:
            return

        h, w = frame_gpu.shape[:2]

        # 1. temporal_counter 초기화
        if self.temporal_counter is None or self.temporal_counter.shape != (h, w):
            self.temporal_counter = cp.zeros((h, w), dtype=cp.uint8)

        # 2. 알파 마스크 2D 변환
        alpha_2d = person_alpha
        if alpha_2d.ndim == 3:
            alpha_2d = alpha_2d.squeeze()

        # 3. 공간 검증: 7x7 이웃 중 최대 alpha 계산
        # 주변에 사람이 있으면 업데이트 안 함 (잔상 방지)
        neighbor_max_alpha = cupyx.scipy.ndimage.maximum_filter(
            alpha_2d, size=self.SPATIAL_SIZE
        )

        # 4. 배경 영역 판정 (공간 + alpha 임계값)
        # 본인 픽셀 AND 주변 7x7 모두 배경이어야 함
        is_bg_now = (neighbor_max_alpha < self.ALPHA_THRESHOLD)

        # 5. temporal_counter 업데이트
        # 배경인 픽셀: counter += 1 (최대 255)
        # 사람인 픽셀: counter = 0 (즉시 리셋)
        self.temporal_counter = cp.where(
            is_bg_now,
            cp.minimum(self.temporal_counter + 1, 255).astype(cp.uint8),
            cp.uint8(0)
        )

        # 6. N프레임 연속 배경인 픽셀만 업데이트
        is_safe_to_update = (self.temporal_counter >= self.TEMPORAL_THRESHOLD)
        update_mask = is_safe_to_update[..., None]  # (H, W, 1)

        # 7. 배경 업데이트 (안전한 영역만)
        current_float = frame_gpu.astype(cp.float32)
        self.bg_buffer = cp.where(update_mask, current_float, self.bg_buffer)

        # 8. 주기적 파일 저장
        if self.bg_file_path:
            self.update_counter += 1
            if self.update_counter >= self.file_save_interval:
                self.update_counter = 0
                self._save_background_async()

    def _save_background_async(self):
        """[V5.0] 배경 버퍼를 파일에 비동기 저장"""
        if self.bg_buffer is None or self.bg_file_path is None:
            return

        try:
            bg_copy = self.bg_buffer.get().astype(np.uint8)
        except Exception as e:
            print(f"[BG] Failed to copy buffer: {e}")
            return

        def save_worker(img, path):
            try:
                cv2.imwrite(path, img)
                print(f"[BG] Auto-saved background to: {path}")
            except Exception as e:
                print(f"[BG] Failed to save: {e}")

        t = threading.Thread(target=save_worker, args=(bg_copy, self.bg_file_path))
        t.daemon = True
        t.start()

    def is_background_stable(self):
        """
        [V6] 배경 안정성 확인
        정적 배경이 로드된 경우에만 True 반환.
        - True: 파일에서 로드되었거나 'B'키로 캡처됨 → 슬리밍 합성 가능
        - False: 적응형 배경 모드 → 슬리밍 합성 시 일렁임 발생 가능
        """
        return self.is_static_loaded

    def get_background(self):
        if self.bg_buffer is None:
            return cp.zeros((self.h, self.w, 3), dtype=cp.uint8)
        return self.bg_buffer.astype(cp.uint8)