# Project MUSE - body_tracker.py
# Target: Mode A (Personalized Student Model Integrated)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import time
import math
import os

# [Change] Smart Import: Try TensorRT first, then PyTorch
# 'src.' prefix removed for compatibility when running from src/main.py
try:
    from ai.distillation.student.inference_trt import StudentInferenceTRT
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ai.distillation.student.inference import StudentInference

# ==============================================================================
# [Core Algorithm] OneEuro Filter Implementation
# 지진(Jitter) 현상을 잡기 위한 적응형 필터입니다.
# ==============================================================================
class OneEuroFilter:
    def __init__(self, t0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        
        # Avoid division by zero
        if t_e <= 0.0: return self.x_prev

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class BodyTracker:
    def __init__(self):
        """
        [BodyTracker V2.1]
        - Engine: Personalized Student Model (Lightweight)
        - Priority: TensorRT (.engine) > PyTorch (.pth)
        - Jitter Control: OneEuro Filter
        """
        self.model = None
        self.engine_ready = False
        self.mode = "None"
        
        # 1. Try TensorRT First
        if TRT_AVAILABLE:
            print("💪 [BodyTracker] TensorRT 엔진(High-Perf) 로드 시도...")
            try:
                trt_model = StudentInferenceTRT()
                if trt_model.is_ready:
                    self.model = trt_model
                    self.engine_ready = True
                    self.mode = "TensorRT"
                    print("   ✅ TensorRT 가속 활성화됨.")
            except Exception as e:
                print(f"   ⚠️ TensorRT 로드 실패: {e}")

        # 2. Fallback to PyTorch
        if not self.engine_ready:
            print("💪 [BodyTracker] PyTorch 엔진(Fallback) 로드 시도...")
            try:
                self.model = StudentInference()
                if self.model.is_ready:
                    self.engine_ready = True
                    self.mode = "PyTorch"
                    print("   ✅ PyTorch 엔진 활성화됨 (최적화를 위해 TensorRT 변환 권장).")
            except Exception as e:
                print(f"❌ [BodyTracker] 모든 모델 로드 실패: {e}")
                self.engine_ready = False
            
        self.last_log_time = time.time()
        self.latest_mask = None # 마스크 저장용
        
        # [Jitter Control]
        self.filter = OneEuroFilter(time.time(), min_cutoff=0.5, beta=0.2, d_cutoff=1.0)

    def process(self, frame_bgr):
        """
        :return: keypoints numpy array (17, 3) -> [x, y, conf]
        """
        if not self.engine_ready or frame_bgr is None:
            return None
        
        # 1. Student 추론 실행 (Mask + Pose)
        mask, raw_keypoints = self.model.infer(frame_bgr)
        
        if raw_keypoints is None:
            return None
            
        # [Important] 마스크 저장 (나중에 렌더러가 가져갈 수 있게)
        self.latest_mask = mask

        # 2. [Core] OneEuro Filter 적용 (Pose Jitter 제거)
        curr_time = time.time()
        coords = raw_keypoints[:, :2] # (17, 2)
        confs = raw_keypoints[:, 2:3] # (17, 1)
        
        # 필터링 수행
        smoothed_coords = self.filter(curr_time, coords)
        
        # 다시 합치기
        smoothed_keypoints = np.hstack([smoothed_coords, confs])

        # [Log] 2초에 한 번씩만 상태 출력
        if curr_time - self.last_log_time > 2.0:
            max_conf = np.max(confs)
            print(f"🔍 [BodyTracker] Tracking ({self.mode}): Conf={max_conf:.2f}")
            self.last_log_time = curr_time
        
        return smoothed_keypoints

    def get_mask(self):
        """최신 프레임의 배경 제거 마스크 반환"""
        return self.latest_mask

    def draw_debug(self, frame, keypoints):
        """
        [Visual Check] 뼈대 그리기 (기존 로직 유지)
        """
        if keypoints is None:
            return frame

        CONF_THRESH = 0.4

        # 1. 점 찍기
        for i in range(17):
            x, y, conf = keypoints[i]
            h, w = frame.shape[:2]
            if x < 0 or x >= w or y < 0 or y >= h: continue

            if conf > CONF_THRESH:
                color = (255, 100, 0) if i % 2 == 1 else (0, 100, 255)
                if i <= 4: color = (0, 255, 255) # Face
                radius = 4 if i <= 4 else 6
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                cv2.circle(frame, (int(x), int(y)), radius+1, (255, 255, 255), 1)

        # 2. 선 연결
        skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),      # Arms
            (11, 13), (13, 15), (12, 14), (14, 16), # Legs
            (5, 6), (11, 12), (5, 11), (6, 12),     # Torso
            (0, 1), (0, 2), (1, 3), (2, 4)        # Face
        ]

        for p1, p2 in skeleton:
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            if c1 > CONF_THRESH and c2 > CONF_THRESH:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        return frame