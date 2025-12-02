# Project MUSE - body_tracker.py
# Target: Multi-Model Student Loader
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import time
import math
import os
import glob

# [Change] Smart Import
try:
    from ai.distillation.student.inference_trt import StudentInferenceTRT
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ai.distillation.student.inference import StudentInference

# OneEuro Filter (Jitter Control)
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
        if t_e <= 0.0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class BodyTracker:
    def __init__(self, profiles=[]):
        """
        [BodyTracker V3.0] Multi-Model Support
        - profiles: ['front', 'top', 'default'...]
        - 모든 프로파일에 대한 엔진을 미리 로드합니다.
        """
        self.models = {} # {'front': model_obj, ...}
        self.active_profile = None
        self.active_model = None
        
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.model_dir = os.path.join(self.root_dir, "assets", "models", "personal")
        
        print(f"🧠 [BodyTracker] 멀티 모델 로딩 시작 (대상: {profiles})")
        
        # 1. 기본 모델 검색 (student_model.engine -> fallback)
        default_engine = os.path.join(self.model_dir, "student_model.engine")
        default_model = self._load_model("default", default_engine)
        if default_model:
            self.models['default'] = default_model
            self.active_profile = 'default'
            self.active_model = default_model

        # 2. 프로파일별 모델 로드
        for p_name in profiles:
            engine_path = os.path.join(self.model_dir, f"student_{p_name}.engine")
            # 만약 전용 모델이 없으면 -> default 모델을 공유해서 씀 (메모리 절약)
            if not os.path.exists(engine_path):
                if 'default' in self.models:
                    self.models[p_name] = self.models['default']
                    print(f"   ⚠️ [{p_name}] 전용 모델 없음 -> Default 모델 공유")
                continue
                
            model = self._load_model(p_name, engine_path)
            if model:
                self.models[p_name] = model

        # 필터
        self.filter = OneEuroFilter(time.time(), min_cutoff=0.5, beta=0.2, d_cutoff=1.0)
        self.latest_mask = None
        self.last_log_time = time.time()

    def _load_model(self, name, path):
        if not TRT_AVAILABLE:
            print("   ❌ TensorRT 모듈 없음. PyTorch 모드로 대체합니다.")
            return StudentInference() # PyTorch (Single Model only for now)

        try:
            print(f"   Load '{name}' <- {os.path.basename(path)} ...", end=" ")
            model = StudentInferenceTRT(engine_path=path)
            if model.is_ready:
                print("✅ Success")
                return model
            else:
                print("❌ Failed (Not Ready)")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def set_profile(self, profile_name):
        """활성 모델 변경 (Instant Switch)"""
        if profile_name in self.models:
            if self.active_profile != profile_name:
                self.active_profile = profile_name
                self.active_model = self.models[profile_name]
                print(f"🧠 [BodyTracker] Model Switched to: {profile_name}")
                # 필터 초기화 (위치가 확 바뀌므로 튀는거 방지)
                self.filter = OneEuroFilter(time.time(), min_cutoff=0.5, beta=0.2, d_cutoff=1.0)
            return True
        else:
            print(f"⚠️ [BodyTracker] Profile '{profile_name}' model not found.")
            return False

    def process(self, frame_bgr):
        if self.active_model is None or frame_bgr is None:
            return None
        
        # 추론
        mask, raw_keypoints = self.active_model.infer(frame_bgr)
        
        if raw_keypoints is None: return None
            
        self.latest_mask = mask

        # 필터링
        curr_time = time.time()
        coords = raw_keypoints[:, :2]
        confs = raw_keypoints[:, 2:3]
        smoothed_coords = self.filter(curr_time, coords)
        smoothed_keypoints = np.hstack([smoothed_coords, confs])

        # 로그
        if curr_time - self.last_log_time > 5.0: # 5초마다
            print(f"   [Tracker] Active: {self.active_profile}")
            self.last_log_time = curr_time
        
        return smoothed_keypoints

    def get_mask(self):
        return self.latest_mask

    def draw_debug(self, frame, keypoints):
        """
        [Visual Check] 뼈대 그리기 (기존 로직 유지 + 복원)
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