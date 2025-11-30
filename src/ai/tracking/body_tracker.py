# Project MUSE - body_tracker.py
# Target: RTX 3060/4090 Mode A (High Performance)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import time
import math
from ai.tracking.vitpose_trt import VitPoseTrt

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
        [Body Tracking Engine]
        - Engine: ViTPose-Huge (TensorRT Accelerated)
        - Keypoints: COCO 17 Format
        - Feature: OneEuro Filter for Jitter Reduction
        """
        print("💪 [BodyTracker] ViTPose-Huge 엔진(TensorRT) 로드 중...")
        try:
            self.model = VitPoseTrt()
            self.engine_ready = True
        except Exception as e:
            print(f"❌ [BodyTracker] 엔진 로드 실패: {e}")
            self.engine_ready = False
            
        self.last_log_time = time.time()
        
        # [Jitter Control] 필터 초기화
        # min_cutoff: 기본 떨림 억제 강도 (클수록 덜 떨림, 0.1~1.0 추천)
        # beta: 빠른 움직임 반응 속도 (클수록 반응 빠름, 0.001~0.1 추천)
        # 현재 설정: 정지 상태에서는 강력하게 잡고(0.5), 움직일 땐 적당히 따라감(0.005)
        self.filter = OneEuroFilter(time.time(), min_cutoff=0.5, beta=0.005, d_cutoff=1.0)

    def process(self, frame_bgr):
        """
        :return: keypoints numpy array (17, 3) -> [x, y, conf]
        """
        if not self.engine_ready or frame_bgr is None:
            return None
        
        # 1. ViTPose 추론 실행
        raw_keypoints = self.model.inference(frame_bgr)
        
        if raw_keypoints is None:
            return None

        # 2. [Core] OneEuro Filter 적용
        # (x, y) 좌표만 필터링하고 신뢰도(conf)는 그대로 둡니다.
        # raw_keypoints shape: (17, 3) -> x, y, conf
        
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
            nose_x, nose_y = smoothed_keypoints[0, :2]
            print(f"🔍 [BodyTracker] Tracking: MaxConf={max_conf:.2f}, Nose=({int(nose_x)}, {int(nose_y)})")
            self.last_log_time = curr_time
        
        return smoothed_keypoints

    def draw_debug(self, frame, keypoints):
        """
        [Visual Check] COCO 포맷(17 Keypoints) 뼈대 그리기
        """
        if keypoints is None:
            return frame

        # [Prod] 신뢰도 임계값
        CONF_THRESH = 0.4

        # 1. 점 찍기
        for i in range(17):
            x, y, conf = keypoints[i]
            
            # 좌표가 화면 밖이면 스킵
            h, w = frame.shape[:2]
            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            if conf > CONF_THRESH:
                # 관절마다 색깔 다르게 (좌:파랑, 우:빨강)
                color = (255, 100, 0) if i % 2 == 1 else (0, 100, 255)
                
                # 얼굴 부위(0~4: 코,눈,귀)는 노란색 계열로 강조
                if i <= 4: 
                    color = (0, 255, 255) # Yellow
                    radius = 4
                else:
                    radius = 6 # 몸통은 좀 더 크게
                
                cv2.circle(frame, (int(x), int(y)), radius, color, -1)
                
                # [Visual] 테두리 추가 (가시성 확보)
                cv2.circle(frame, (int(x), int(y)), radius+1, (255, 255, 255), 1)

        # 2. 선 연결 (Skeleton)
        skeleton = [
            # 팔
            (5, 7), (7, 9),       # 왼팔
            (6, 8), (8, 10),      # 오른팔
            # 다리
            (11, 13), (13, 15),   # 왼다리
            (12, 14), (14, 16),   # 오른다리
            # 몸통
            (5, 6),               # 어깨선
            (11, 12),             # 골반선
            (5, 11), (6, 12),     # 옆구리
            # 얼굴 (갈매기 모양)
            (0, 1), (0, 2),       # 코-눈
            (1, 3), (2, 4)        # 눈-귀
        ]

        for p1, p2 in skeleton:
            x1, y1, c1 = keypoints[p1]
            x2, y2, c2 = keypoints[p2]
            
            if c1 > CONF_THRESH and c2 > CONF_THRESH:
                # 얼굴 연결선은 얇게, 몸통은 굵게
                thickness = 2
                color = (0, 255, 0) # Green
                
                if p1 <= 4 and p2 <= 4:
                    thickness = 1
                    color = (100, 255, 100) # Light Green

                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        return frame