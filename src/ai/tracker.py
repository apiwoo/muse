# Project MUSE - src/ai/tracker.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from src.utils.logger import get_logger

# =========================================================
# [1] OneEuroFilter 구현 클래스 (떨림 보정용 알고리즘)
# 논문: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
# =========================================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: 최소 차단 주파수 (낮을수록 떨림이 줄어들지만 딜레이 발생)
        beta: 속도 계수 (높을수록 빠른 움직임에 민감하게 반응하여 딜레이 감소)
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        
        # 시간 간격이 너무 작으면 계산 건너뜀 (안정성 확보)
        if t_e <= 0.0: 
            return self.x_prev

        # 1. 변화율(속도) 추정 (Jitter vs Movement 구분)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 2. 속도에 따른 차단 주파수 조절
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. 최종 값 필터링
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 상태 업데이트
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

# =========================================================
# [2] FaceTracker 클래스 (AI 엔진 + 필터링 + 예외처리)
# =========================================================
class FaceTracker:
    def __init__(self):
        self.logger = get_logger("AI_Tracker")
        
        # 1. MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True, # Iris(눈동자) 포함
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 2. 필터 시스템 초기화
        self.filters = []     # 478개 점 각각에 대한 필터 객체 리스트
        self.is_initialized = False
        
        # 필터 파라미터 튜닝 (High Responsiveness Setting)
        # 지연(Lag)을 없애기 위해 min_cutoff와 beta 값을 대폭 상향 조정했습니다.
        self.cfg_min_cutoff = 1.0
        self.cfg_beta = 10.0
        
        # 3. 이상치 방어용 상태 변수
        self.prev_landmarks = None  # 직전 프레임의 '정상' 랜드마크 저장
        self.loss_count = 0         # 추적 실패 연속 카운트 (초기화 방지용)
        self.LOSS_THRESHOLD = 30    # 30프레임(약 1초) 이상 놓쳐야 초기화

        self.logger.info("🤖 AI Face Tracker v2.2 (Stability Improved) 초기화 완료")

    def _init_filters(self, t, landmarks):
        """최초 감지 시 필터들을 초기화합니다."""
        self.filters = []
        for lm in landmarks:
            # X, Y, Z 각각에 대해 필터 생성
            f_x = OneEuroFilter(t, lm.x, min_cutoff=self.cfg_min_cutoff, beta=self.cfg_beta)
            f_y = OneEuroFilter(t, lm.y, min_cutoff=self.cfg_min_cutoff, beta=self.cfg_beta)
            f_z = OneEuroFilter(t, lm.z, min_cutoff=self.cfg_min_cutoff, beta=self.cfg_beta)
            self.filters.append((f_x, f_y, f_z))
        self.is_initialized = True
        self.logger.info("✨ 필터 시스템 가동 시작 (478 points)")

    def process(self, frame):
        """
        Input: BGR 이미지
        Output: 안정화된 results 객체 (또는 None)
        """
        if frame is None:
            return None

        current_time = time.time()
        
        # 1. 이미지 전처리
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # 2. AI 추론
        raw_results = self.face_mesh.process(frame_rgb)

        # 3. 결과 후처리 (필터링 및 이상치 제거)
        if raw_results and raw_results.multi_face_landmarks:
            raw_landmarks = raw_results.multi_face_landmarks[0] # 첫 번째 얼굴만
            
            # (A) [가림 방어] 물리적으로 불가능한 형태인지 검사
            if self._is_anomaly(raw_landmarks):
                if self.prev_landmarks:
                    # MediaPipe 결과 구조체에 이전 좌표를 덮어씌워 반환
                    raw_results.multi_face_landmarks[0] = self.prev_landmarks
                    return raw_results
                else:
                    return None 
            
            # (B) [떨림 방지] OneEuroFilter 적용
            if not self.is_initialized:
                self._init_filters(current_time, raw_landmarks.landmark)
                self.prev_landmarks = raw_landmarks # 초기값 저장
                return raw_results
            
            # 필터링
            stabilized_landmarks = raw_landmarks 
            
            for i, lm in enumerate(raw_landmarks.landmark):
                f_x, f_y, f_z = self.filters[i]
                
                new_x = f_x.filter(current_time, lm.x)
                new_y = f_y.filter(current_time, lm.y)
                new_z = f_z.filter(current_time, lm.z)
                
                stabilized_landmarks.landmark[i].x = new_x
                stabilized_landmarks.landmark[i].y = new_y
                stabilized_landmarks.landmark[i].z = new_z

            # 정상적으로 처리된 결과를 저장
            self.prev_landmarks = stabilized_landmarks
            self.loss_count = 0 # 추적 성공 시 카운트 리셋
            
            return raw_results

        else:
            # 얼굴을 놓쳤을 때
            self.loss_count += 1
            
            # [FIX] 즉시 초기화하지 않고, 일정 시간(THRESHOLD) 경과 후에만 초기화
            if self.loss_count > self.LOSS_THRESHOLD:
                if self.is_initialized:
                    self.logger.warning("⚠️ 얼굴 추적 중단됨 (Reset Filters)")
                self.is_initialized = False
                self.prev_landmarks = None
            
            # 짧은 순간 놓친 건 무시 (None 반환 -> 렌더러가 원본 프레임 보여줌)
            return None

    def _is_anomaly(self, landmarks):
        """
        얼굴 랜드마크가 비정상적인지(손 가림, 튀는 값) 검사합니다.
        """
        lms = landmarks.landmark
        
        # [검사 1] 입이 비정상적으로 벌어졌는가? (Face Mesh Index: 13=윗입술, 14=아랫입술)
        mouth_open_dist = abs(lms[13].y - lms[14].y)
        
        if mouth_open_dist > 0.15: 
            return True
            
        return False

    def draw_debug(self, frame, results):
        pass # 렌더러가 있으므로 사용 안 함