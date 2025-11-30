# Project MUSE - facemesh.py
# Created for Mode A (Visual Supremacy)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import os

# GPU 가속 라이브러리 (선택적)
try:
    import cupy as cp
except ImportError:
    cp = None

# InsightFace (고정밀 얼굴 분석)
from insightface.app import FaceAnalysis

class FaceMesh:
    # [Core] 성형(Warping)을 위한 부위별 인덱스 정의 (Custom 106 Model Layout)
    # 분석된 랜드마크 구조에 맞춰 인덱스를 재정의했습니다.
    FACE_INDICES = {
        # [얼굴 윤곽] 0: 턱 중앙, 1: 왼쪽 관자놀이, 17: 오른쪽 관자놀이
        # 좌측 라인: 1 -> 9~16(외곽) -> 2~8(턱선) -> 0
        # 우측 라인: 17 -> 25~32(외곽) -> 18~24(턱선) -> 0
        "JAW_L": [1] + list(range(9, 17)) + list(range(2, 9)),  # 왼쪽 얼굴 라인
        "JAW_R": [17] + list(range(25, 33)) + list(range(18, 25)), # 오른쪽 얼굴 라인
        "CHIN_CENTER": [0],

        # [눈썹]
        "EYEBROW_L": list(range(43, 52)),       # 왼쪽 눈썹 (43~51)
        "EYEBROW_R": list(range(97, 106)),      # 오른쪽 눈썹 (97~105)

        # [눈]
        "EYE_L": list(range(33, 43)),           # 왼쪽 눈 (33~42)
        "EYE_R": list(range(87, 97)),           # 오른쪽 눈 (87~96)

        # [코]
        "NOSE_ROOT": [72],                      # 미간 (콧대 시작)
        "NOSE_BRIDGE": [73, 74, 86],            # 콧대 ~ 코끝(86)
        "NOSE_TIP": [86],                       # 코에서 가장 높은 점
        "NOSE_BASE": [80],                      # 코 밑 중앙
        "NOSE_BODY": list(range(75, 87)),       # 코 전체 영역

        # [입]
        "MOUTH_ALL": list(range(52, 72)),       # 입 전체
        "MOUTH_CORNERS": [52, 61]               # 입꼬리 (좌:52, 우:61)
    }

    def __init__(self, root_dir="assets/models"):
        """
        [Mode A] High-Poly Face Tracking Engine
        - 모델: InsightFace 'buffalo_l' (Detection + 106 Landmark)
        - 역할: 얼굴 좌표 추출 및 분석 (성형용 데이터 제공)
        """
        print("🧠 [FaceMesh] AI 엔진 로딩 중... (InsightFace)")
        
        # 모델 경로: assets/models/insightface
        # [중요] landmark_2d_106 모델을 명시적으로 지정하여 106개 포인트를 강제함
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=root_dir, 
            allowed_modules=['detection', 'landmark_2d_106', 'genderage'], 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 엔진 준비
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ [FaceMesh] 엔진 장전 완료 (CUDA Accelerated)")
        except Exception as e:
            print(f"⚠️ [FaceMesh] 엔진 로딩 실패: {e}")
            self.app = None

    def process(self, frame_bgr):
        """
        프레임에서 얼굴 랜드마크를 추출합니다.
        """
        if self.app is None or frame_bgr is None:
            return []

        try:
            # InsightFace 추론 (좌표 추출)
            faces = self.app.get(frame_bgr)
            return faces
        except Exception as e:
            return []

    def draw_debug(self, frame, faces):
        """
        [Simple Debug] 점만 찍어서 트래킹 여부만 가볍게 확인
        """
        if not faces:
            return frame

        for face in faces:
            lm = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(int)
            elif face.kps is not None:
                lm = face.kps.astype(int)

            if lm is not None:
                for p in lm:
                    cv2.circle(frame, tuple(p), 2, (0, 255, 255), -1)
        return frame

    def draw_mesh_debug(self, frame, faces):
        """
        [Visual Check] 정의된 부위별로 색상을 다르게 표시 (연결선 X, 그룹 확인용)
        이 함수는 우리가 정의한 'FACE_INDICES'가 실제 얼굴 부위와 매칭되는지 색깔로 검증할 때 씁니다.
        """
        if not faces:
            return frame

        for face in faces:
            lm = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(int)
            elif face.kps is not None:
                lm = face.kps.astype(int)
            
            if lm is None or len(lm) != 106:
                continue

            # 그룹별 색상 지정 (BGR)
            colors = {
                "JAW_L": (255, 200, 200),     # 살구색 (왼쪽 턱)
                "JAW_R": (255, 200, 200),     # 살구색 (오른쪽 턱)
                "CHIN_CENTER": (255, 100, 100), # 진한 살구색 (턱 끝)
                "EYEBROW_L": (200, 255, 200),   # 연두색
                "EYEBROW_R": (200, 255, 200),
                "EYE_L": (0, 255, 0),           # 초록색 (눈)
                "EYE_R": (0, 255, 0),
                "NOSE_BRIDGE": (200, 200, 255), # 연하늘
                "NOSE_BODY": (255, 255, 0),     # 노란색 (코)
                "MOUTH_ALL": (0, 0, 255),       # 빨간색 (입술)
            }

            # 정의된 그룹에 따라 점 찍기 (선 연결 X)
            for group_name, indices in self.FACE_INDICES.items():
                color = colors.get(group_name, (255, 255, 255))
                for idx in indices:
                    if idx < len(lm):
                        cv2.circle(frame, tuple(lm[idx]), 2, color, -1)

            # 코 끝 강조 (86번)
            cv2.circle(frame, tuple(lm[86]), 3, (0, 255, 255), -1)

        return frame

    def draw_indices_debug(self, frame, faces):
        """
        [Compatibility] main.py와의 호환성을 위해 유지.
        """
        return self.draw_mesh_debug(frame, faces)