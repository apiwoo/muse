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
    # [Core] 성형(Warping)을 위한 부위별 인덱스 정의 (Standard 106 Landmarks)
    # 이 상수를 BeautyEngine에서 import하여 사용합니다.
    FACE_INDICES = {
        "CONTOUR": list(range(0, 33)),          # 얼굴 윤곽 (턱 깎기용)
        "EYEBROW_L": list(range(33, 38)),       # 왼쪽 눈썹
        "EYEBROW_R": list(range(38, 43)),       # 오른쪽 눈썹
        "NOSE_BRIDGE": list(range(52, 57)),     # 콧대 (코 높이기)
        "NOSE_BASE": list(range(57, 66)),       # 코 볼/끝 (코 축소)
        "EYE_L": list(range(66, 75)),           # 왼쪽 눈 (눈 키우기)
        "EYE_R": list(range(75, 84)),           # 오른쪽 눈 (눈 키우기)
        "MOUTH_OUTER": list(range(84, 96)),     # 입술 외곽
        "MOUTH_INNER": list(range(96, 104)),    # 입술 안쪽
        "PUPIL_L": [104],                       # 왼쪽 눈동자
        "PUPIL_R": [105]                        # 오른쪽 눈동자
    }

    def __init__(self, root_dir="assets/models"):
        """
        [Mode A] High-Poly Face Tracking Engine
        - 모델: InsightFace 'buffalo_l' (Detection + 106 Landmark)
        - 역할: 얼굴 좌표 추출 및 분석
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
            # InsightFace 추론
            faces = self.app.get(frame_bgr)
            return faces
        except Exception as e:
            return []

    def draw_debug(self, frame, faces):
        """
        [Simple Debug] 점만 찍어서 트래킹 여부 확인
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
                "CONTOUR": (255, 200, 200),     # 살구색
                "EYEBROW_L": (200, 255, 200),   # 연두색
                "EYEBROW_R": (200, 255, 200),
                "EYE_L": (0, 255, 0),           # 초록색
                "EYE_R": (0, 255, 0),
                "NOSE_BRIDGE": (200, 200, 255), # 연하늘
                "NOSE_BASE": (255, 255, 0),     # 노란색
                "MOUTH_OUTER": (0, 0, 255),     # 빨간색
                "MOUTH_INNER": (100, 100, 255)  # 진한 빨강
            }

            # 정의된 그룹에 따라 점 찍기
            for group_name, indices in self.FACE_INDICES.items():
                color = colors.get(group_name, (255, 255, 255))
                for idx in indices:
                    if idx < len(lm):
                        cv2.circle(frame, tuple(lm[idx]), 2, color, -1)

            # 눈동자 강조
            if len(lm) > 105:
                cv2.circle(frame, tuple(lm[104]), 3, (0, 255, 255), -1) # 노란색 눈동자

        return frame

    # export_debug_log 제거됨 (불필요)