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
        [Debug] 2D 랜드마크 시각화 (기본)
        """
        if not faces:
            return frame

        for face in faces:
            if hasattr(face, 'bbox'):
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

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
        [New] 3D Mesh 느낌으로 랜드마크를 연결하여 시각화 (InsightFace 106 Standard)
        """
        if not faces:
            return frame

        # InsightFace 106 랜드마크 연결 정의
        connections = [
            # 1. 얼굴 윤곽
            list(range(0, 33)),
            # 2. 눈썹
            list(range(33, 38)), list(range(38, 43)) + [33],
            list(range(43, 48)), list(range(48, 52)) + [43],
            # 3. 콧대
            list(range(52, 57)),
            # 4. 코 밑부분
            list(range(57, 66)),
            # 5. 왼쪽 눈
            list(range(66, 74)) + [66],
            # 6. 오른쪽 눈
            list(range(75, 83)) + [75],
            # 7. 입술 외곽
            list(range(84, 96)) + [84],
            # 8. 입술 안쪽
            list(range(96, 104)) + [96]
        ]

        for face in faces:
            lm = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(int)
            elif face.kps is not None:
                lm = face.kps.astype(int)
            
            if lm is None:
                continue
            
            # 106개가 아니면 Mesh 그리기를 건너뜀 (5개일 경우 점만 찍음)
            if len(lm) != 106:
                for p in lm:
                    cv2.circle(frame, tuple(p), 3, (0, 0, 255), -1)
                continue

            # 1. 모든 점 그리기
            for p in lm:
                cv2.circle(frame, tuple(p), 1, (150, 150, 150), -1)

            # 2. 선 그리기
            for path in connections:
                for i in range(len(path) - 1):
                    if path[i] >= len(lm) or path[i+1] >= len(lm):
                        continue

                    idx1 = path[i]
                    idx2 = path[i+1]
                    pt1 = tuple(lm[idx1])
                    pt2 = tuple(lm[idx2])
                    
                    color = (255, 255, 255)
                    if idx1 < 33: color = (255, 200, 200) # 턱
                    elif 66 <= idx1 <= 83: color = (200, 255, 200) # 눈
                    elif idx1 >= 84: color = (200, 200, 255) # 입
                    
                    cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

        return frame

    def draw_indices_debug(self, frame, faces):
        """
        [New] 인덱스 번호 시각화
        """
        if not faces:
            return frame

        for face in faces:
            lm = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(int)
            elif hasattr(face, 'kps') and face.kps is not None:
                lm = face.kps.astype(int)
            
            if lm is None:
                continue

            for idx, p in enumerate(lm):
                x, y = p
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                
                font_scale = 0.3
                color = (255, 255, 255)
                if idx < 33: color = (200, 200, 255)
                elif 33 <= idx < 52: color = (200, 255, 200)
                elif 52 <= idx < 66: color = (255, 200, 255)
                elif 66 <= idx < 84: color = (255, 255, 0)
                elif idx >= 84: color = (200, 200, 255)

                cv2.putText(frame, str(idx), (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        return frame