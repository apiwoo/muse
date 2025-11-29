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
        """
        print("🧠 [FaceMesh] AI 엔진 로딩 중... (InsightFace)")
        
        # 모델 경로: assets/models/insightface
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=root_dir, 
            allowed_modules=['detection', 'landmark_2d_106', 'genderage'], 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        try:
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ [FaceMesh] 엔진 장전 완료 (CUDA Accelerated)")
        except Exception as e:
            print(f"⚠️ [FaceMesh] 엔진 로딩 실패: {e}")
            self.app = None

    def process(self, frame_bgr):
        if self.app is None or frame_bgr is None:
            return []
        try:
            faces = self.app.get(frame_bgr)
            return faces
        except Exception as e:
            return []

    def draw_debug(self, frame, faces):
        """기본 디버깅 (박스 + 점)"""
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

    def draw_indices_debug(self, frame, faces):
        """
        [New] 인덱스 번호 시각화 (좌표 검증용)
        - 각 랜드마크 위치에 해당 점의 인덱스 번호(0~105)를 텍스트로 적습니다.
        - 이게 보이면 좌표는 정확한데 순서만 문제라는 것을 알 수 있습니다.
        """
        if not faces:
            return frame

        for face in faces:
            # 106개 랜드마크 가져오기
            lm = None
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lm = face.landmark_2d_106.astype(int)
            
            if lm is None or len(lm) != 106:
                continue

            # 모든 점에 번호 적기
            for idx, p in enumerate(lm):
                x, y = p
                
                # 점 찍기 (노란색)
                cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                
                # 번호 적기 (흰색, 아주 작게)
                # 겹침 방지를 위해 5의 배수나 특정 구간만 크게 볼 수도 있지만,
                # 일단은 다 찍어서 확인합니다.
                font_scale = 0.3
                color = (255, 255, 255)
                
                # 주요 부위별로 색상 다르게 (디버깅 용이)
                if idx < 33: color = (200, 200, 255) # 턱 (파랑)
                elif 33 <= idx < 52: color = (200, 255, 200) # 눈썹 (초록)
                elif 52 <= idx < 66: color = (255, 200, 255) # 코 (보라)
                elif 66 <= idx < 84: color = (255, 255, 0) # 눈 (하늘)
                elif idx >= 84: color = (200, 200, 255) # 입 (빨강)

                cv2.putText(frame, str(idx), (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        return frame