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
        # root_dir이 'assets/models'라면, insightface는 그 하위를 탐색함
        self.app = FaceAnalysis(
            name='buffalo_l', 
            root=root_dir, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # 엔진 준비
        try:
            # det_size: 감지 해상도 (정사각형 권장). 높을수록 작거나 멀리 있는 얼굴 잘 잡음.
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print("✅ [FaceMesh] 엔진 장전 완료 (CUDA Accelerated)")
        except Exception as e:
            print(f"⚠️ [FaceMesh] 엔진 로딩 실패: {e}")
            print("   -> 'tools/download_models.py'를 실행하여 모델을 받았는지 확인하세요.")
            self.app = None

    def process(self, frame_bgr):
        """
        프레임에서 얼굴 랜드마크를 추출합니다.
        Input:
            frame_bgr: Numpy Array (CPU) - BGR 포맷
        Returns:
            faces: 감지된 얼굴 객체 리스트
        """
        if self.app is None or frame_bgr is None:
            return []

        try:
            # InsightFace 추론 (Detection -> Landmark)
            faces = self.app.get(frame_bgr)
            return faces
        except Exception as e:
            # 추론 에러 방어
            # print(f"⚠️ Tracking Error: {e}") 
            return []

    def draw_debug(self, frame, faces):
        """
        [Debug] 얼굴 랜드마크 시각화
        """
        if not faces:
            return frame

        for face in faces:
            # 1. 박스 그리기
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # 2. 랜드마크(106개) 그리기
            if face.kps is not None:
                for p in face.kps:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)
            
            # 3. 정보 표시 (성별, 나이)
            # buffalo_l 모델은 sex, age 속성을 가짐
            gender = 'M' if face.sex == 1 else 'F'
            age = int(face.age)
            label = f"{gender}, {age}"
            
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame