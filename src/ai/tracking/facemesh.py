# Project MUSE - facemesh.py
# Target: MediaPipe Migration (Lightweight & High Detail)
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import numpy as np
import mediapipe as mp

class FaceMesh:
    # [MediaPipe Indices Map]
    # 성형 엔진에서 사용할 부위별 랜드마크 인덱스 매핑 (478 Landmarks 기준)
    FACE_INDICES = {
        # [눈] (왼쪽/오른쪽) - 눈꺼풀 위아래 및 눈꼬리 포함
        "EYE_L": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 144, 145, 153],
        "EYE_R": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390],
        
        # [턱 라인] (V라인 성형용) - 인덱스 보강
        # 귀(234) -> 턱끝(152) -> 귀(454) 이어지는 전체 외곽선
        # 이전보다 포인트를 더 촘촘하게 배치하여 울퉁불퉁함 방지
        "JAW_L": [
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152
        ],
        "JAW_R": [
            454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152
        ],
        
        # [코] (중심점)
        "NOSE_TIP": [1],
    }

    def __init__(self, root_dir=None):
        """
        [Mode A] MediaPipe Face Mesh Engine
        - 장점: 초경량(CPU 가능), 478개 상세 랜드마크
        - 변경: InsightFace(Heavy) -> MediaPipe(Light)
        """
        print("🧠 [FaceMesh] AI 엔진 로딩 중... (MediaPipe)")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # 눈동자(Iris) 추적 포함
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ [FaceMesh] 엔진 장전 완료 (CPU Optimized)")

    class FaceResult:
        """BeautyEngine과 호환성을 위한 결과 래퍼"""
        def __init__(self, landmarks):
            self.landmarks = landmarks # numpy array (478, 2)

    def process(self, frame_bgr):
        """
        :param frame_bgr: 입력 프레임 (BGR)
        :return: [FaceResult] 리스트
        """
        if frame_bgr is None:
            return []

        h, w = frame_bgr.shape[:2]
        
        # 1. BGR -> RGB 변환 (MediaPipe는 RGB를 사용)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. 추론
        results = self.face_mesh.process(frame_rgb)
        
        final_faces = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 3. Normalized(0~1) -> Pixel(x,y) 변환
                # 속도를 위해 numpy 연산 사용
                lm_array = np.array([
                    [lm.x * w, lm.y * h] for lm in face_landmarks.landmark
                ], dtype=int)
                
                final_faces.append(self.FaceResult(lm_array))
                
        return final_faces

    def draw_debug(self, frame, faces):
        """
        [Visual Check] 랜드마크 시각화
        """
        if not faces:
            return frame

        for face in faces:
            # 주요 부위만 점 찍기
            for idx in range(0, len(face.landmarks), 2): # 너무 많으니 2개당 1개만
                pt = face.landmarks[idx]
                cv2.circle(frame, tuple(pt), 1, (100, 255, 100), -1)
                
            # 코 끝 강조
            nose = face.landmarks[1]
            cv2.circle(frame, tuple(nose), 4, (0, 0, 255), -1)

        return frame

    def draw_mesh_debug(self, frame, faces):
        return self.draw_debug(frame, faces)

    def draw_indices_debug(self, frame, faces):
        return self.draw_debug(frame, faces)