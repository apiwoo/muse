# Project MUSE - src/ai/tracker.py
# Created for AI Beauty Cam Project
# (C) 2025 MUSE Corp. All rights reserved.

import cv2
import mediapipe as mp
import numpy as np
from src.utils.logger import get_logger

class FaceTracker:
    def __init__(self):
        self.logger = get_logger("AI_Tracker")
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,      # 동영상 모드 (속도 최적화)
            max_num_faces=1,              # 1명만 추적 (방송용)
            refine_landmarks=True,        # 눈동자(Iris) 디테일 추적 켜기
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.logger.info("🤖 MediaPipe Face Mesh 초기화 완료 (Refine=True)")

    def process(self, frame):
        """
        이미지 프레임을 받아 얼굴 랜드마크를 반환합니다.
        Input: BGR 이미지 (OpenCV 포맷)
        Output: results 객체 (multi_face_landmarks 포함)
        """
        if frame is None:
            return None

        # 1. 색상 변환 (BGR -> RGB)
        # MediaPipe는 RGB 이미지를 사용합니다.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. 성능 최적화를 위해 쓰기 금지 설정 (Pass-by-reference)
        frame_rgb.flags.writeable = False
        
        # 3. 추론 실행
        results = self.face_mesh.process(frame_rgb)
        
        return results

    def draw_debug(self, frame, results):
        """
        디버깅용으로 얼굴에 그물망(Mesh)을 그립니다.
        """
        if not results or not results.multi_face_landmarks:
            return

        # 그리기 도구
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        for face_landmarks in results.multi_face_landmarks:
            # 478개 점 그리기 (테셀레이션)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # 눈, 눈썹 윤곽선 강조
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )